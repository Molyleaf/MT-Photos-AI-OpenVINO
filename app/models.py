import asyncio
import gc
import inspect
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import openvino as ov
import yaml
from PIL import Image
from insightface.app import FaceAnalysis
from rapidocr import RapidOCR

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_QA_CLIP_ROOT = _APP_DIR / "QA-CLIP"
_QA_CLIP_CLIP_ROOT = _QA_CLIP_ROOT / "clip"

if str(_QA_CLIP_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QA_CLIP_CLIP_ROOT))

# noinspection PyUnresolvedReferences
from bert_tokenizer import FullTokenizer  # noqa: E402

from schemas import FacialArea, OCRBox, OCRResult, RepresentResult

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
CLIP_INFERENCE_DEVICE = os.environ.get("CLIP_INFERENCE_DEVICE", INFERENCE_DEVICE)

MODEL_NAME = os.environ.get("MODEL_NAME", "antelopv2")
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77
CLIP_IMAGE_RESOLUTION = 224

_CLIP_IMAGE_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32).reshape(1, 1, 3)
_CLIP_IMAGE_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32).reshape(1, 1, 3)
try:
    _PIL_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    _PIL_BICUBIC = Image.BICUBIC

_TOKENIZER = FullTokenizer()
_PAD_TOKEN_ID = int(_TOKENIZER.vocab["[PAD]"])
_CLS_TOKEN_ID = int(_TOKENIZER.vocab["[CLS]"])
_SEP_TOKEN_ID = int(_TOKENIZER.vocab["[SEP]"])

QUEUE_MAX_SIZE = int(os.environ.get("INFERENCE_QUEUE_MAX_SIZE", "64"))
TEXT_BATCH_SIZE = int(os.environ.get("TEXT_CLIP_BATCH_SIZE", "8"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_TASK_TIMEOUT", "120"))
RAPIDOCR_V5_MOBILE_DET_FILE = "ch_PP-OCRv5_mobile_det.onnx"
RAPIDOCR_V5_MOBILE_REC_FILE = "ch_PP-OCRv5_rec_mobile_infer.onnx"
RAPIDOCR_V5_DICT_FILE = "ppocrv5_dict.txt"
RAPIDOCR_CLS_MOBILE_V2_FILE = "ch_ppocr_mobile_v2.0_cls_infer.onnx"

LOG = logging.getLogger(__name__)


def _patch_rapidocr_openvino_multi_output() -> None:
    try:
        from rapidocr.inference_engine.openvino.main import OpenVINOError, OpenVINOInferSession
    except Exception:
        return

    if getattr(OpenVINOInferSession, "_mt_multi_output_patch", False):
        return

    # RapidOCR 3.6.0 OpenVINO session assumes single output; server det models are multi-output.
    def _patched_call(self: Any, input_content: np.ndarray) -> Any:
        try:
            self.session.infer(inputs=[input_content])
            outputs = getattr(self.session, "model_outputs", [])
            if len(outputs) > 1:
                return self.session.get_output_tensor(0).data
            return self.session.get_output_tensor().data
        except Exception as exc:
            error_info = traceback.format_exc()
            raise OpenVINOError(error_info) from exc

    OpenVINOInferSession.__call__ = _patched_call  # type: ignore[assignment]
    OpenVINOInferSession._mt_multi_output_patch = True  # type: ignore[attr-defined]


_patch_rapidocr_openvino_multi_output()


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


TEXT_MODEL_RESTORE_DELAY_MS = max(
    0,
    _as_int(
        os.environ.get(
            "TEXT_MODEL_RESTORE_DELAY_MS",
            os.environ.get("RESTART_TEXT_RESTORE_DELAY_MS", "2000"),
        ),
        2000,
    ),
)


def _format_ov_bool(value: bool) -> str:
    return "YES" if value else "NO"


def _tokenize_for_clip(texts: List[str], context_length: int = CONTEXT_LENGTH) -> np.ndarray:
    if context_length < 2:
        raise ValueError("context_length must be >= 2")

    token_rows: List[List[int]] = []
    for text in texts:
        token_ids = _TOKENIZER.convert_tokens_to_ids(_TOKENIZER.tokenize(text))
        token_ids = token_ids[: context_length - 2]
        row = [_CLS_TOKEN_ID, *token_ids, _SEP_TOKEN_ID]
        token_rows.append(row[:context_length])

    result = np.zeros((len(token_rows), context_length), dtype=np.int64)
    for index, row in enumerate(token_rows):
        result[index, : len(row)] = np.asarray(row, dtype=np.int64)
    return result


def _preprocess_clip_image(
    image: Image.Image,
    image_resolution: int = CLIP_IMAGE_RESOLUTION,
) -> np.ndarray:
    resized = image.convert("RGB").resize((image_resolution, image_resolution), _PIL_BICUBIC)
    pixel_values = np.asarray(resized, dtype=np.float32) / 255.0
    pixel_values = (pixel_values - _CLIP_IMAGE_MEAN) / _CLIP_IMAGE_STD
    chw = np.transpose(pixel_values, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


TaskType = Literal["clip_txt", "clip_img", "ocr", "face", "warmup_text"]


@dataclass(slots=True)
class _InferenceTask:
    kind: TaskType
    payload: Any
    future: Future
    created_at: float


class AIModels:
    """
    Single-worker scheduler with bounded queues.
    - Text CLIP queue has higher priority.
    - Text CLIP tasks are micro-batched inside each worker.
    - Exactly one model family remains resident at a time.
    """

    def __init__(self) -> None:
        self.model_base_path = Path(
            os.environ.get("MODEL_PATH", str(_PROJECT_ROOT / "models"))
        )
        self.insightface_root = self.model_base_path / "insightface"
        self.qa_clip_path = self.model_base_path / "qa-clip" / "openvino"

        self.ov_cache_dir = Path(
            os.environ.get("OV_CACHE_DIR", str(_PROJECT_ROOT / "cache" / "openvino"))
        )
        self.ov_cache_dir.mkdir(parents=True, exist_ok=True)

        self.rapidocr_config_path = Path(
            os.environ.get(
                "RAPIDOCR_OPENVINO_CONFIG_PATH",
                str(_APP_DIR / "config" / "cfg_openvino_cpu.yaml"),
            )
        )
        self.rapidocr_model_dir = os.environ.get(
            "RAPIDOCR_MODEL_DIR", str(self.model_base_path / "rapidocr")
        )
        self.rapidocr_model_dir_path = Path(self.rapidocr_model_dir).expanduser().resolve()
        self.rapidocr_font_path = os.environ.get("RAPIDOCR_FONT_PATH", "")

        self.core = ov.Core()
        self._configure_openvino_cache()

        self._clip_remote_context = self._init_clip_remote_context()

        self._model_lock = threading.Lock()
        self._active_family: Optional[str] = None

        self._clip_text_model: Optional[ov.CompiledModel] = None
        self._clip_text_request: Optional[ov.InferRequest] = None
        self._clip_text_input_names: Optional[Tuple[str, str]] = None
        self._clip_text_host_input_cache: Dict[
            int, Tuple[ov.Tensor, np.ndarray, ov.Tensor, np.ndarray]
        ] = {}
        self._clip_text_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_vision_model: Optional[ov.CompiledModel] = None
        self._clip_vision_request: Optional[ov.InferRequest] = None
        self._clip_vision_input_name: Optional[str] = None
        self._clip_vision_host_tensor: Optional[ov.Tensor] = None
        self._clip_vision_host_view: Optional[np.ndarray] = None
        self._clip_vision_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_image_resolution: Optional[int] = None
        self._rapidocr_engine: Optional[RapidOCR] = None
        self._face_engine: Optional[FaceAnalysis] = None

        self._condition = threading.Condition()
        self._text_queue: Deque[_InferenceTask] = deque()
        self._normal_queue: Deque[_InferenceTask] = deque()
        self._queue_capacity = max(1, QUEUE_MAX_SIZE)
        self._text_batch_size = max(1, TEXT_BATCH_SIZE)
        self._task_timeout_seconds = max(1, TASK_TIMEOUT_SECONDS)
        self._text_restore_delay_seconds = TEXT_MODEL_RESTORE_DELAY_MS / 1000.0
        self._release_requested = False
        self._deferred_text_restore_deadline: Optional[float] = None
        self._stopping = False

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ai-model-worker",
            daemon=True,
        )
        self._worker.start()

        LOG.warning(
            "AIModels ready: clip_device=%s queue=%s text_batch=%s text_restore_delay_ms=%s",
            CLIP_INFERENCE_DEVICE,
            self._queue_capacity,
            self._text_batch_size,
            TEXT_MODEL_RESTORE_DELAY_MS,
        )

    def _configure_openvino_cache(self) -> None:
        try:
            self.core.set_property({"CACHE_DIR": str(self.ov_cache_dir)})
            LOG.warning("OpenVINO cache enabled: %s", self.ov_cache_dir)
        except Exception as exc:
            LOG.warning("Failed to set global OpenVINO cache dir: %s", exc)

    def _init_clip_remote_context(self) -> Optional[Any]:
        if "GPU" not in CLIP_INFERENCE_DEVICE.upper():
            return None
        try:
            remote_context = self.core.get_default_context("GPU")
            LOG.warning("OpenVINO GPU remote context enabled for CLIP.")
            return remote_context
        except Exception as exc:
            LOG.warning("GPU remote context unavailable, fallback to normal tensors: %s", exc)
            return None

    def _queue_size_locked(self) -> int:
        return len(self._text_queue) + len(self._normal_queue)

    def _submit_task(self, kind: TaskType, payload: Any, text_priority: bool) -> Future:
        future: Future = Future()
        task = _InferenceTask(kind=kind, payload=payload, future=future, created_at=time.time())
        with self._condition:
            if self._stopping:
                future.set_exception(RuntimeError("模型服务已关闭"))
                return future
            if self._queue_size_locked() >= self._queue_capacity:
                future.set_exception(
                    RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试")
                )
                return future
            if (
                kind in {"clip_img", "ocr", "face"}
                and self._deferred_text_restore_deadline is not None
            ):
                self._deferred_text_restore_deadline = None
                LOG.warning(
                    "Deferred text restore was cancelled by incoming %s task.", kind
                )
            if text_priority:
                self._text_queue.append(task)
            else:
                self._normal_queue.append(task)
            self._condition.notify()
        return future

    def _wait_future(self, future: Future) -> Any:
        try:
            return future.result(timeout=self._task_timeout_seconds)
        except FutureTimeoutError as exc:
            raise RuntimeError(
                f"推理任务超时（>{self._task_timeout_seconds}s）"
            ) from exc

    async def _await_future(self, future: Future) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=self._task_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"推理任务超时（>{self._task_timeout_seconds}s）"
            ) from exc

    def _safe_set_result(self, future: Future, value: Any) -> None:
        if not future.done():
            future.set_result(value)

    def _safe_set_exception(self, future: Future, exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

    def _worker_loop(self) -> None:
        while True:
            maintenance_action: Optional[str] = None
            text_batch: List[_InferenceTask] = []
            normal_task: Optional[_InferenceTask] = None

            with self._condition:
                while True:
                    if self._stopping:
                        break

                    has_pending_tasks = bool(self._text_queue or self._normal_queue)
                    if has_pending_tasks or self._release_requested:
                        break

                    if self._deferred_text_restore_deadline is None:
                        self._condition.wait(timeout=0.25)
                        continue

                    remaining = self._deferred_text_restore_deadline - time.time()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=min(0.25, remaining))

                if self._stopping:
                    break

                now = time.time()
                if (
                    self._release_requested
                    and not self._text_queue
                    and not self._normal_queue
                ):
                    self._release_requested = False
                    maintenance_action = "release_only"
                    if (
                        self._deferred_text_restore_deadline is not None
                        and self._deferred_text_restore_deadline <= now
                    ):
                        self._deferred_text_restore_deadline = None
                        maintenance_action = "release_and_restore_text"
                elif (
                    not self._text_queue
                    and not self._normal_queue
                    and self._deferred_text_restore_deadline is not None
                    and self._deferred_text_restore_deadline <= now
                ):
                    self._deferred_text_restore_deadline = None
                    maintenance_action = "restore_text"
                elif self._text_queue:
                    text_batch.append(self._text_queue.popleft())
                    while self._text_queue and len(text_batch) < self._text_batch_size:
                        text_batch.append(self._text_queue.popleft())
                elif self._normal_queue:
                    normal_task = self._normal_queue.popleft()

            if maintenance_action == "release_only":
                try:
                    self._unload_active_family()
                except Exception as exc:
                    LOG.error("Failed to release active model family: %s", exc, exc_info=True)
                continue

            if maintenance_action == "release_and_restore_text":
                try:
                    self._unload_active_family()
                    self._switch_family("text")
                except Exception as exc:
                    LOG.error("Failed to complete idle release maintenance: %s", exc, exc_info=True)
                continue

            if maintenance_action == "restore_text":
                try:
                    self._switch_family("text")
                except Exception as exc:
                    LOG.error("Failed to restore text model after defer window: %s", exc, exc_info=True)
                continue

            if text_batch:
                self._handle_text_batch(text_batch)
                self._return_to_text_if_idle()
                continue

            if normal_task is not None:
                self._handle_single_task(normal_task)
                self._return_to_text_if_idle()

    def _return_to_text_if_idle(self) -> None:
        restore_immediately = False
        with self._condition:
            if self._stopping or self._release_requested:
                return
            if self._text_queue or self._normal_queue:
                return
            if self._active_family == "text":
                self._deferred_text_restore_deadline = None
                return
            if self._deferred_text_restore_deadline is not None:
                return
            if self._text_restore_delay_seconds <= 0:
                restore_immediately = True
            else:
                self._deferred_text_restore_deadline = (
                    time.time() + self._text_restore_delay_seconds
                )
                self._condition.notify()

        if restore_immediately:
            try:
                self._switch_family("text")
            except Exception as exc:
                LOG.error("Failed to restore text model while idle: %s", exc, exc_info=True)

    def _handle_text_batch(self, tasks: List[_InferenceTask]) -> None:
        try:
            texts: List[str] = []
            for task in tasks:
                if task.kind != "clip_txt":
                    raise RuntimeError(f"Invalid text batch task kind: {task.kind}")
                texts.append(str(task.payload))
            self._switch_family("text")
            embeddings = self._infer_clip_text_batch(texts)
            if len(embeddings) != len(tasks):
                raise RuntimeError(
                    f"Text batch output mismatch: expected={len(tasks)} got={len(embeddings)}"
                )
            for index, task in enumerate(tasks):
                self._safe_set_result(task.future, embeddings[index])
        except Exception as exc:
            LOG.error("Text batch inference failed: %s", exc, exc_info=True)
            for task in tasks:
                self._safe_set_exception(task.future, exc)

    def _handle_single_task(self, task: _InferenceTask) -> None:
        try:
            if task.kind == "clip_img":
                image, filename = task.payload
                self._switch_family("vision")
                result = self._infer_clip_image(image, filename=filename)
                self._safe_set_result(task.future, result)
                return

            if task.kind == "ocr":
                self._switch_family("ocr")
                self._safe_set_result(task.future, self._infer_ocr(task.payload))
                return

            if task.kind == "face":
                self._switch_family("face")
                self._safe_set_result(task.future, self._infer_face(task.payload))
                return

            if task.kind == "warmup_text":
                self._switch_family("text")
                self._safe_set_result(task.future, True)
                return

            raise RuntimeError(f"Unsupported task kind: {task.kind}")
        except Exception as exc:
            LOG.error("Task %s failed: %s", task.kind, exc, exc_info=True)
            self._safe_set_exception(task.future, exc)

    def _unload_active_family(self) -> None:
        with self._model_lock:
            if self._active_family is None:
                return
            self._unload_active_family_locked()

    def _switch_family(self, target_family: str) -> None:
        with self._model_lock:
            if self._active_family == target_family:
                return

            self._unload_active_family_locked()

            if target_family == "text":
                self._load_clip_text_locked()
            elif target_family == "vision":
                self._load_clip_vision_locked()
            elif target_family == "ocr":
                self._load_rapidocr_locked()
            elif target_family == "face":
                self._load_face_locked()
            else:
                raise RuntimeError(f"Unsupported model family: {target_family}")

            self._active_family = target_family

    def _unload_active_family_locked(self) -> None:
        if self._active_family == "text":
            self._clip_text_request = None
            self._clip_text_model = None
            self._clip_text_input_names = None
            self._clip_text_host_input_cache.clear()
        elif self._active_family == "vision":
            self._clip_vision_request = None
            self._clip_vision_model = None
            self._clip_vision_input_name = None
            self._clip_vision_host_tensor = None
            self._clip_vision_host_view = None
            self._clip_image_resolution = None
        elif self._active_family == "ocr":
            self._rapidocr_engine = None
        elif self._active_family == "face":
            self._face_engine = None

        self._active_family = None
        gc.collect()

    def _unload_everything_locked(self) -> None:
        self._clip_text_request = None
        self._clip_text_model = None
        self._clip_text_input_names = None
        self._clip_text_host_input_cache.clear()
        self._clip_vision_request = None
        self._clip_vision_model = None
        self._clip_vision_input_name = None
        self._clip_vision_host_tensor = None
        self._clip_vision_host_view = None
        self._clip_image_resolution = None
        self._rapidocr_engine = None
        self._face_engine = None
        self._active_family = None
        gc.collect()

    def _compile_clip_model(
        self,
        model_path: Path,
        performance_hint: str,
    ) -> ov.CompiledModel:
        config = {
            "PERFORMANCE_HINT": performance_hint,
            "CACHE_DIR": str(self.ov_cache_dir),
        }

        if self._clip_remote_context is not None:
            try:
                model = self.core.read_model(str(model_path))
                return self.core.compile_model(model, self._clip_remote_context, config)
            except Exception as exc:
                LOG.warning("Remote context compile failed, fallback to device compile: %s", exc)

        return self.core.compile_model(str(model_path), CLIP_INFERENCE_DEVICE, config)

    def _get_text_host_tensors(
        self, batch_size: int
    ) -> Optional[Tuple[ov.Tensor, np.ndarray, ov.Tensor, np.ndarray]]:
        if (
            not self._clip_text_host_tensor_enabled
            or self._clip_remote_context is None
            or self._clip_text_model is None
        ):
            return None

        cached = self._clip_text_host_input_cache.get(batch_size)
        if cached is not None:
            return cached

        try:
            tensor_shape = ov.Shape([int(batch_size), int(CONTEXT_LENGTH)])
            input_0 = self._clip_text_model.inputs[0]
            input_1 = self._clip_text_model.inputs[1]
            input_tensor_0 = self._clip_remote_context.create_host_tensor(
                input_0.get_element_type(), tensor_shape
            )
            input_tensor_1 = self._clip_remote_context.create_host_tensor(
                input_1.get_element_type(), tensor_shape
            )
            input_view_0 = np.asarray(input_tensor_0.data)
            input_view_1 = np.asarray(input_tensor_1.data)
            entry = (input_tensor_0, input_view_0, input_tensor_1, input_view_1)
            self._clip_text_host_input_cache[batch_size] = entry
            return entry
        except Exception as exc:
            LOG.warning(
                "CLIP text host tensor allocation failed, fallback to shared numpy inputs: %s",
                exc,
            )
            self._clip_text_host_tensor_enabled = False
            self._clip_text_host_input_cache.clear()
            return None

    def _load_clip_text_locked(self) -> None:
        text_model_path = self.qa_clip_path / "openvino_text_fp16.xml"
        if not text_model_path.exists():
            raise FileNotFoundError(f"Missing text model: {text_model_path}")

        compiled_model = self._compile_clip_model(
            model_path=text_model_path,
            performance_hint="LATENCY",
        )
        output_dim = compiled_model.outputs[0].get_partial_shape()[1].get_length()
        if output_dim != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Text embedding dims mismatch: expected={CLIP_EMBEDDING_DIMS}, got={output_dim}"
            )

        self._clip_text_model = compiled_model
        self._clip_text_request = compiled_model.create_infer_request()
        self._clip_text_input_names = (
            self._clip_text_model.inputs[0].any_name,
            self._clip_text_model.inputs[1].any_name,
        )
        self._clip_text_host_input_cache.clear()
        self._clip_text_host_tensor_enabled = self._clip_remote_context is not None
        LOG.warning("CLIP Text model loaded on %s.", CLIP_INFERENCE_DEVICE)

    def _load_clip_vision_locked(self) -> None:
        vision_model_path = self.qa_clip_path / "openvino_image_fp16.xml"
        if not vision_model_path.exists():
            raise FileNotFoundError(f"Missing vision model: {vision_model_path}")

        compiled_model = self._compile_clip_model(
            model_path=vision_model_path,
            performance_hint="THROUGHPUT",
        )
        output_dim = compiled_model.outputs[0].get_partial_shape()[1].get_length()
        if output_dim != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Vision embedding dims mismatch: expected={CLIP_EMBEDDING_DIMS}, got={output_dim}"
            )

        self._clip_vision_model = compiled_model
        self._clip_vision_request = compiled_model.create_infer_request()
        self._clip_vision_input_name = self._clip_vision_model.inputs[0].any_name
        self._clip_vision_host_tensor = None
        self._clip_vision_host_view = None
        self._clip_vision_host_tensor_enabled = self._clip_remote_context is not None
        if self._clip_vision_host_tensor_enabled and self._clip_remote_context is not None:
            try:
                input_port = self._clip_vision_model.inputs[0]
                # Vision input may expose dynamic batch dimension; use fixed single-image shape.
                input_shape = ov.Shape(
                    [1, 3, int(CLIP_IMAGE_RESOLUTION), int(CLIP_IMAGE_RESOLUTION)]
                )
                self._clip_vision_host_tensor = self._clip_remote_context.create_host_tensor(
                    input_port.get_element_type(),
                    input_shape,
                )
                self._clip_vision_host_view = np.asarray(self._clip_vision_host_tensor.data)
                LOG.warning("CLIP Vision host tensor path enabled for GPU context.")
            except Exception as exc:
                LOG.warning(
                    "CLIP Vision host tensor unavailable, fallback to shared numpy inputs: %s",
                    exc,
                )
                self._clip_vision_host_tensor = None
                self._clip_vision_host_view = None
                self._clip_vision_host_tensor_enabled = False
        self._clip_image_resolution = CLIP_IMAGE_RESOLUTION
        LOG.warning("CLIP Vision model loaded on %s.", CLIP_INFERENCE_DEVICE)

    def _load_rapidocr_openvino_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "inference_num_threads": _as_int(os.environ.get("RAPIDOCR_INFERENCE_NUM_THREADS"), 8),
            "performance_hint": os.environ.get("RAPIDOCR_PERFORMANCE_HINT", "LATENCY"),
            "performance_num_requests": _as_int(
                os.environ.get("RAPIDOCR_PERFORMANCE_NUM_REQUESTS"), -1
            ),
            "enable_cpu_pinning": _as_bool(os.environ.get("RAPIDOCR_ENABLE_CPU_PINNING"), True),
            "num_streams": _as_int(os.environ.get("RAPIDOCR_NUM_STREAMS"), -1),
            "enable_hyper_threading": _as_bool(
                os.environ.get("RAPIDOCR_ENABLE_HYPER_THREADING"), True
            ),
            "scheduling_core_type": os.environ.get("RAPIDOCR_SCHEDULING_CORE_TYPE", "ANY_CORE"),
        }

        if self.rapidocr_config_path.exists():
            try:
                loaded = yaml.safe_load(self.rapidocr_config_path.read_text(encoding="utf-8")) or {}
                ov_cfg = loaded.get("EngineConfig", {}).get("openvino", {})
                for key in list(config.keys()):
                    if key in ov_cfg:
                        config[key] = ov_cfg[key]
                LOG.warning("RapidOCR OpenVINO config loaded: %s", self.rapidocr_config_path)
            except Exception as exc:
                LOG.warning("Unable to parse RapidOCR config %s: %s", self.rapidocr_config_path, exc)

        config["cache_dir"] = str(self.ov_cache_dir)
        return config

    def _to_openvino_plugin_config(self, cfg: Dict[str, Any]) -> Dict[str, str]:
        return {
            "INFERENCE_NUM_THREADS": str(_as_int(cfg.get("inference_num_threads"), 8)),
            "PERFORMANCE_HINT": str(cfg.get("performance_hint", "LATENCY")),
            "PERFORMANCE_NUM_REQUESTS": str(_as_int(cfg.get("performance_num_requests"), -1)),
            "ENABLE_CPU_PINNING": _format_ov_bool(_as_bool(cfg.get("enable_cpu_pinning"), True)),
            "NUM_STREAMS": str(_as_int(cfg.get("num_streams"), -1)),
            "ENABLE_HYPER_THREADING": _format_ov_bool(
                _as_bool(cfg.get("enable_hyper_threading"), True)
            ),
            "SCHEDULING_CORE_TYPE": str(cfg.get("scheduling_core_type", "ANY_CORE")),
            "CACHE_DIR": str(cfg.get("cache_dir", self.ov_cache_dir)),
        }

    def _resolve_rapidocr_local_asset(self, filename: str) -> Optional[Path]:
        candidate = self.rapidocr_model_dir_path / filename
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def _build_rapidocr_runtime_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "EngineConfig.openvino.inference_num_threads": _as_int(
                cfg.get("inference_num_threads"), -1
            ),
            "EngineConfig.openvino.performance_hint": str(
                cfg.get("performance_hint", "LATENCY")
            ),
            "EngineConfig.openvino.performance_num_requests": _as_int(
                cfg.get("performance_num_requests"), -1
            ),
            "EngineConfig.openvino.enable_cpu_pinning": _as_bool(
                cfg.get("enable_cpu_pinning"), True
            ),
            "EngineConfig.openvino.num_streams": _as_int(cfg.get("num_streams"), -1),
            "EngineConfig.openvino.enable_hyper_threading": _as_bool(
                cfg.get("enable_hyper_threading"), True
            ),
            "EngineConfig.openvino.scheduling_core_type": str(
                cfg.get("scheduling_core_type", "ANY_CORE")
            ),
        }

        if self.rapidocr_font_path:
            params["Global.font_path"] = self.rapidocr_font_path

        det_model_path = self._resolve_rapidocr_local_asset(RAPIDOCR_V5_MOBILE_DET_FILE)
        rec_model_path = self._resolve_rapidocr_local_asset(RAPIDOCR_V5_MOBILE_REC_FILE)
        rec_dict_path = self._resolve_rapidocr_local_asset(RAPIDOCR_V5_DICT_FILE)
        cls_model_path = self._resolve_rapidocr_local_asset(RAPIDOCR_CLS_MOBILE_V2_FILE)

        if det_model_path:
            params["Det.model_path"] = str(det_model_path)
        else:
            LOG.warning(
                "RapidOCR det model not found at %s; fallback to RapidOCR downloader.",
                self.rapidocr_model_dir_path / RAPIDOCR_V5_MOBILE_DET_FILE,
            )

        if rec_model_path:
            params["Rec.model_path"] = str(rec_model_path)
        else:
            LOG.warning(
                "RapidOCR rec model not found at %s; fallback to RapidOCR downloader.",
                self.rapidocr_model_dir_path / RAPIDOCR_V5_MOBILE_REC_FILE,
            )

        if rec_dict_path:
            params["Rec.rec_keys_path"] = str(rec_dict_path)
        else:
            LOG.warning(
                "RapidOCR rec dict not found at %s; fallback to RapidOCR downloader.",
                self.rapidocr_model_dir_path / RAPIDOCR_V5_DICT_FILE,
            )

        if cls_model_path:
            # RapidOCR initializes cls session even when Global.use_cls=false.
            params["Cls.model_path"] = str(cls_model_path)
        else:
            LOG.warning(
                "RapidOCR cls model not found at %s; fallback to RapidOCR downloader.",
                self.rapidocr_model_dir_path / RAPIDOCR_CLS_MOBILE_V2_FILE,
            )

        return params

    @staticmethod
    def _constructor_filter_kwargs(kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            signature = inspect.signature(RapidOCR)
            parameters = signature.parameters
            has_var_kwargs = any(
                item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
            )
            if has_var_kwargs:
                return [kwargs]

            filtered = {k: v for k, v in kwargs.items() if k in parameters}
            if filtered:
                return [filtered]
            return []
        except (TypeError, ValueError):
            return [kwargs]

    def _instantiate_rapidocr(self, ov_cfg: Dict[str, str], params: Dict[str, Any]) -> RapidOCR:
        model_dir_value = self.rapidocr_model_dir if self.rapidocr_model_dir else None
        font_path_value = self.rapidocr_font_path if self.rapidocr_font_path else None

        common_values: Dict[str, Any] = {
            "device": "CPU",
            "device_name": "CPU",
            "ov_config": ov_cfg,
            "openvino_config": ov_cfg,
            "inference_engine": "openvino",
            "engine_type": "openvino",
            "det_engine": "openvino",
            "cls_engine": "openvino",
            "rec_engine": "openvino",
        }
        if model_dir_value:
            common_values.update(
                {
                    "model_dir": model_dir_value,
                    "models_dir": model_dir_value,
                    "model_path": model_dir_value,
                }
            )
        if font_path_value:
            common_values["vis_font_path"] = font_path_value

        candidate_kwargs: List[Dict[str, Any]] = []
        if self.rapidocr_config_path.exists():
            config_path_text = str(self.rapidocr_config_path)
            candidate_kwargs.extend(
                [
                    {"config_path": config_path_text, "params": params},
                    {"params_path": config_path_text, "params": params},
                    {"cfg_path": config_path_text, "params": params},
                    {"config_path": config_path_text},
                ]
            )
        elif params:
            candidate_kwargs.append({"params": params})

        candidate_kwargs.extend(
            [
                {
                    "det_engine": common_values["det_engine"],
                    "cls_engine": common_values["cls_engine"],
                    "rec_engine": common_values["rec_engine"],
                    "device_name": common_values["device_name"],
                    "ov_config": common_values["ov_config"],
                    "model_dir": model_dir_value,
                    "vis_font_path": font_path_value,
                },
                {
                    "engine_type": common_values["engine_type"],
                    "device": common_values["device"],
                    "openvino_config": common_values["openvino_config"],
                    "model_dir": model_dir_value,
                    "vis_font_path": font_path_value,
                },
                {
                    "device_name": common_values["device_name"],
                    "ov_config": common_values["ov_config"],
                    "model_dir": model_dir_value,
                    "vis_font_path": font_path_value,
                },
            ]
        )

        last_error: Optional[Exception] = None
        for kwargs in candidate_kwargs:
            compact_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            for filtered_kwargs in self._constructor_filter_kwargs(compact_kwargs):
                try:
                    return RapidOCR(**filtered_kwargs)
                except Exception as exc:
                    last_error = exc

        raise RuntimeError(
            "RapidOCR 初始化失败，无法以 OpenVINO CPU 配置启动。"
        ) from last_error

    def _load_rapidocr_locked(self) -> None:
        config = self._load_rapidocr_openvino_config()
        ov_cfg = self._to_openvino_plugin_config(config)
        rapidocr_params = self._build_rapidocr_runtime_params(config)
        self._rapidocr_engine = self._instantiate_rapidocr(ov_cfg, rapidocr_params)
        LOG.warning("RapidOCR loaded with OpenVINO CPU backend.")

    def _load_face_locked(self) -> None:
        provider_device = os.environ.get("INSIGHTFACE_OV_DEVICE", "CPU_FP32")
        providers: List[Any] = [
            ("OpenVINOExecutionProvider", {"device_type": provider_device}),
            "CPUExecutionProvider",
        ]
        try:
            face_app = FaceAnalysis(
                name=MODEL_NAME,
                root=str(self.insightface_root),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_engine = face_app
            LOG.warning("InsightFace loaded with providers=%s", providers)
        except Exception as ov_exc:
            LOG.warning("OpenVINO EP unavailable for InsightFace, fallback to CPU EP: %s", ov_exc)
            fallback = FaceAnalysis(
                name=MODEL_NAME,
                root=str(self.insightface_root),
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            fallback.prepare(ctx_id=0, det_size=(640, 640))
            self._face_engine = fallback
            LOG.warning("InsightFace fallback providers=['CPUExecutionProvider']")

    def _infer_clip_text_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._clip_text_model or not self._clip_text_request:
            raise RuntimeError("CLIP text model is not loaded.")
        if self._clip_text_input_names is None:
            raise RuntimeError("CLIP text input metadata is not initialized.")

        input_ids = _tokenize_for_clip(texts, context_length=CONTEXT_LENGTH)
        attention_mask = np.array(input_ids != _PAD_TOKEN_ID, dtype=np.int64)
        input_name_0, input_name_1 = self._clip_text_input_names

        host_tensors = self._get_text_host_tensors(batch_size=input_ids.shape[0])
        if host_tensors is not None:
            input_tensor_0, input_view_0, input_tensor_1, input_view_1 = host_tensors
            np.copyto(input_view_0, input_ids, casting="no")
            np.copyto(input_view_1, attention_mask, casting="no")
            self._clip_text_request.set_input_tensor(0, input_tensor_0)
            self._clip_text_request.set_input_tensor(1, input_tensor_1)
            self._clip_text_request.infer()
            embeddings = np.asarray(self._clip_text_request.get_output_tensor(0).data)
        else:
            outputs = self._clip_text_request.infer(
                {
                    input_name_0: input_ids,
                    input_name_1: attention_mask,
                },
                share_inputs=True,
                share_outputs=True,
            )
            embeddings = np.asarray(outputs[self._clip_text_model.outputs[0]])
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[-1] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid text embedding dims: expected={CLIP_EMBEDDING_DIMS}, got={embeddings.shape[-1]}"
            )
        return embeddings.astype(np.float32, copy=False).tolist()

    def _infer_clip_image(self, image: Image.Image, filename: str) -> List[float]:
        if (
            not self._clip_vision_model
            or not self._clip_vision_request
            or self._clip_image_resolution is None
        ):
            raise RuntimeError("CLIP vision model is not loaded.")

        pixel_values = _preprocess_clip_image(
            image=image,
            image_resolution=self._clip_image_resolution,
        )

        if self._clip_vision_host_tensor is not None and self._clip_vision_host_view is not None:
            np.copyto(self._clip_vision_host_view, pixel_values, casting="no")
            self._clip_vision_request.set_input_tensor(0, self._clip_vision_host_tensor)
            self._clip_vision_request.infer()
            embedding = np.asarray(self._clip_vision_request.get_output_tensor(0).data).reshape(-1)
        else:
            if not self._clip_vision_input_name:
                raise RuntimeError("CLIP vision input metadata is not initialized.")
            outputs = self._clip_vision_request.infer(
                {self._clip_vision_input_name: pixel_values},
                share_inputs=True,
                share_outputs=True,
            )
            embedding = np.asarray(outputs[self._clip_vision_model.outputs[0]]).reshape(-1)
        if embedding.shape[0] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid image embedding dims for {filename}: "
                f"expected={CLIP_EMBEDDING_DIMS}, got={embedding.shape[0]}"
            )
        return embedding.astype(np.float32, copy=False).tolist()

    def _infer_ocr(self, image: np.ndarray) -> OCRResult:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")

        raw_result = self._rapidocr_engine(image)
        ocr_items: Any
        if all(hasattr(raw_result, field) for field in ("boxes", "txts", "scores")):
            boxes_data = getattr(raw_result, "boxes", None)
            txts_data = getattr(raw_result, "txts", None) or ()
            scores_data = getattr(raw_result, "scores", None) or ()
            if boxes_data is None:
                ocr_items = []
            else:
                count = min(len(boxes_data), len(txts_data), len(scores_data))
                ocr_items = [
                    (boxes_data[idx], txts_data[idx], scores_data[idx]) for idx in range(count)
                ]
        elif isinstance(raw_result, tuple):
            ocr_items = raw_result[0]
        else:
            ocr_items = raw_result

        if not ocr_items:
            return OCRResult(texts=[], scores=[], boxes=[])

        texts: List[str] = []
        scores: List[str] = []
        boxes: List[OCRBox] = []

        for item in ocr_items:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            points = np.array(item[0], dtype=np.float32)
            if points.shape != (4, 2):
                continue
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            boxes.append(
                OCRBox(
                    x=str(round(float(x_min), 2)),
                    y=str(round(float(y_min), 2)),
                    width=str(round(float(x_max - x_min), 2)),
                    height=str(round(float(y_max - y_min), 2)),
                )
            )
            texts.append(str(item[1]))
            scores.append(f"{float(item[2]):.2f}")

        return OCRResult(texts=texts, scores=scores, boxes=boxes)

    def _infer_face(self, image: np.ndarray) -> List[RepresentResult]:
        if self._face_engine is None:
            raise RuntimeError("Face model is not loaded.")

        faces = self._face_engine.get(image)
        if not faces:
            return []

        results: List[RepresentResult] = []
        for face in faces:
            bbox = np.array(face.bbox).astype(int)
            x1, y1, x2, y2 = bbox
            results.append(
                RepresentResult(
                    embedding=[float(value) for value in face.normed_embedding],
                    facial_area=FacialArea(
                        x=int(x1),
                        y=int(y1),
                        w=int(x2 - x1),
                        h=int(y2 - y1),
                    ),
                    face_confidence=float(face.det_score),
                )
            )
        return results

    def ensure_clip_text_model_loaded(self) -> None:
        future = self._submit_task(kind="warmup_text", payload=None, text_priority=False)
        self._wait_future(future)

    async def ensure_clip_text_model_loaded_async(self) -> None:
        future = self._submit_task(kind="warmup_text", payload=None, text_priority=False)
        await self._await_future(future)

    def release_models(self) -> None:
        self._schedule_release(text_restore_delay_seconds=0.0)

    def release_models_for_restart(self, text_restore_delay_seconds: Optional[float] = None) -> None:
        restore_delay_seconds = (
            self._text_restore_delay_seconds
            if text_restore_delay_seconds is None
            else max(0.0, float(text_restore_delay_seconds))
        )
        self._schedule_release(text_restore_delay_seconds=restore_delay_seconds)

    def _schedule_release(self, text_restore_delay_seconds: float) -> None:
        delay_seconds = max(0.0, float(text_restore_delay_seconds))
        with self._condition:
            if self._stopping:
                return
            self._release_requested = True
            self._deferred_text_restore_deadline = time.time() + delay_seconds
            self._condition.notify()

    def release_all_models(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            pending_tasks = list(self._text_queue) + list(self._normal_queue)
            self._text_queue.clear()
            self._normal_queue.clear()
            self._condition.notify_all()

        for task in pending_tasks:
            self._safe_set_exception(task.future, RuntimeError("模型服务正在关闭"))

        self._worker.join()

        with self._model_lock:
            self._unload_everything_locked()
        LOG.warning("All models released.")

    def get_text_embedding(self, text: str) -> List[float]:
        future = self._submit_task(kind="clip_txt", payload=text, text_priority=True)
        return self._wait_future(future)

    async def get_text_embedding_async(self, text: str) -> List[float]:
        future = self._submit_task(kind="clip_txt", payload=text, text_priority=True)
        return await self._await_future(future)

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        payload = (image, filename)
        future = self._submit_task(kind="clip_img", payload=payload, text_priority=False)
        return self._wait_future(future)

    async def get_image_embedding_async(
        self, image: Image.Image, filename: str = "unknown"
    ) -> List[float]:
        payload = (image, filename)
        future = self._submit_task(kind="clip_img", payload=payload, text_priority=False)
        return await self._await_future(future)

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        future = self._submit_task(kind="ocr", payload=image, text_priority=False)
        return self._wait_future(future)

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        future = self._submit_task(kind="ocr", payload=image, text_priority=False)
        return await self._await_future(future)

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        future = self._submit_task(kind="face", payload=image, text_priority=False)
        return self._wait_future(future)

    async def get_face_representation_async(self, image: np.ndarray) -> List[RepresentResult]:
        future = self._submit_task(kind="face", payload=image, text_priority=False)
        return await self._await_future(future)
