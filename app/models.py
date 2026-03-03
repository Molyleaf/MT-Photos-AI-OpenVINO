import asyncio
import gc
import inspect
import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

import numpy as np
import openvino as ov
import yaml
from PIL import Image
from insightface.app import FaceAnalysis
from rapidocr import RapidOCR

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_QA_CLIP_ROOT = _APP_DIR / "QA-CLIP"

if str(_QA_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QA_CLIP_ROOT))

import clip  # noqa: E402
try:  # noqa: E402
    from clip.utils import MODEL_INFO, image_transform  # noqa: E402
except ImportError:  # noqa: E402
    from clip.utils import _MODEL_INFO as MODEL_INFO, image_transform  # noqa: E402

from schemas import FacialArea, OCRBox, OCRResult, RepresentResult

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "GPU")
CLIP_INFERENCE_DEVICE = os.environ.get("CLIP_INFERENCE_DEVICE", INFERENCE_DEVICE)

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

QUEUE_MAX_SIZE = int(os.environ.get("INFERENCE_QUEUE_MAX_SIZE", "64"))
TEXT_BATCH_SIZE = int(os.environ.get("TEXT_CLIP_BATCH_SIZE", "8"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_TASK_TIMEOUT", "120"))

LOG = logging.getLogger(__name__)


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


def _format_ov_bool(value: bool) -> str:
    return "YES" if value else "NO"


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
                str(_PROJECT_ROOT / "example" / "cfg_openvino_cpu.yaml"),
            )
        )
        self.rapidocr_model_dir = os.environ.get(
            "RAPIDOCR_MODEL_DIR", str(self.model_base_path / "rapidocr")
        )
        self.rapidocr_font_path = os.environ.get("RAPIDOCR_FONT_PATH", "")

        self.core = ov.Core()
        self._configure_openvino_cache()

        self._clip_remote_context = self._init_clip_remote_context()

        self._model_lock = threading.Lock()
        self._active_family: Optional[str] = None

        self._clip_text_model: Optional[ov.CompiledModel] = None
        self._clip_text_request: Optional[ov.InferRequest] = None
        self._clip_vision_model: Optional[ov.CompiledModel] = None
        self._clip_vision_request: Optional[ov.InferRequest] = None
        self._clip_image_preprocessor = None
        self._rapidocr_engine: Optional[RapidOCR] = None
        self._face_engine: Optional[FaceAnalysis] = None

        self._condition = threading.Condition()
        self._text_queue: Deque[_InferenceTask] = deque()
        self._normal_queue: Deque[_InferenceTask] = deque()
        self._queue_capacity = max(1, QUEUE_MAX_SIZE)
        self._text_batch_size = max(1, TEXT_BATCH_SIZE)
        self._task_timeout_seconds = max(1, TASK_TIMEOUT_SECONDS)
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
            "AIModels ready: clip_device=%s queue=%s text_batch=%s",
            CLIP_INFERENCE_DEVICE,
            self._queue_capacity,
            self._text_batch_size,
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
                and self._deferred_text_restore_deadline > time.time()
            ):
                self._deferred_text_restore_deadline = time.time()
                LOG.warning(
                    "Deferred text restore was accelerated by incoming %s task.", kind
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
        with self._condition:
            has_pending = bool(
                self._text_queue or self._normal_queue or self._stopping or self._release_requested
            )
            has_deferred_restore = self._deferred_text_restore_deadline is not None
        if not has_pending and not has_deferred_restore:
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
        elif self._active_family == "vision":
            self._clip_vision_request = None
            self._clip_vision_model = None
            self._clip_image_preprocessor = None
        elif self._active_family == "ocr":
            self._rapidocr_engine = None
        elif self._active_family == "face":
            self._face_engine = None

        self._active_family = None
        gc.collect()

    def _unload_everything_locked(self) -> None:
        self._clip_text_request = None
        self._clip_text_model = None
        self._clip_vision_request = None
        self._clip_vision_model = None
        self._clip_image_preprocessor = None
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
                return self.core.compile_model(str(model_path), self._clip_remote_context, config)
            except Exception as exc:
                LOG.warning("Remote context compile failed, fallback to device compile: %s", exc)

        return self.core.compile_model(str(model_path), CLIP_INFERENCE_DEVICE, config)

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
        self._clip_image_preprocessor = image_transform(MODEL_INFO[MODEL_ARCH]["input_resolution"])
        LOG.warning("CLIP Vision model loaded on %s.", CLIP_INFERENCE_DEVICE)

    def _load_rapidocr_server_config(self) -> Dict[str, Any]:
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

    def _instantiate_rapidocr(self, ov_cfg: Dict[str, str]) -> RapidOCR:
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
                    {"params_path": config_path_text},
                    {"config_path": config_path_text},
                    {"cfg_path": config_path_text},
                ]
            )

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
        config = self._load_rapidocr_server_config()
        ov_cfg = self._to_openvino_plugin_config(config)
        self._rapidocr_engine = self._instantiate_rapidocr(ov_cfg)
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

        input_ids = clip.tokenize(texts, context_length=CONTEXT_LENGTH).numpy()
        pad_token_id = clip._tokenizer.vocab["[PAD]"]
        attention_mask = np.array(input_ids != pad_token_id, dtype=np.int64)

        input_name_0 = self._clip_text_model.inputs[0].any_name
        input_name_1 = self._clip_text_model.inputs[1].any_name

        outputs = self._clip_text_request.infer(
            {
                input_name_0: input_ids,
                input_name_1: attention_mask,
            }
        )
        embeddings = np.asarray(outputs[self._clip_text_model.outputs[0]])
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[-1] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid text embedding dims: expected={CLIP_EMBEDDING_DIMS}, got={embeddings.shape[-1]}"
            )
        return [[float(value) for value in row] for row in embeddings]

    def _infer_clip_image(self, image: Image.Image, filename: str) -> List[float]:
        if not self._clip_vision_model or not self._clip_vision_request or not self._clip_image_preprocessor:
            raise RuntimeError("CLIP vision model is not loaded.")

        inputs = self._clip_image_preprocessor(image).unsqueeze(0)
        pixel_values = inputs.numpy()

        outputs = self._clip_vision_request.infer(
            {self._clip_vision_model.inputs[0].any_name: pixel_values}
        )
        embedding = np.asarray(outputs[self._clip_vision_model.outputs[0]]).reshape(-1)
        if embedding.shape[0] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid image embedding dims for {filename}: "
                f"expected={CLIP_EMBEDDING_DIMS}, got={embedding.shape[0]}"
            )
        return [float(value) for value in embedding]

    def _infer_ocr(self, image: np.ndarray) -> OCRResult:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")

        raw_result = self._rapidocr_engine(image)
        if isinstance(raw_result, tuple):
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

    def release_models_for_restart(self, text_restore_delay_seconds: float = 5.0) -> None:
        self._schedule_release(text_restore_delay_seconds=text_restore_delay_seconds)

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
