import asyncio
import gc
import inspect
import json
import logging
import os
import shutil
import socket
import socketserver
import sys
import threading
import time
import traceback
import types
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import openvino as ov
import yaml
from insightface.app import FaceAnalysis
from rapidocr import RapidOCR
from rapidocr.main import RapidOCRError, TextClsOutput, TextDetOutput, TextRecOutput
from rapidocr.ch_ppocr_rec.main import LangRec, VisRes, reorder_bidi_for_display
from rapidocr.utils.process_img import apply_vertical_padding, get_rotate_crop_image
from rapidocr.utils.typings import EngineType

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

MODEL_NAME = "antelopev2"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77
CLIP_IMAGE_RESOLUTION = 224

_CLIP_IMAGE_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
_CLIP_IMAGE_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
_INSIGHTFACE_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_TOKENIZER = FullTokenizer()
_PAD_TOKEN_ID = int(_TOKENIZER.vocab["[PAD]"])
_CLS_TOKEN_ID = int(_TOKENIZER.vocab["[CLS]"])
_SEP_TOKEN_ID = int(_TOKENIZER.vocab["[SEP]"])

QUEUE_MAX_SIZE = int(os.environ.get("INFERENCE_QUEUE_MAX_SIZE", "64"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_TASK_TIMEOUT", "10"))
QUEUE_TIMEOUT_SECONDS = int(
    os.environ.get("INFERENCE_QUEUE_TIMEOUT", str(TASK_TIMEOUT_SECONDS))
)
EXEC_TIMEOUT_SECONDS = int(
    os.environ.get("INFERENCE_EXEC_TIMEOUT", str(max(30, TASK_TIMEOUT_SECONDS)))
)
RAPIDOCR_V5_MOBILE_DET_FILE = "ch_PP-OCRv5_mobile_det.onnx"
RAPIDOCR_V5_MOBILE_REC_FILE = "ch_PP-OCRv5_rec_mobile_infer.onnx"
RAPIDOCR_V5_DICT_FILE = "ppocrv5_dict.txt"
RAPIDOCR_CLS_MOBILE_V2_FILE = "ch_ppocr_mobile_v2.0_cls_infer.onnx"

LOG = logging.getLogger(__name__)
_DEFAULT_NON_TEXT_OV_DEVICE = "AUTO"
_PROCESS_LOCK_POLL_SECONDS = 0.05
_TEXT_RPC_HOST = "127.0.0.1"

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt  # type: ignore[attr-defined]
except ImportError:
    msvcrt = None  # type: ignore[assignment]

_CONFIGURED_MODEL_NAME = os.environ.get("MODEL_NAME")
if _CONFIGURED_MODEL_NAME and _CONFIGURED_MODEL_NAME != MODEL_NAME:
    LOG.warning(
        "MODEL_NAME is fixed to %s; ignoring environment value: %s",
        MODEL_NAME,
        _CONFIGURED_MODEL_NAME,
    )


def _patch_rapidocr_openvino_multi_output() -> None:
    try:
        from rapidocr.inference_engine.openvino.main import OpenVINOError, OpenVINOInferSession
    except Exception:
        return

    if getattr(OpenVINOInferSession, "_mt_multi_output_patch", False):
        return

    def _get_thread_local_request(self: Any, compiled_model: ov.CompiledModel) -> ov.InferRequest:
        request_local = getattr(self, "_mt_request_local", None)
        if request_local is None:
            request_local = threading.local()
            self._mt_request_local = request_local

        request = getattr(request_local, "request", None)
        if request is None:
            request = compiled_model.create_infer_request()
            request_local.request = request
        return request

    def _patched_call(self: Any, input_content: np.ndarray) -> Any:
        try:
            prepared = np.ascontiguousarray(input_content)
            compiled_model = getattr(self, "_mt_compiled_model", None)
            if compiled_model is None:
                session = getattr(self, "session", None)
                get_compiled_model = getattr(session, "get_compiled_model", None)
                if callable(get_compiled_model):
                    compiled_model = get_compiled_model()
                    self._mt_compiled_model = compiled_model
            if compiled_model is None:
                self.session.infer(inputs=[prepared])
                return np.asarray(self.session.get_output_tensor().data)

            infer_request = _get_thread_local_request(self, compiled_model)
            infer_request.set_input_tensor(0, ov.Tensor(prepared, shared_memory=True))
            infer_request.infer()
            return np.asarray(infer_request.get_output_tensor(0).data)
        except Exception as exc:
            error_info = traceback.format_exc()
            raise OpenVINOError(error_info) from exc

    OpenVINOInferSession.__call__ = _patched_call  # type: ignore[assignment]
    OpenVINOInferSession._mt_multi_output_patch = True  # type: ignore[attr-defined]


def _patch_rapidocr_openvino_device_selection() -> None:
    try:
        from rapidocr.inference_engine.openvino.main import CPUConfig, Core, OpenVINOInferSession
    except Exception:
        return

    if getattr(OpenVINOInferSession, "_mt_device_patch", False):
        return

    def _patched_init(self: Any, cfg: Any) -> None:
        engine_cfg = cfg.get("engine_cfg", {}) or {}
        stage_device = cfg.get("device_name", None)
        configured_device = _normalize_non_text_openvino_device(
            str(
                stage_device
                if stage_device not in (None, "")
                else engine_cfg.get("device_name", os.environ.get("RAPIDOCR_DEVICE", INFERENCE_DEVICE))
            )
        )
        core = Core()
        selected_device = _resolve_non_text_openvino_runtime_device(
            configured_device,
            core.available_devices,
            consumer="openvino",
        )

        model_path = cfg.get("model_path", None)
        if model_path is None:
            raise RuntimeError(
                "RapidOCR OpenVINO local model_path is required. "
                "Online download fallback is disabled."
            )
        model_path = Path(model_path)
        self._verify_model(model_path)

        cpu_config = CPUConfig(engine_cfg)
        core.set_property("CPU", cpu_config.get_config())

        cache_dir = engine_cfg.get("cache_dir")
        if cache_dir not in (None, ""):
            core.set_property({"CACHE_DIR": str(cache_dir)})

        compile_cfg: Dict[str, str] = {}
        for key, ov_key in (
            ("performance_hint", "PERFORMANCE_HINT"),
            ("performance_num_requests", "PERFORMANCE_HINT_NUM_REQUESTS"),
            ("num_streams", "NUM_STREAMS"),
        ):
            value = engine_cfg.get(key)
            if value in (None, "", -1):
                continue
            compile_cfg[ov_key] = str(value)

        if selected_device.startswith("AUTO:"):
            device_priorities = ",".join(_split_openvino_device_expr(selected_device))
            if device_priorities:
                compile_cfg[str(ov.properties.device.priorities)] = device_priorities
            compile_cfg[str(ov.properties.intel_auto.enable_startup_fallback)] = "false"
            compile_cfg[str(ov.properties.intel_auto.enable_runtime_fallback)] = "false"

        try:
            self.model = core.read_model(model_path)
            compiled_model = core.compile_model(
                model=self.model,
                device_name=selected_device,
                config=compile_cfg,
            )
        except Exception as exc:
            raise RuntimeError(
                f"RapidOCR OpenVINO compile_model failed on device={selected_device}. "
                "No silent fallback is allowed."
            ) from exc

        self._mt_core = core
        self._mt_compiled_model = compiled_model
        self._mt_request_local = threading.local()
        self.session = compiled_model.create_infer_request()
        self._mt_device_name = selected_device
        self._mt_configured_device_name = configured_device
        self._mt_execution_devices = tuple(_get_compiled_model_execution_devices(compiled_model))

    OpenVINOInferSession.__init__ = _patched_init  # type: ignore[assignment]
    OpenVINOInferSession._mt_device_patch = True  # type: ignore[attr-defined]


_patch_rapidocr_openvino_multi_output()
_patch_rapidocr_openvino_device_selection()


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


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_openvino_devices(devices: Any) -> List[str]:
    normalized: List[str] = []
    for item in devices or []:
        device_name = str(item).strip().upper().strip("()")
        if device_name:
            normalized.append(device_name)
    return normalized


def _get_openvino_gpu_devices(devices: Any) -> List[str]:
    gpu_devices: List[str] = []
    seen: set[str] = set()
    for device_name in _normalize_openvino_devices(devices):
        if not device_name.startswith("GPU") or device_name in seen:
            continue
        gpu_devices.append(device_name)
        seen.add(device_name)
    return gpu_devices


def _extract_explicit_gpu_devices(device_expr: str) -> List[str]:
    requested: List[str] = []
    seen: set[str] = set()
    for token in str(device_expr or "").strip().upper().replace(":", ",").split(","):
        candidate = token.strip()
        if not candidate.startswith("GPU.") or candidate in seen:
            continue
        requested.append(candidate)
        seen.add(candidate)
    return requested


def _summarize_exception(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message or exc.__class__.__name__


def _has_openvino_gpu_device(devices: Any) -> bool:
    return bool(_get_openvino_gpu_devices(devices))


def _dedupe_preserve_order(tokens: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return deduped


def _split_openvino_device_expr(device_expr: str) -> List[str]:
    normalized = str(device_expr or "").strip().upper()
    if not normalized:
        return []

    prefix, separator, suffix = normalized.partition(":")
    if prefix in {"AUTO", "MULTI", "HETERO", "BATCH"} and separator:
        raw_tokens = suffix.split(",")
    else:
        raw_tokens = normalized.replace(";", ",").split(",")

    return _dedupe_preserve_order([token.strip() for token in raw_tokens if token.strip()])


def _normalize_non_text_openvino_device(device_expr: str) -> str:
    normalized = str(device_expr or "").strip().upper()
    if not normalized:
        return _DEFAULT_NON_TEXT_OV_DEVICE

    if normalized.startswith("AUTO:"):
        return "AUTO"

    if normalized.startswith("MULTI:"):
        tokens = _split_openvino_device_expr(normalized)
        if tokens in (["GPU", "CPU"], ["CPU", "GPU"]):
            return "AUTO"
        return normalized

    if normalized in {"AUTO", "GPU", "CPU", "NPU"}:
        return normalized

    alias_map = {
        "GPU_FP16": "GPU",
        "GPU_FP32": "GPU",
        "CPU_FP16": "CPU",
        "CPU_FP32": "CPU",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    if normalized.startswith(("GPU.", "CPU.", "NPU.")):
        return normalized

    tokens = _split_openvino_device_expr(normalized)
    if tokens and set(tokens) == {"GPU", "CPU"}:
        return "AUTO"

    return normalized


def _resolve_non_text_openvino_runtime_device(
    device_expr: str,
    available_devices: Any,
    *,
    consumer: Literal["openvino", "ort_ep"],
) -> str:
    normalized = _normalize_non_text_openvino_device(device_expr)
    available = _normalize_openvino_devices(available_devices)
    available_set = set(available)
    has_gpu = _has_openvino_gpu_device(available)
    explicit_gpu_devices = _extract_explicit_gpu_devices(normalized)
    missing_explicit_gpu = [
        device_name for device_name in explicit_gpu_devices if device_name not in available_set
    ]
    if missing_explicit_gpu:
        raise RuntimeError(
            "Requested GPU device is unavailable. "
            f"requested={missing_explicit_gpu} available_devices={sorted(available)}. "
            "No silent fallback is allowed."
        )

    gpu_requested = normalized.startswith("GPU") or any(
        token.startswith("GPU") for token in _split_openvino_device_expr(normalized)
    )
    if gpu_requested and not has_gpu:
        raise RuntimeError(
            f"OpenVINO device={normalized} requires GPU, "
            f"but available_devices={sorted(available)}. No silent fallback is allowed."
        )

    if consumer == "ort_ep":
        if normalized == "AUTO":
            return "GPU" if has_gpu else "AUTO"
        if normalized.startswith("GPU."):
            return "GPU"
        if normalized.startswith("CPU."):
            return "CPU"
        if normalized.startswith("NPU."):
            return "NPU"
        if normalized.startswith("MULTI:"):
            tokens = _split_openvino_device_expr(normalized)
            if any(token.startswith("GPU") for token in tokens) and has_gpu:
                return "GPU"
            if tokens and all(token.startswith("CPU") for token in tokens):
                return "CPU"
            if tokens and all(token.startswith("NPU") for token in tokens):
                return "NPU"
            return "AUTO"
        return normalized

    if normalized == "AUTO":
        return "AUTO:GPU,CPU" if has_gpu else "AUTO"
    return normalized


def _get_compiled_model_execution_devices(compiled_model: Any) -> List[str]:
    if compiled_model is None:
        return []
    get_property = getattr(compiled_model, "get_property", None)
    if not callable(get_property):
        return []

    for property_name in ("EXECUTION_DEVICES", "EXECUTION_DEVICE"):
        try:
            value = get_property(property_name)
        except Exception:
            continue
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]
    return []


def _get_openvino_session_execution_devices(session_like: Any) -> List[str]:
    compiled_model = getattr(session_like, "_mt_compiled_model", None)
    if compiled_model is None:
        session = getattr(session_like, "session", None)
        get_compiled_model = getattr(session, "get_compiled_model", None)
        if callable(get_compiled_model):
            try:
                compiled_model = get_compiled_model()
            except Exception:
                compiled_model = None
    return _get_compiled_model_execution_devices(compiled_model)


def _openvino_device_expr_requests_gpu(device_expr: str) -> bool:
    normalized = str(device_expr or "").strip().upper()
    if not normalized:
        return False
    if normalized.startswith("GPU"):
        return True
    return any(token.startswith("GPU") for token in _split_openvino_device_expr(normalized))


def _openvino_device_expr_requests_cpu(device_expr: str) -> bool:
    normalized = str(device_expr or "").strip().upper()
    if not normalized:
        return False
    if normalized.startswith("CPU"):
        return True
    return any(token.startswith("CPU") for token in _split_openvino_device_expr(normalized))


def _openvino_device_expr_requests_npu(device_expr: str) -> bool:
    normalized = str(device_expr or "").strip().upper()
    if not normalized:
        return False
    if normalized.startswith("NPU"):
        return True
    return any(token.startswith("NPU") for token in _split_openvino_device_expr(normalized))


def _default_rapidocr_stage_device(base_device: str, stage_name: str) -> str:
    normalized = _normalize_non_text_openvino_device(base_device)
    stage = str(stage_name).strip().lower()
    if stage == "cls":
        if _openvino_device_expr_requests_npu(normalized) and not _openvino_device_expr_requests_cpu(
            normalized
        ):
            return "CPU"
        return "CPU"

    if _openvino_device_expr_requests_cpu(normalized):
        return "CPU"
    if _openvino_device_expr_requests_npu(normalized):
        return "NPU"
    if _openvino_device_expr_requests_gpu(normalized) or normalized == "AUTO":
        return "GPU"
    return normalized


def _ensure_intel_opencl_device(feature_name: str) -> Tuple[str, str]:
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.haveOpenCL() or not cv2.ocl.useOpenCL():
        raise RuntimeError(
            f"{feature_name} requires OpenCV OpenCL, but OpenCL is unavailable. "
            "No silent fallback is allowed."
        )

    device = cv2.ocl.Device_getDefault()
    vendor = str(device.vendorName())
    name = str(device.name())
    if "INTEL" not in vendor.upper():
        raise RuntimeError(
            f"{feature_name} requires Intel OpenCL device. "
            f"Current OpenCL device: {name} ({vendor}). "
            "Set OPENCV_OPENCL_DEVICE to Intel GPU and retry. "
            "No silent fallback is allowed."
        )
    return name, vendor


def _normalize_rapidocr_limit_type(value: Any, default: str = "max") -> str:
    normalized = str(value).strip().lower()
    if normalized in {"max", "min"}:
        return normalized
    return default


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


def _as_contiguous_bgr_uint8(image: Any, context: str) -> np.ndarray:
    if isinstance(image, cv2.UMat):
        image = image.get()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{context} expects BGR image with shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    if image.flags.c_contiguous:
        return image
    return np.ascontiguousarray(image)


def _estimate_similarity_transform_matrix(src: Any, dst: Any) -> np.ndarray:
    src_points = np.asarray(src, dtype=np.float64)
    dst_points = np.asarray(dst, dtype=np.float64)
    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError(
            "InsightFace alignment expects matching 2D landmarks, "
            f"got src={src_points.shape}, dst={dst_points.shape}"
        )

    point_count = src_points.shape[0]
    if point_count < 2:
        raise ValueError(
            "InsightFace alignment requires at least 2 points, "
            f"got {point_count}"
        )

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_demean = src_points - src_mean
    dst_demean = dst_points - dst_mean

    covariance = dst_demean.T @ src_demean / float(point_count)
    diagonal = np.ones((2,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        diagonal[1] = -1.0

    u_mat, singular_values, vh_mat = np.linalg.svd(covariance)
    rank = int(np.linalg.matrix_rank(covariance))
    if rank == 0:
        raise RuntimeError("InsightFace alignment landmarks are ill-conditioned.")

    if rank == 1:
        if np.linalg.det(u_mat) * np.linalg.det(vh_mat) > 0:
            rotation = u_mat @ vh_mat
        else:
            last_value = diagonal[1]
            diagonal[1] = -1.0
            rotation = u_mat @ np.diag(diagonal) @ vh_mat
            diagonal[1] = last_value
    else:
        rotation = u_mat @ np.diag(diagonal) @ vh_mat

    src_variance = float(src_demean.var(axis=0).sum())
    if src_variance <= 0.0:
        raise RuntimeError("InsightFace alignment source landmarks have zero variance.")
    scale = float(singular_values @ diagonal) / src_variance

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = dst_mean - scale * (rotation @ src_mean.T)
    transform[:2, :2] *= scale

    if np.isnan(transform).any():
        raise RuntimeError("InsightFace alignment similarity transform contains NaN.")
    return transform[:2, :].astype(np.float32, copy=False)


def _estimate_insightface_norm_matrix(
    landmark: Any,
    image_size: int,
) -> np.ndarray:
    landmark_array = np.asarray(landmark, dtype=np.float32)
    if landmark_array.shape != (5, 2):
        raise ValueError(
            "InsightFace alignment expects 5 facial landmarks, "
            f"got {landmark_array.shape}"
        )

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(
            "InsightFace alignment image_size must be divisible by 112 or 128, "
            f"got {image_size}"
        )

    destination = _INSIGHTFACE_ARCFACE_TEMPLATE.copy() * ratio
    destination[:, 0] += diff_x
    return _estimate_similarity_transform_matrix(landmark_array, destination)


def _to_channel_triplet(value: Any) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        as_list = [float(item) for item in value]
        if len(as_list) == 1:
            return [as_list[0], as_list[0], as_list[0]]
        if len(as_list) != 3:
            raise ValueError(f"Expected channel scalar or len=3 list, got len={len(as_list)}")
        return as_list
    scalar = float(value)
    return [scalar, scalar, scalar]


def _is_lock_contention_error(exc: OSError) -> bool:
    if isinstance(exc, (BlockingIOError, PermissionError)):
        return True
    if getattr(exc, "errno", None) in {11, 13}:
        return True
    if getattr(exc, "winerror", None) in {33, 36}:
        return True
    return False


class _InterProcessFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Optional[Any] = None

    def acquire(self, timeout: Optional[float], blocking: bool = True) -> bool:
        if self._handle is not None:
            return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0 and self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                self._try_lock(handle)
                self._handle = handle
                return True
            except OSError as exc:
                if not _is_lock_contention_error(exc):
                    handle.close()
                    raise
                if not blocking:
                    handle.close()
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    return False
                time.sleep(_PROCESS_LOCK_POLL_SECONDS)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return

        try:
            if os.name == "nt":
                if msvcrt is None:
                    raise RuntimeError("msvcrt is required for Windows file locking.")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                if fcntl is None:
                    raise RuntimeError("fcntl is required for POSIX file locking.")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    @staticmethod
    def _try_lock(handle: Any) -> None:
        if os.name == "nt":
            if msvcrt is None:
                raise RuntimeError("msvcrt is required for Windows file locking.")
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return

        if fcntl is None:
            raise RuntimeError("fcntl is required for POSIX file locking.")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@dataclass(slots=True)
class _OpenVinoPreprocessRunner:
    compiled_model: ov.CompiledModel
    input_port: Any
    output_port: Any
    runner_name: str
    input_height: int
    input_width: int
    _request_local: threading.local = field(
        init=False,
        repr=False,
        default_factory=threading.local,
    )

    def _get_request(self) -> ov.InferRequest:
        request = getattr(self._request_local, "request", None)
        if request is None:
            # OpenVINO InferRequest cannot be reused concurrently across worker threads.
            request = self.compiled_model.create_infer_request()
            self._request_local.request = request
        return request

    def run(self, tensor: np.ndarray) -> np.ndarray:
        prepared = np.ascontiguousarray(tensor, dtype=np.uint8)
        request = self._get_request()
        request.set_tensor(self.input_port, ov.Tensor(prepared, shared_memory=True))
        # Avoid InferRequest.infer() OVDict wrapping for anonymous Result ports on some OV runtimes.
        request.start_async()
        request.wait()
        return np.asarray(request.get_tensor(self.output_port).data)

    def validate(self) -> None:
        sample = np.zeros((1, self.input_height, self.input_width, 3), dtype=np.uint8)
        output = self.run(sample)
        expected_shape = (1, 3, self.input_height, self.input_width)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{self.runner_name} PPP output shape mismatch: "
                f"expected={expected_shape} got={tuple(output.shape)}"
            )
        if output.dtype != np.float32:
            raise RuntimeError(
                f"{self.runner_name} PPP output dtype mismatch: expected=float32 got={output.dtype}"
            )


class _TextClipRpcServer(socketserver.TCPServer):
    allow_reuse_address = True


TaskType = Literal["clip_img", "ocr", "face"]
NonTextFamily = Literal["vision", "ocr", "face"]


@dataclass(slots=True)
class _InferenceTask:
    kind: TaskType
    payload: Any
    future: Future
    created_at: float
    started_at: Optional[float] = None
    started_event: threading.Event = field(default_factory=threading.Event)


class AIModels:
    """
    Text-CLIP stays resident as a single-threaded singleton service.
    Image-CLIP uses a dedicated batch queue after standardized preprocessing.
    Non-text families are lazy-loaded and switch synchronously so only one
    vision/OCR/face family stays resident at a time.
    """

    def __init__(self) -> None:
        self.model_base_path = Path(
            os.environ.get("MODEL_PATH", str(_PROJECT_ROOT / "models"))
        )
        self.insightface_root = self.model_base_path / "insightface"
        self.insightface_model_root = self.insightface_root / "models"
        self.qa_clip_path = self.model_base_path / "qa-clip" / "openvino"

        cache_dir_raw = str(os.environ.get("OV_CACHE_DIR", "")).strip()
        if cache_dir_raw:
            self.ov_cache_dir = Path(cache_dir_raw).expanduser().resolve()
        else:
            self.ov_cache_dir = (_PROJECT_ROOT / "cache" / "openvino").resolve()
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

        self._clip_remote_context_device_name: Optional[str] = None
        self._clip_remote_context = self._init_clip_remote_context()

        self._model_lock = threading.Lock()
        self._clip_vision_load_lock = threading.Lock()
        self._rapidocr_load_lock = threading.Lock()
        self._face_load_lock = threading.Lock()

        self._clip_text_model: Optional[ov.CompiledModel] = None
        self._clip_text_request: Optional[ov.InferRequest] = None
        self._clip_text_input_names: Optional[Tuple[str, str]] = None
        self._clip_text_host_input_cache: Dict[
            int, Tuple[ov.Tensor, np.ndarray, ov.Tensor, np.ndarray]
        ] = {}
        self._clip_text_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_text_lock = threading.Lock()
        self._runtime_state_dir = (_PROJECT_ROOT / "cache" / "runtime").resolve()
        self._runtime_state_dir.mkdir(parents=True, exist_ok=True)
        self._text_service_meta_path = self._runtime_state_dir / "text-clip-service.json"
        self._text_service_lock = _InterProcessFileLock(
            self._runtime_state_dir / "text-clip-service.lock"
        )
        self._text_service_owner = False
        self._text_service_server: Optional[_TextClipRpcServer] = None
        self._text_service_thread: Optional[threading.Thread] = None
        self._text_service_port: Optional[int] = None
        self._clip_vision_model: Optional[ov.CompiledModel] = None
        self._clip_vision_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._clip_vision_request: Optional[ov.InferRequest] = None
        self._clip_vision_input_name: Optional[str] = None
        self._clip_vision_host_input_cache: Dict[
            Tuple[int, int, int], Tuple[ov.Tensor, np.ndarray]
        ] = {}
        self._clip_vision_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_image_batch_size = max(
            1,
            _as_int(
                os.environ.get("CLIP_IMAGE_BATCH", os.environ.get("CLIP_IMAGE_BATCH_SIZE")),
                8,
            ),
        )
        self._clip_image_batch_wait_seconds = max(
            0.0,
            _as_float(os.environ.get("CLIP_IMAGE_BATCH_WAIT_MS"), 5.0) / 1000.0,
        )
        self._rapidocr_engine: Optional[RapidOCR] = None
        self._rapidocr_runtime_cfg: Optional[Dict[str, Any]] = None
        self._ocr_opencl_device_name: Optional[str] = None
        self._ocr_opencl_device_vendor: Optional[str] = None
        self._face_engine: Optional[FaceAnalysis] = None
        self._face_det_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._face_rec_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._shared_cpu_executor = ThreadPoolExecutor(
            max_workers=max(2, min(8, os.cpu_count() or 4)),
            thread_name_prefix="ai-cpu",
        )
        self._ocr_stage_worker_count = max(
            1,
            _as_int(os.environ.get("RAPIDOCR_PERFORMANCE_NUM_REQUESTS"), 2),
        )
        self._ocr_det_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-det",
        )
        self._ocr_cls_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-cls",
        )
        self._ocr_rec_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-rec",
        )
        self._face_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-ov")
        self._family_load_locks = {
            family: _InterProcessFileLock(self._runtime_state_dir / f"{family}.load.lock")
            for family in ("vision", "ocr", "face")
        }

        self._condition = threading.Condition()
        self._normal_queue: Deque[_InferenceTask] = deque()
        self._queue_capacity = max(1, QUEUE_MAX_SIZE)
        self._queue_timeout_seconds = max(1, QUEUE_TIMEOUT_SECONDS)
        self._execution_timeout_seconds = max(1, EXEC_TIMEOUT_SECONDS)
        self._ocr_execution_timeout_seconds = max(
            1,
            _as_int(
                os.environ.get("OCR_EXEC_TIMEOUT"),
                max(30, self._execution_timeout_seconds),
            ),
        )
        self._load_lock_timeout_seconds = max(30.0, float(self._execution_timeout_seconds))
        self._non_text_condition = threading.Condition()
        self._non_text_active_family: Optional[NonTextFamily] = None
        self._non_text_inflight: Dict[NonTextFamily, int] = {
            "vision": 0,
            "ocr": 0,
            "face": 0,
        }
        self._non_text_transition_in_progress = False
        self._stopping = False
        self._background_prewarm_cancel = threading.Event()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ai-model-worker",
            daemon=True,
        )
        self._worker.start()
        self._ensure_text_service_ready(preload=True)
        self._background_prewarm_thread: Optional[threading.Thread] = None
        self._start_background_prewarm()

        LOG.info(
            "AIModels ready: clip_device=%s clip_context=%s cache=%s clip_queue=%s queue_timeout=%ss exec_timeout=%ss ocr_exec_timeout=%ss clip_batch=%s/%sms text_service=%s ocr_prewarm=%s",
            CLIP_INFERENCE_DEVICE,
            self._clip_remote_context_device_name or "disabled",
            self.ov_cache_dir or "default",
            self._queue_capacity,
            self._queue_timeout_seconds,
            self._execution_timeout_seconds,
            self._ocr_execution_timeout_seconds,
            self._clip_image_batch_size,
            int(self._clip_image_batch_wait_seconds * 1000.0),
            "owner" if self._text_service_owner else "client",
            _as_bool(os.environ.get("OCR_PREWARM_ENABLED"), False),
        )

    def _configure_openvino_cache(self) -> None:
        if self.ov_cache_dir is None:
            return
        try:
            self.core.set_property({"CACHE_DIR": str(self.ov_cache_dir)})
        except Exception as exc:
            LOG.warning("Failed to set global OpenVINO cache dir: %s", exc)

    def _start_background_prewarm(self) -> None:
        if not _as_bool(os.environ.get("OCR_PREWARM_ENABLED"), False):
            return
        self._background_prewarm_cancel.clear()
        self._background_prewarm_thread = threading.Thread(
            target=self._background_prewarm_loop,
            name="ai-ocr-prewarm",
            daemon=True,
        )
        self._background_prewarm_thread.start()

    def _background_prewarm_loop(self) -> None:
        delay_seconds = max(0.0, _as_float(os.environ.get("OCR_PREWARM_DELAY_SECONDS"), 1.0))
        if self._background_prewarm_cancel.wait(timeout=delay_seconds):
            return
        if self._stopping or self._background_prewarm_cancel.is_set():
            return
        try:
            started_at = time.monotonic()
            self._prewarm_ocr_family()
            if self._stopping or self._background_prewarm_cancel.is_set():
                LOG.info("RapidOCR background prewarm cancelled before OCR stayed resident.")
                return
            self._release_non_text_models_sync(
                reason="ocr-prewarm",
                cancel_background_prewarm=False,
                join_background_prewarm=False,
            )
            LOG.info(
                "RapidOCR background prewarm completed in %.2fs and released OCR family.",
                time.monotonic() - started_at,
            )
        except Exception as exc:
            LOG.warning("RapidOCR background prewarm failed: %s", exc, exc_info=True)

    def _prewarm_ocr_family(self) -> None:
        if self._stopping or self._background_prewarm_cancel.is_set():
            return
        leased = self._acquire_non_text_family_lease(
            "ocr",
            abort_event=self._background_prewarm_cancel,
        )
        if not leased:
            return
        try:
            if self._stopping or self._background_prewarm_cancel.is_set():
                return
            self._ensure_rapidocr_loaded()
            with self._rapidocr_load_lock:
                if self._stopping or self._background_prewarm_cancel.is_set():
                    return
                self._warmup_rapidocr_locked()
        finally:
            self._release_non_text_family_lease("ocr")

    def _init_clip_remote_context(self) -> Optional[Any]:
        clip_device = CLIP_INFERENCE_DEVICE.strip().upper()
        force_gpu_remote_context = clip_device == "AUTO"
        wants_gpu_remote_context = force_gpu_remote_context or ("GPU" in clip_device)
        if not wants_gpu_remote_context:
            return None

        available_devices = _normalize_openvino_devices(self.core.available_devices)
        gpu_devices = _get_openvino_gpu_devices(available_devices)
        runtime_hint = (
            " Ensure the container exposes a real Intel /dev/dri render node and installs "
            "OpenVINO/OpenCL runtime packages (Debian 13 stable baseline: libze1, "
            "ocl-icd-libopencl1, mesa-opencl-icd; optional diagnostics such as clinfo "
            "are not bundled in the runtime image by default)."
        )
        if not gpu_devices:
            LOG.info(
                "OpenVINO available_devices does not include GPU for CLIP request=%s "
                "(available_devices=%s). Continue probing explicit GPU remote context APIs.",
                CLIP_INFERENCE_DEVICE,
                sorted(available_devices),
            )

        explicit_gpu_devices = _extract_explicit_gpu_devices(clip_device)
        context_candidates: List[str] = []
        for candidate in [*explicit_gpu_devices, "GPU", *gpu_devices]:
            if candidate not in context_candidates:
                context_candidates.append(candidate)

        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None

        for candidate in context_candidates:
            try:
                remote_context = self.core.get_default_context(candidate)
                resolved_device = str(remote_context.get_device_name()).strip().upper()
                if explicit_gpu_devices and resolved_device not in explicit_gpu_devices:
                    raise RuntimeError(
                        f"resolved device {resolved_device} does not match requested "
                        f"{explicit_gpu_devices}"
                    )
                self._clip_remote_context_device_name = resolved_device
                return remote_context
            except Exception as exc:
                last_exc = exc
                attempt_errors.append(
                    f"get_default_context({candidate}) failed: {_summarize_exception(exc)}"
                )

        try:
            remote_context = self.core.create_context("GPU", {})
            resolved_device = str(remote_context.get_device_name()).strip().upper()
            if explicit_gpu_devices and resolved_device not in explicit_gpu_devices:
                raise RuntimeError(
                    f"create_context(GPU) resolved to {resolved_device}, "
                    f"expected one of {explicit_gpu_devices}"
                )
            self._clip_remote_context_device_name = resolved_device
            return remote_context
        except Exception as exc:
            last_exc = exc
            attempt_errors.append(
                f"create_context(GPU) failed: {_summarize_exception(exc)}"
            )

        LOG.error(
            "CLIP GPU remote context initialization failed: request=%s available_devices=%s attempts=%s",
            CLIP_INFERENCE_DEVICE,
            sorted(available_devices),
            " | ".join(attempt_errors),
        )
        if force_gpu_remote_context:
            raise RuntimeError(
                "CLIP_INFERENCE_DEVICE=AUTO requires GPU Remote Context. "
                "OpenVINO GPU context initialization failed. "
                f"available_devices={sorted(available_devices)}.{runtime_hint}"
            ) from last_exc
        raise RuntimeError(
            f"CLIP_INFERENCE_DEVICE={CLIP_INFERENCE_DEVICE} requests GPU execution, "
            "but GPU Remote Context initialization failed. "
            f"No silent fallback is allowed. available_devices={sorted(available_devices)}."
            f"{runtime_hint}"
        ) from last_exc

    def _queue_size_locked(self) -> int:
        return len(self._normal_queue)

    def _submit_task(self, kind: TaskType, payload: Any) -> _InferenceTask:
        future: Future = Future()
        task = _InferenceTask(kind=kind, payload=payload, future=future, created_at=time.time())
        with self._condition:
            if self._stopping:
                future.set_exception(RuntimeError("模型服务已关闭"))
                return task
            if self._queue_size_locked() >= self._queue_capacity:
                future.set_exception(
                    RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试")
                )
                return task
            self._normal_queue.append(task)
            self._condition.notify()
        return task

    def _cancel_task_if_queued(self, task: _InferenceTask, exc: Exception) -> bool:
        with self._condition:
            if task.started_event.is_set():
                return False
            try:
                self._normal_queue.remove(task)
            except ValueError:
                return False
        self._safe_set_exception(task.future, exc)
        return True

    def _wait_task(self, task: _InferenceTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = task.started_event.wait(timeout=self._queue_timeout_seconds)
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_task_if_queued(task, queue_exc):
                raise queue_exc
            if not task.started_event.wait(timeout=0.05) and not task.future.done():
                raise queue_exc

        try:
            return task.future.result(timeout=self._execution_timeout_seconds)
        except FutureTimeoutError as exc:
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    async def _await_task(self, task: _InferenceTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = await asyncio.to_thread(task.started_event.wait, self._queue_timeout_seconds)
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_task_if_queued(task, queue_exc):
                raise queue_exc
            started = await asyncio.to_thread(task.started_event.wait, 0.05)
            if not started and not task.future.done():
                raise queue_exc

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(task.future),
                timeout=self._execution_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    def _safe_set_result(self, future: Future, value: Any) -> None:
        if not future.done():
            future.set_result(value)

    def _safe_set_exception(self, future: Future, exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

    @staticmethod
    def _log_detached_async_task_failure(task: "asyncio.Task[Any]", task_name: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            LOG.error("%s failed after caller detached: %s", task_name, exc, exc_info=exc)

    def _bind_non_text_lease_to_future(self, family: NonTextFamily, future: Future) -> None:
        future.add_done_callback(lambda _future: self._release_non_text_family_lease(family))

    def _join_background_prewarm_thread(self, timeout_seconds: Optional[float]) -> None:
        thread = self._background_prewarm_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            LOG.warning(
                "RapidOCR background prewarm thread did not exit within %.1fs.",
                float(timeout_seconds or 0.0),
            )
            return
        self._background_prewarm_thread = None

    def _acquire_non_text_family_lease(
        self,
        family: NonTextFamily,
        abort_event: Optional[threading.Event] = None,
    ) -> bool:
        while True:
            previous_family: Optional[NonTextFamily]
            with self._non_text_condition:
                while self._non_text_transition_in_progress:
                    if abort_event is not None and abort_event.is_set():
                        return False
                    if self._stopping:
                        raise RuntimeError("模型服务已关闭")
                    self._non_text_condition.wait(
                        timeout=0.1 if abort_event is not None else None
                    )
                if abort_event is not None and abort_event.is_set():
                    return False
                if self._stopping:
                    raise RuntimeError("模型服务已关闭")

                previous_family = self._non_text_active_family
                if previous_family in (None, family):
                    self._non_text_active_family = family
                    self._non_text_inflight[family] += 1
                    return True

                if self._non_text_inflight[previous_family] > 0:
                    if abort_event is not None and abort_event.is_set():
                        return False
                    self._non_text_condition.wait(
                        timeout=0.1 if abort_event is not None else None
                    )
                    continue

                self._non_text_transition_in_progress = True

            switch_exc: Optional[Exception] = None
            unloaded_families: List[NonTextFamily] = []
            try:
                unloaded_families = self._unload_non_text_models(keep_family=family)
            except Exception as exc:
                switch_exc = exc

            aborted = abort_event is not None and abort_event.is_set()
            with self._non_text_condition:
                if switch_exc is None and not aborted:
                    self._non_text_active_family = family
                    self._non_text_inflight[family] += 1
                self._non_text_transition_in_progress = False
                self._non_text_condition.notify_all()

            if switch_exc is not None:
                raise switch_exc
            if aborted:
                return False
            if unloaded_families:
                LOG.info(
                    "Switched non-text family from %s to %s; released=%s",
                    previous_family,
                    family,
                    ",".join(unloaded_families),
                )
            return True

    def _release_non_text_family_lease(self, family: NonTextFamily) -> None:
        with self._non_text_condition:
            inflight = self._non_text_inflight[family]
            if inflight <= 0:
                return
            self._non_text_inflight[family] = inflight - 1
            if self._non_text_inflight[family] == 0:
                self._non_text_condition.notify_all()

    def _unload_non_text_models_locked(
        self,
        keep_family: Optional[NonTextFamily] = None,
    ) -> List[NonTextFamily]:
        unloaded: List[NonTextFamily] = []
        if keep_family != "vision" and (
            self._clip_vision_model is not None
            or self._clip_vision_request is not None
            or self._clip_vision_ppp is not None
        ):
            self._unload_clip_vision_model_locked()
            unloaded.append("vision")
        if keep_family != "ocr" and self._rapidocr_engine is not None:
            self._unload_rapidocr_model_locked()
            unloaded.append("ocr")
        if keep_family != "face" and (
            self._face_engine is not None
            or self._face_det_ppp is not None
            or self._face_rec_ppp is not None
        ):
            self._unload_face_model_locked()
            unloaded.append("face")
        return unloaded

    def _unload_non_text_models(
        self,
        keep_family: Optional[NonTextFamily] = None,
    ) -> List[NonTextFamily]:
        with self._clip_vision_load_lock, self._rapidocr_load_lock, self._face_load_lock:
            unloaded = self._unload_non_text_models_locked(keep_family=keep_family)
        if unloaded:
            gc.collect()
        return unloaded

    def _release_non_text_models_sync(
        self,
        reason: str,
        *,
        cancel_background_prewarm: bool = True,
        join_background_prewarm: bool = True,
    ) -> None:
        join_timeout_seconds = max(2.0, float(self._execution_timeout_seconds))
        with self._non_text_condition:
            while self._non_text_transition_in_progress:
                if self._stopping:
                    return
                self._non_text_condition.wait()
            self._non_text_transition_in_progress = True

        try:
            if cancel_background_prewarm:
                self._background_prewarm_cancel.set()
            if join_background_prewarm:
                self._join_background_prewarm_thread(timeout_seconds=join_timeout_seconds)

            with self._non_text_condition:
                while any(self._non_text_inflight.values()):
                    self._non_text_condition.wait()

            unloaded = self._unload_non_text_models()
            LOG.info(
                "Non-text model release complete: reason=%s unloaded=%s",
                reason,
                ",".join(unloaded) if unloaded else "none",
            )
        finally:
            with self._non_text_condition:
                self._non_text_active_family = None
                self._non_text_transition_in_progress = False
                self._non_text_condition.notify_all()

    def _load_family_with_process_lock(
        self, family: Literal["vision", "ocr", "face"], loader: Callable[[], None]
    ) -> None:
        lock = self._family_load_locks[family]
        started_at = time.monotonic()
        acquired = lock.acquire(timeout=self._load_lock_timeout_seconds, blocking=True)
        if not acquired:
            raise RuntimeError(
                f"{family} 模型加载等待跨进程锁超时（>{self._load_lock_timeout_seconds:.0f}s）"
            )
        try:
            waited = time.monotonic() - started_at
            if waited >= 0.25:
                LOG.info(
                    "Waited %.2fs for %s model load lock in pid=%s.",
                    waited,
                    family,
                    os.getpid(),
                )
            loader()
        finally:
            lock.release()

    def _read_text_service_meta(self) -> Optional[Dict[str, Any]]:
        try:
            raw = json.loads(self._text_service_meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            LOG.warning("Failed to read Text-CLIP service metadata: %s", exc)
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def _write_text_service_meta(self, port: int) -> None:
        payload = {"port": int(port), "pid": os.getpid()}
        self._text_service_meta_path.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )

    def _probe_text_service(self, port: int, timeout_seconds: float = 0.5) -> bool:
        try:
            with socket.create_connection((_TEXT_RPC_HOST, int(port)), timeout=timeout_seconds) as conn:
                request = {"op": "ping"}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                response_line = b""
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    response_line += chunk
        except OSError:
            return False

        if not response_line:
            return False
        try:
            response = json.loads(response_line.decode("utf-8"))
        except Exception:
            return False
        return response.get("ok") is True

    def _load_text_model_once_locked(self) -> None:
        if self._clip_text_model is not None and self._clip_text_request is not None:
            return
        self._load_clip_text_locked()

    def _load_text_model_once(self) -> None:
        with self._clip_text_lock:
            self._load_text_model_once_locked()

    def _start_text_service_locked(self) -> None:
        if self._text_service_server is not None and self._text_service_thread is not None:
            return

        self._load_text_model_once_locked()
        parent = self

        class _TextClipRequestHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request_line = self.rfile.readline()
                    if not request_line:
                        return
                    request = json.loads(request_line.decode("utf-8"))
                    op = str(request.get("op", "")).strip().lower()
                    if op == "ping":
                        response = {"ok": True}
                    elif op == "embed":
                        text = str(request.get("text", ""))
                        response = {"result": parent._infer_text_locally(text)}
                    else:
                        response = {"error": f"Unsupported op: {op or 'missing'}"}
                except Exception as exc:
                    response = {"error": str(exc)}
                self.wfile.write((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))

        server = _TextClipRpcServer((_TEXT_RPC_HOST, 0), _TextClipRequestHandler)
        self._text_service_server = server
        self._text_service_port = int(server.server_address[1])
        self._write_text_service_meta(self._text_service_port)
        self._text_service_thread = threading.Thread(
            target=server.serve_forever,
            name="text-clip-rpc",
            daemon=True,
        )
        self._text_service_thread.start()
        LOG.info(
            "Text-CLIP RPC service started on %s:%s in pid=%s.",
            _TEXT_RPC_HOST,
            self._text_service_port,
            os.getpid(),
        )

    def _ensure_text_service_ready(self, preload: bool) -> None:
        if self._text_service_owner and self._text_service_port is not None:
            if preload:
                self._load_text_model_once()
            return

        metadata = self._read_text_service_meta()
        if metadata is not None:
            candidate_port = int(metadata.get("port", 0) or 0)
            if candidate_port > 0 and self._probe_text_service(candidate_port):
                self._text_service_port = candidate_port
                return

        acquired = self._text_service_lock.acquire(timeout=0.0, blocking=False)
        if not acquired:
            deadline = time.monotonic() + max(3.0, float(self._execution_timeout_seconds))
            while time.monotonic() < deadline:
                metadata = self._read_text_service_meta()
                if metadata is not None:
                    candidate_port = int(metadata.get("port", 0) or 0)
                    if candidate_port > 0 and self._probe_text_service(candidate_port):
                        self._text_service_port = candidate_port
                        return
                time.sleep(_PROCESS_LOCK_POLL_SECONDS)
            raise RuntimeError("Text-CLIP RPC service is unavailable.")

        self._text_service_owner = True
        try:
            metadata = self._read_text_service_meta()
            if metadata is not None:
                candidate_port = int(metadata.get("port", 0) or 0)
                if candidate_port > 0 and self._probe_text_service(candidate_port):
                    self._text_service_port = candidate_port
                    return
            self._start_text_service_locked()
        except Exception:
            self._text_service_owner = False
            self._text_service_lock.release()
            raise

    def _infer_text_locally(self, text: str) -> List[float]:
        with self._clip_text_lock:
            self._load_text_model_once_locked()
            return self._infer_clip_text_batch([text])[0]

    def _request_text_embedding_remote(self, text: str) -> List[float]:
        self._ensure_text_service_ready(preload=False)
        if self._text_service_owner:
            return self._infer_text_locally(text)
        if self._text_service_port is None:
            raise RuntimeError("Text-CLIP RPC service port is unavailable.")

        response_line = b""
        try:
            with socket.create_connection(
                (_TEXT_RPC_HOST, self._text_service_port),
                timeout=max(1.0, float(self._execution_timeout_seconds)),
            ) as conn:
                request = {"op": "embed", "text": text}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(16384)
                    if not chunk:
                        break
                    response_line += chunk
        except OSError:
            self._text_service_port = None
            self._ensure_text_service_ready(preload=False)
            if self._text_service_owner:
                return self._infer_text_locally(text)
            if self._text_service_port is None:
                raise RuntimeError("Text-CLIP RPC service port is unavailable.")
            response_line = b""
            with socket.create_connection(
                (_TEXT_RPC_HOST, self._text_service_port),
                timeout=max(1.0, float(self._execution_timeout_seconds)),
            ) as conn:
                request = {"op": "embed", "text": text}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(16384)
                    if not chunk:
                        break
                    response_line += chunk

        if not response_line:
            raise RuntimeError("Text-CLIP RPC service returned empty response.")
        response = json.loads(response_line.decode("utf-8"))
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        result = response.get("result")
        if not isinstance(result, list):
            raise RuntimeError("Text-CLIP RPC service returned invalid payload.")
        return [float(item) for item in result]

    def _shutdown_text_service(self) -> None:
        if self._text_service_server is not None:
            self._text_service_server.shutdown()
            self._text_service_server.server_close()
            self._text_service_server = None
        if self._text_service_thread is not None:
            self._text_service_thread.join(timeout=2.0)
            self._text_service_thread = None
        if self._text_service_owner:
            self._text_service_lock.release()
            self._text_service_owner = False
        if self._text_service_meta_path.exists():
            metadata = self._read_text_service_meta()
            if metadata is not None and int(metadata.get("pid", -1)) == os.getpid():
                try:
                    self._text_service_meta_path.unlink()
                except OSError:
                    pass

    def _worker_loop(self) -> None:
        while True:
            normal_task: Optional[_InferenceTask] = None

            with self._condition:
                while True:
                    if self._stopping or self._normal_queue:
                        break
                    self._condition.wait()

                if self._stopping:
                    break

                if self._normal_queue:
                    normal_task = self._normal_queue.popleft()
                    normal_task.started_at = time.monotonic()
                    normal_task.started_event.set()

            if normal_task is not None:
                self._handle_clip_image_tasks(self._collect_clip_image_batch(normal_task))

    def _collect_clip_image_batch(self, first_task: _InferenceTask) -> List[_InferenceTask]:
        batch = [first_task]
        if self._clip_image_batch_size <= 1:
            return batch

        deadline = time.monotonic() + self._clip_image_batch_wait_seconds

        while len(batch) < self._clip_image_batch_size:
            with self._condition:
                if not self._normal_queue:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return batch
                    self._condition.wait(timeout=remaining)
                    continue

                next_task = self._normal_queue[0]
                if next_task.kind != "clip_img":
                    return batch
                popped = self._normal_queue.popleft()
                popped.started_at = time.monotonic()
                popped.started_event.set()
                batch.append(popped)
        return batch

    def _handle_clip_image_tasks(self, tasks: List[_InferenceTask]) -> None:
        if not tasks:
            return
        try:
            self._ensure_clip_vision_loaded()
            payloads = [np.asarray(task.payload, dtype=np.float32) for task in tasks]
            results = self._infer_clip_image_tensor_batch(payloads)
            for task, result in zip(tasks, results):
                self._safe_set_result(task.future, result)
        except Exception as exc:
            LOG.error("Task clip_img failed: %s", exc, exc_info=True)
            for task in tasks:
                self._safe_set_exception(task.future, exc)

    def _unload_text_model_locked(self) -> None:
        self._clip_text_request = None
        self._clip_text_model = None
        self._clip_text_input_names = None
        self._clip_text_host_input_cache.clear()
        self._text_service_port = None
        gc.collect()

    def _unload_everything_locked(self) -> None:
        self._unload_text_model_locked()
        self._unload_clip_vision_model_locked()
        self._unload_rapidocr_model_locked()
        self._unload_face_model_locked()
        gc.collect()

    def _unload_clip_vision_model_locked(self) -> None:
        self._clip_vision_request = None
        self._clip_vision_model = None
        self._clip_vision_ppp = None
        self._clip_vision_input_name = None
        self._clip_vision_host_input_cache.clear()

    def _unload_clip_vision_model(self) -> None:
        with self._clip_vision_load_lock:
            self._unload_clip_vision_model_locked()

    def _unload_rapidocr_model_locked(self) -> None:
        self._rapidocr_engine = None
        self._rapidocr_runtime_cfg = None
        self._ocr_opencl_device_name = None
        self._ocr_opencl_device_vendor = None

    def _unload_rapidocr_model(self) -> None:
        with self._rapidocr_load_lock:
            self._unload_rapidocr_model_locked()

    def _unload_face_model_locked(self) -> None:
        self._face_engine = None
        self._face_det_ppp = None
        self._face_rec_ppp = None

    def _unload_face_model(self) -> None:
        with self._face_load_lock:
            self._unload_face_model_locked()

    def _ensure_clip_vision_loaded(self) -> None:
        with self._clip_vision_load_lock:
            if self._clip_vision_model is not None and self._clip_vision_request is not None:
                return
            self._load_family_with_process_lock("vision", self._load_clip_vision_locked)

    def _ensure_rapidocr_loaded(self) -> None:
        with self._rapidocr_load_lock:
            if self._rapidocr_engine is not None:
                return
            self._load_family_with_process_lock("ocr", self._load_rapidocr_locked)

    def _ensure_face_loaded(self) -> None:
        with self._face_load_lock:
            if self._face_engine is not None:
                return
            self._load_family_with_process_lock("face", self._load_face_locked)

    def _compile_clip_model(
        self,
        model_or_path: Any,
        performance_hint: str,
    ) -> ov.CompiledModel:
        config = {
            "PERFORMANCE_HINT": performance_hint,
        }

        if self._clip_remote_context is not None:
            if isinstance(model_or_path, Path):
                model = self.core.read_model(str(model_or_path))
            else:
                model = model_or_path
            return self.core.compile_model(model, self._clip_remote_context, config)

        if isinstance(model_or_path, Path):
            return self.core.compile_model(str(model_or_path), CLIP_INFERENCE_DEVICE, config)
        return self.core.compile_model(model_or_path, CLIP_INFERENCE_DEVICE, config)

    def _build_clip_vision_ppp_model(self, model_path: Path) -> ov.Model:
        model = self.core.read_model(str(model_path))
        input_shape = model.inputs[0].get_partial_shape()
        if not input_shape.rank.is_static or input_shape.rank.get_length() != 4:
            raise RuntimeError(
                "CLIP vision input rank must be 4, "
                f"got rank={input_shape.rank}"
            )

        model_height = input_shape[2]
        model_width = input_shape[3]
        if model_height.is_static and int(model_height.get_length()) != int(CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP image height mismatch: "
                f"expected={CLIP_IMAGE_RESOLUTION}, got={int(model_height.get_length())}"
            )
        if model_width.is_static and int(model_width.get_length()) != int(CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP image width mismatch: "
                f"expected={CLIP_IMAGE_RESOLUTION}, got={int(model_width.get_length())}"
            )

        ppp = ov.preprocess.PrePostProcessor(model)
        ppp.input().tensor().set_shape([-1, -1, -1, 3]).set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")
        ).set_color_format(ov.preprocess.ColorFormat.BGR)
        # openvino_image_fp16 may have dynamic NCHW input; explicit target size avoids PPP static-shape errors.
        ppp.input().preprocess().resize(
            ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR,
            int(CLIP_IMAGE_RESOLUTION),
            int(CLIP_IMAGE_RESOLUTION),
        )
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        ppp.input().preprocess().scale([255.0, 255.0, 255.0])
        ppp.input().preprocess().mean(_CLIP_IMAGE_MEAN.tolist())
        ppp.input().preprocess().scale(_CLIP_IMAGE_STD.tolist())
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        return ppp.build()

    def _get_clip_vision_host_input(
        self, batch_size: int, image_height: int, image_width: int
    ) -> Optional[Tuple[ov.Tensor, np.ndarray]]:
        if (
            not self._clip_vision_host_tensor_enabled
            or self._clip_remote_context is None
            or self._clip_vision_model is None
        ):
            return None

        cache_key = (int(batch_size), int(image_height), int(image_width))
        cached = self._clip_vision_host_input_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            input_port = self._clip_vision_model.inputs[0]
            input_shape = ov.Shape([int(batch_size), int(image_height), int(image_width), 3])
            input_tensor = self._clip_remote_context.create_host_tensor(
                input_port.get_element_type(), input_shape
            )
            input_view = np.asarray(input_tensor.data)
            cache_entry = (input_tensor, input_view)
            self._clip_vision_host_input_cache[cache_key] = cache_entry
            return cache_entry
        except Exception as exc:
            LOG.warning(
                "CLIP vision host tensor allocation failed, fallback to shared numpy inputs: %s",
                exc,
            )
            self._clip_vision_host_tensor_enabled = False
            self._clip_vision_host_input_cache.clear()
            return None

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
            model_or_path=text_model_path,
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
        LOG.info("CLIP Text model loaded on %s.", CLIP_INFERENCE_DEVICE)

    def _load_clip_vision_locked(self) -> None:
        vision_model_path = self.qa_clip_path / "openvino_image_fp16.xml"
        if not vision_model_path.exists():
            raise FileNotFoundError(f"Missing vision model: {vision_model_path}")

        model = self.core.read_model(str(vision_model_path))
        model.reshape(
            {
                model.input(0): ov.PartialShape(
                    [ov.Dimension.dynamic(), 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION]
                )
            }
        )
        compiled_model = self._compile_clip_model(
            model_or_path=model,
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
        self._clip_vision_ppp = self._build_openvino_preprocess_runner(
            runner_name="clip_vision",
            device_name=CLIP_INFERENCE_DEVICE,
            output_height=CLIP_IMAGE_RESOLUTION,
            output_width=CLIP_IMAGE_RESOLUTION,
            mean_values=_CLIP_IMAGE_MEAN.tolist(),
            std_values=_CLIP_IMAGE_STD.tolist(),
        )
        self._clip_vision_host_input_cache.clear()
        self._clip_vision_host_tensor_enabled = False
        LOG.info(
            "CLIP Vision model loaded on %s with post-preprocess batching (batch=%s).",
            CLIP_INFERENCE_DEVICE,
            self._clip_image_batch_size,
        )

    @staticmethod
    def _resize_and_center_crop_clip_image(image: np.ndarray) -> np.ndarray:
        image_bgr = _as_contiguous_bgr_uint8(image, context="CLIP vision")
        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"CLIP vision expects non-empty image, got shape={image_bgr.shape}")

        scale = float(CLIP_IMAGE_RESOLUTION) / float(min(height, width))
        resized_width = max(CLIP_IMAGE_RESOLUTION, int(round(width * scale)))
        resized_height = max(CLIP_IMAGE_RESOLUTION, int(round(height * scale)))
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        top = max(0, (resized_height - CLIP_IMAGE_RESOLUTION) // 2)
        left = max(0, (resized_width - CLIP_IMAGE_RESOLUTION) // 2)
        cropped = resized[
            top : top + CLIP_IMAGE_RESOLUTION,
            left : left + CLIP_IMAGE_RESOLUTION,
        ]
        if cropped.shape[:2] != (CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION):
            cropped = cv2.resize(
                cropped,
                (CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION),
                interpolation=cv2.INTER_CUBIC,
            )
        return np.ascontiguousarray(cropped, dtype=np.uint8)

    def _preprocess_clip_image_tensor(self, image: np.ndarray) -> np.ndarray:
        runner = self._clip_vision_ppp
        if runner is None:
            raise RuntimeError("CLIP vision preprocess runner is not loaded.")
        cropped = self._resize_and_center_crop_clip_image(image)
        normalized = runner.run(cropped[np.newaxis, ...])
        if normalized.shape != (1, 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP preprocess output shape mismatch: "
                f"got={tuple(normalized.shape)}"
            )
        return np.ascontiguousarray(normalized[0], dtype=np.float32)

    def _require_rapidocr_config_path(self) -> Path:
        if not self.rapidocr_config_path.is_file():
            raise FileNotFoundError(
                "RapidOCR OpenVINO config file is required and must exist. "
                f"Configured path: {self.rapidocr_config_path}"
            )
        return self.rapidocr_config_path

    def _load_rapidocr_openvino_config(self) -> Dict[str, Any]:
        base_device = _normalize_non_text_openvino_device(
            os.environ.get("RAPIDOCR_DEVICE", INFERENCE_DEVICE)
        )
        config: Dict[str, Any] = {
            "device_name": base_device,
            "det_device_name": None,
            "cls_device_name": None,
            "rec_device_name": None,
            "inference_num_threads": _as_int(os.environ.get("RAPIDOCR_INFERENCE_NUM_THREADS"), -1),
            "performance_hint": os.environ.get("RAPIDOCR_PERFORMANCE_HINT", "THROUGHPUT"),
            "performance_num_requests": _as_int(
                os.environ.get("RAPIDOCR_PERFORMANCE_NUM_REQUESTS"), 2
            ),
            "enable_cpu_pinning": _as_bool(os.environ.get("RAPIDOCR_ENABLE_CPU_PINNING"), True),
            "num_streams": _as_int(os.environ.get("RAPIDOCR_NUM_STREAMS"), 2),
            "enable_hyper_threading": _as_bool(
                os.environ.get("RAPIDOCR_ENABLE_HYPER_THREADING"), True
            ),
            "scheduling_core_type": os.environ.get("RAPIDOCR_SCHEDULING_CORE_TYPE", "ANY_CORE"),
            "use_cls": _as_bool(os.environ.get("RAPIDOCR_USE_CLS"), True),
            "max_side_len": _as_int(os.environ.get("RAPIDOCR_MAX_SIDE_LEN"), 960),
            "det_limit_side_len": _as_int(os.environ.get("RAPIDOCR_DET_LIMIT_SIDE_LEN"), 960),
            "det_limit_type": _normalize_rapidocr_limit_type(
                os.environ.get("RAPIDOCR_DET_LIMIT_TYPE", "max")
            ),
            "rec_batch_num": max(1, _as_int(os.environ.get("RAPIDOCR_REC_BATCH_NUM"), 8)),
            "cls_batch_num": max(1, _as_int(os.environ.get("RAPIDOCR_CLS_BATCH_NUM"), 8)),
        }

        config_path = self._require_rapidocr_config_path()
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            ov_cfg = loaded.get("EngineConfig", {}).get("openvino", {})
            for key in list(config.keys()):
                if key in ov_cfg:
                    config[key] = ov_cfg[key]

            global_cfg = loaded.get("Global", {})
            if "use_cls" in global_cfg:
                config["use_cls"] = _as_bool(global_cfg.get("use_cls"), config["use_cls"])
            if "max_side_len" in global_cfg:
                config["max_side_len"] = _as_int(
                    global_cfg.get("max_side_len"), config["max_side_len"]
                )

            det_cfg = loaded.get("Det", {})
            if "device_name" in det_cfg:
                config["det_device_name"] = det_cfg.get("device_name")
            if "limit_side_len" in det_cfg:
                config["det_limit_side_len"] = _as_int(
                    det_cfg.get("limit_side_len"), config["det_limit_side_len"]
                )
            if "limit_type" in det_cfg:
                config["det_limit_type"] = _normalize_rapidocr_limit_type(
                    det_cfg.get("limit_type"),
                    config["det_limit_type"],
                )

            rec_cfg = loaded.get("Rec", {})
            if "device_name" in rec_cfg:
                config["rec_device_name"] = rec_cfg.get("device_name")
            if "rec_batch_num" in rec_cfg:
                config["rec_batch_num"] = max(
                    1, _as_int(rec_cfg.get("rec_batch_num"), config["rec_batch_num"])
                )

            cls_cfg = loaded.get("Cls", {})
            if "device_name" in cls_cfg:
                config["cls_device_name"] = cls_cfg.get("device_name")
            if "cls_batch_num" in cls_cfg:
                config["cls_batch_num"] = max(
                    1, _as_int(cls_cfg.get("cls_batch_num"), config["cls_batch_num"])
                )
        except Exception as exc:
            LOG.warning("Unable to parse RapidOCR config %s: %s", config_path, exc)
            raise

        # Explicit runtime env vars must override YAML to avoid stale cfg forcing unintended devices.
        env_override_map: Dict[str, Tuple[str, Any]] = {
            "RAPIDOCR_DEVICE": (
                "device_name",
                lambda value, default: _normalize_non_text_openvino_device(
                    str(value).strip() or default
                ),
            ),
            "RAPIDOCR_DET_DEVICE": (
                "det_device_name",
                lambda value, default: _normalize_non_text_openvino_device(
                    str(value).strip() or default
                ),
            ),
            "RAPIDOCR_CLS_DEVICE": (
                "cls_device_name",
                lambda value, default: _normalize_non_text_openvino_device(
                    str(value).strip() or default
                ),
            ),
            "RAPIDOCR_REC_DEVICE": (
                "rec_device_name",
                lambda value, default: _normalize_non_text_openvino_device(
                    str(value).strip() or default
                ),
            ),
            "RAPIDOCR_INFERENCE_NUM_THREADS": (
                "inference_num_threads",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_PERFORMANCE_HINT": (
                "performance_hint",
                lambda value, default: str(value).strip() or default,
            ),
            "RAPIDOCR_PERFORMANCE_NUM_REQUESTS": (
                "performance_num_requests",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_ENABLE_CPU_PINNING": (
                "enable_cpu_pinning",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_NUM_STREAMS": (
                "num_streams",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_ENABLE_HYPER_THREADING": (
                "enable_hyper_threading",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_SCHEDULING_CORE_TYPE": (
                "scheduling_core_type",
                lambda value, default: str(value).strip() or default,
            ),
            "RAPIDOCR_USE_CLS": (
                "use_cls",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_MAX_SIDE_LEN": (
                "max_side_len",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_DET_LIMIT_SIDE_LEN": (
                "det_limit_side_len",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_DET_LIMIT_TYPE": (
                "det_limit_type",
                lambda value, default: _normalize_rapidocr_limit_type(value, default),
            ),
            "RAPIDOCR_REC_BATCH_NUM": (
                "rec_batch_num",
                lambda value, default: max(1, _as_int(value, default)),
            ),
            "RAPIDOCR_CLS_BATCH_NUM": (
                "cls_batch_num",
                lambda value, default: max(1, _as_int(value, default)),
            ),
        }
        for env_name, (cfg_name, parser) in env_override_map.items():
            raw_env = os.environ.get(env_name)
            if raw_env is None or str(raw_env).strip() == "":
                continue
            config[cfg_name] = parser(raw_env, config[cfg_name])

        config["device_name"] = _normalize_non_text_openvino_device(
            config.get("device_name", _DEFAULT_NON_TEXT_OV_DEVICE)
        )
        for stage_name in ("det", "cls", "rec"):
            stage_key = f"{stage_name}_device_name"
            stage_value = config.get(stage_key)
            if stage_value in (None, ""):
                stage_value = _default_rapidocr_stage_device(config["device_name"], stage_name)
            config[stage_key] = _normalize_non_text_openvino_device(stage_value)
        if self.ov_cache_dir is not None:
            config["cache_dir"] = str(self.ov_cache_dir)
        config["runtime_device_name"] = _resolve_non_text_openvino_runtime_device(
            str(config["device_name"]),
            self.core.available_devices,
            consumer="openvino",
        )
        for stage_name in ("det", "cls", "rec"):
            stage_runtime_key = f"{stage_name}_runtime_device_name"
            stage_device_key = f"{stage_name}_device_name"
            config[stage_runtime_key] = _resolve_non_text_openvino_runtime_device(
                str(config[stage_device_key]),
                self.core.available_devices,
                consumer="openvino",
            )
        config["preprocess_backends"] = {
            "det": (
                "opencl"
                if _openvino_device_expr_requests_gpu(str(config["det_runtime_device_name"]))
                else "cpu"
            ),
            "cls": "cpu",
            "rec": (
                "opencl"
                if _openvino_device_expr_requests_gpu(str(config["rec_runtime_device_name"]))
                else "cpu"
            ),
        }
        return config

    def _require_rapidocr_local_assets(self) -> Dict[str, Path]:
        required_files = {
            "det": RAPIDOCR_V5_MOBILE_DET_FILE,
            "rec": RAPIDOCR_V5_MOBILE_REC_FILE,
            "dict": RAPIDOCR_V5_DICT_FILE,
            "cls": RAPIDOCR_CLS_MOBILE_V2_FILE,
        }
        resolved: Dict[str, Path] = {}
        missing: List[str] = []
        for key, filename in required_files.items():
            candidate = self.rapidocr_model_dir_path / filename
            if candidate.exists() and candidate.is_file():
                resolved[key] = candidate
            else:
                missing.append(str(candidate))

        if missing:
            joined = "; ".join(missing)
            raise FileNotFoundError(
                "RapidOCR local assets missing; online download fallback is disabled. "
                f"Missing files: {joined}"
            )
        return resolved

    def _build_rapidocr_runtime_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
            "EngineConfig.openvino.device_name": str(
                cfg.get("runtime_device_name", cfg.get("device_name", "AUTO"))
            ),
            "Det.device_name": str(cfg.get("det_runtime_device_name", cfg.get("det_device_name", "AUTO"))),
            "Cls.device_name": str(cfg.get("cls_runtime_device_name", cfg.get("cls_device_name", "AUTO"))),
            "Rec.device_name": str(cfg.get("rec_runtime_device_name", cfg.get("rec_device_name", "AUTO"))),
            "EngineConfig.openvino.inference_num_threads": _as_int(
                cfg.get("inference_num_threads"), -1
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
            "Global.use_cls": _as_bool(cfg.get("use_cls"), True),
            "Global.max_side_len": _as_int(cfg.get("max_side_len"), 960),
            "Det.limit_side_len": _as_int(cfg.get("det_limit_side_len"), 960),
            "Det.limit_type": _normalize_rapidocr_limit_type(cfg.get("det_limit_type"), "max"),
            "Rec.rec_batch_num": max(1, _as_int(cfg.get("rec_batch_num"), 6)),
            "Cls.cls_batch_num": max(1, _as_int(cfg.get("cls_batch_num"), 6)),
        }
        cache_dir = cfg.get("cache_dir")
        if cache_dir not in (None, ""):
            params["EngineConfig.openvino.cache_dir"] = str(cache_dir)
        if cfg.get("performance_hint") not in (None, ""):
            params["EngineConfig.openvino.performance_hint"] = str(
                cfg.get("performance_hint", "LATENCY")
            )

        if self.rapidocr_font_path:
            params["Global.font_path"] = self.rapidocr_font_path

        assets = self._require_rapidocr_local_assets()
        params["Det.model_path"] = str(assets["det"])
        params["Rec.model_path"] = str(assets["rec"])
        params["Rec.rec_keys_path"] = str(assets["dict"])
        # RapidOCR initializes cls session even when Global.use_cls=false.
        params["Cls.model_path"] = str(assets["cls"])

        return params

    @staticmethod
    def _resolve_rapidocr_stage_worker_count(cfg: Dict[str, Any]) -> int:
        return max(1, _as_int(cfg.get("performance_num_requests"), 1))

    def _reconfigure_rapidocr_stage_executors(self, cfg: Dict[str, Any]) -> None:
        worker_count = self._resolve_rapidocr_stage_worker_count(cfg)
        if worker_count == self._ocr_stage_worker_count:
            return

        old_executors = (
            self._ocr_det_executor,
            self._ocr_cls_executor,
            self._ocr_rec_executor,
        )
        self._ocr_det_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-det",
        )
        self._ocr_cls_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-cls",
        )
        self._ocr_rec_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-rec",
        )
        self._ocr_stage_worker_count = worker_count

        for executor in old_executors:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _validate_rapidocr_backend(engine: RapidOCR) -> None:
        cfg = getattr(engine, "cfg", None)
        if cfg is None:
            raise RuntimeError("RapidOCR initialized without runtime cfg metadata.")

        backend_errors: List[str] = []
        for section_name in ("Det", "Cls", "Rec"):
            section_cfg = getattr(cfg, section_name, None)
            engine_type = getattr(section_cfg, "engine_type", None)
            engine_name = str(getattr(engine_type, "value", engine_type)).strip().lower()
            if engine_name != EngineType.OPENVINO.value:
                backend_errors.append(f"{section_name}.engine_type={engine_name or 'missing'}")

        if backend_errors:
            raise RuntimeError(
                "RapidOCR backend validation failed after initialization: "
                f"{', '.join(backend_errors)}. No silent fallback is allowed."
            )

    def _instantiate_rapidocr(self, params: Dict[str, Any]) -> RapidOCR:
        rapidocr_device = str(params.get("EngineConfig.openvino.device_name", "AUTO")).upper()
        config_path = self._require_rapidocr_config_path()
        try:
            engine = RapidOCR(config_path=str(config_path), params=params)
        except Exception as exc:
            raise RuntimeError(
                f"RapidOCR 初始化失败，无法以 OpenVINO({rapidocr_device}) 配置启动。"
            ) from exc
        self._validate_rapidocr_backend(engine)
        return engine

    @staticmethod
    def _collect_rapidocr_execution_devices(engine: RapidOCR) -> Dict[str, List[str]]:
        execution_devices: Dict[str, List[str]] = {}
        for stage_name, attr_name in (
            ("det", "text_det"),
            ("cls", "text_cls"),
            ("rec", "text_rec"),
        ):
            stage = getattr(engine, attr_name, None)
            session = getattr(stage, "session", None)
            devices = _get_openvino_session_execution_devices(session)
            if devices:
                execution_devices[stage_name] = devices
        return execution_devices

    @staticmethod
    def _rapidocr_execution_matches_requested_device(
        requested_device: str,
        actual_devices: List[str],
    ) -> bool:
        normalized_actual = _normalize_openvino_devices(actual_devices)
        if not normalized_actual:
            return False
        if _openvino_device_expr_requests_gpu(requested_device):
            return _has_openvino_gpu_device(normalized_actual)
        if _openvino_device_expr_requests_cpu(requested_device):
            return any(device_name.startswith("CPU") for device_name in normalized_actual)
        if _openvino_device_expr_requests_npu(requested_device):
            return any(device_name.startswith("NPU") for device_name in normalized_actual)
        return False

    @staticmethod
    def _validate_rapidocr_execution_devices(
        cfg: Dict[str, Any],
        execution_devices: Dict[str, List[str]],
    ) -> None:
        mismatched_stages = [
            stage_name
            for stage_name in ("det", "cls", "rec")
            if not AIModels._rapidocr_execution_matches_requested_device(
                str(
                    cfg.get(
                        f"{stage_name}_runtime_device_name",
                        cfg.get(f"{stage_name}_device_name", cfg.get("runtime_device_name", "AUTO")),
                    )
                ),
                execution_devices.get(stage_name, []),
            )
        ]
        if mismatched_stages:
            expected_devices = {
                stage_name: str(
                    cfg.get(
                        f"{stage_name}_runtime_device_name",
                        cfg.get(f"{stage_name}_device_name", cfg.get("runtime_device_name", "AUTO")),
                    )
                )
                for stage_name in ("det", "cls", "rec")
            }
            raise RuntimeError(
                "RapidOCR execution device validation failed: "
                f"expected_devices={expected_devices}, execution_devices={execution_devices}, "
                f"mismatched_stages={mismatched_stages}. No silent fallback is allowed."
            )

    def _ensure_ocr_opencl_preprocess(self, cfg: Dict[str, Any]) -> None:
        preprocess_backends = cfg.get("preprocess_backends", {}) or {}
        if "opencl" not in preprocess_backends.values():
            return

        name, vendor = _ensure_intel_opencl_device("RapidOCR preprocessing")
        self._ocr_opencl_device_name = name
        self._ocr_opencl_device_vendor = vendor

    @staticmethod
    def _resize_image_opencl(image: np.ndarray, width: int, height: int) -> np.ndarray:
        resized = cv2.resize(cv2.UMat(image), (int(width), int(height)))
        if isinstance(resized, cv2.UMat):
            return _as_contiguous_bgr_uint8(resized.get(), context="RapidOCR OpenCL resize")
        return _as_contiguous_bgr_uint8(np.asarray(resized), context="RapidOCR OpenCL resize")

    @staticmethod
    def _rapidocr_blob_from_image_opencl(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> np.ndarray:
        prepared = _as_contiguous_bgr_uint8(image, context="RapidOCR OpenCL preprocess")
        mean = [float(value) for value in mean_values]
        std = [float(value) for value in std_values]
        if not std or max(std) - min(std) > 1e-6:
            raise RuntimeError(
                "RapidOCR OpenCL preprocess requires uniform std values per channel. "
                f"Got std={std_values}"
            )
        scalefactor = 1.0 / (255.0 * std[0])
        mean_pixels = tuple(float(value) * 255.0 for value in mean)
        blob = cv2.dnn.blobFromImage(
            cv2.UMat(prepared),
            scalefactor=scalefactor,
            size=(int(target_width), int(target_height)),
            mean=mean_pixels,
            swapRB=False,
            crop=False,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    @staticmethod
    def _rapidocr_resize_image_within_bounds_opencl(
        image: np.ndarray,
        min_side_len: float,
        max_side_len: float,
    ) -> Tuple[np.ndarray, float, float]:
        img = _as_contiguous_bgr_uint8(image, context="RapidOCR preprocess")
        h, w = img.shape[:2]
        ratio_h = ratio_w = 1.0

        max_value = max(h, w)
        if max_value > max_side_len:
            scale = float(max_side_len) / float(max_value)
            resize_h = int(round((h * scale) / 32.0) * 32)
            resize_w = int(round((w * scale) / 32.0) * 32)
            if resize_h <= 0 or resize_w <= 0:
                raise RapidOCRError("RapidOCR preprocess resize target is invalid.")
            img = AIModels._resize_image_opencl(img, resize_w, resize_h)
            ratio_h = h / float(resize_h)
            ratio_w = w / float(resize_w)

        h, w = img.shape[:2]
        min_value = min(h, w)
        if min_value < min_side_len:
            scale = float(min_side_len) / float(min_value)
            resize_h = int(round((h * scale) / 32.0) * 32)
            resize_w = int(round((w * scale) / 32.0) * 32)
            if resize_h <= 0 or resize_w <= 0:
                raise RapidOCRError("RapidOCR preprocess resize target is invalid.")
            img = AIModels._resize_image_opencl(img, resize_w, resize_h)
            ratio_h = h / float(resize_h)
            ratio_w = w / float(resize_w)
        return img, ratio_h, ratio_w

    @staticmethod
    def _rapidocr_det_resize_shape(
        image: np.ndarray,
        limit_side_len: int,
        limit_type: str,
    ) -> Optional[Tuple[int, int]]:
        h, w = image.shape[:2]
        if limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / float(max(h, w))
            else:
                ratio = 1.0
        else:
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / float(min(h, w))
            else:
                ratio = 1.0

        resize_h = int(round((h * ratio) / 32.0) * 32)
        resize_w = int(round((w * ratio) / 32.0) * 32)
        if resize_h <= 0 or resize_w <= 0:
            return None
        return resize_h, resize_w

    @staticmethod
    def _rapidocr_detect_opencl(detector: Any, image: np.ndarray) -> TextDetOutput:
        start_time = time.perf_counter()
        if image is None:
            raise ValueError("img is None")

        image_bgr = _as_contiguous_bgr_uint8(image, context="RapidOCR det")
        ori_img_shape = image_bgr.shape[0], image_bgr.shape[1]
        preprocess_op = detector.get_preprocess(max(image_bgr.shape[0], image_bgr.shape[1]))
        detector.preprocess_op = preprocess_op
        resize_shape = AIModels._rapidocr_det_resize_shape(
            image_bgr,
            int(preprocess_op.limit_side_len),
            str(preprocess_op.limit_type),
        )
        if resize_shape is None:
            return TextDetOutput()
        resize_h, resize_w = resize_shape
        prepro_img = AIModels._rapidocr_blob_from_image_opencl(
            image_bgr,
            target_width=resize_w,
            target_height=resize_h,
            mean_values=preprocess_op.mean.tolist(),
            std_values=preprocess_op.std.tolist(),
        )
        preds = detector.session(prepro_img)
        boxes, scores = detector.postprocess_op(preds, ori_img_shape)
        if len(boxes) < 1:
            return TextDetOutput()

        boxes = detector.sorted_boxes(boxes)
        return TextDetOutput(image_bgr, boxes, scores, elapse=time.perf_counter() - start_time)

    def _rapidocr_preprocess_img(
        self,
        engine: RapidOCR,
        ori_img: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        runtime_cfg = self._rapidocr_runtime_cfg or {}
        preprocess_backends = runtime_cfg.get("preprocess_backends", {}) or {}
        if preprocess_backends.get("det") != "opencl":
            return engine.preprocess_img(ori_img)

        op_record: Dict[str, Any] = {}
        img, ratio_h, ratio_w = self._rapidocr_resize_image_within_bounds_opencl(
            ori_img,
            engine.min_side_len,
            engine.max_side_len,
        )
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        return img, op_record

    def _warmup_rapidocr_locked(self) -> None:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")
        runtime_cfg = self._rapidocr_runtime_cfg or {}
        warmup_side = max(
            64,
            min(
                1280,
                max(
                    _as_int(runtime_cfg.get("max_side_len"), 960),
                    _as_int(runtime_cfg.get("det_limit_side_len"), 960),
                ),
            ),
        )
        warmup_image = np.full((warmup_side, warmup_side, 3), 255, dtype=np.uint8)
        font_scale = max(1.2, warmup_side / 512.0)
        thickness = max(2, warmup_side // 320)
        baseline_y = max(48, warmup_side // 3)
        cv2.putText(
            warmup_image,
            "rapidocr warmup",
            (max(12, warmup_side // 16), baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            warmup_image,
            "12345",
            (max(12, warmup_side // 8), min(warmup_side - 24, baseline_y * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (32, 32, 32),
            thickness,
            cv2.LINE_AA,
        )
        self._infer_ocr(warmup_image)

    def _load_rapidocr_locked(self) -> None:
        config = self._load_rapidocr_openvino_config()
        self._reconfigure_rapidocr_stage_executors(config)
        self._ensure_ocr_opencl_preprocess(config)
        rapidocr_params = self._build_rapidocr_runtime_params(config)
        self._rapidocr_engine = self._instantiate_rapidocr(rapidocr_params)
        setattr(self._rapidocr_engine.text_det, "_mt_runtime_device_name", config.get("det_runtime_device_name"))
        setattr(self._rapidocr_engine.text_cls, "_mt_runtime_device_name", config.get("cls_runtime_device_name"))
        setattr(self._rapidocr_engine.text_rec, "_mt_runtime_device_name", config.get("rec_runtime_device_name"))
        setattr(
            self._rapidocr_engine.text_det,
            "_mt_preprocess_backend",
            (config.get("preprocess_backends", {}) or {}).get("det", "cpu"),
        )
        setattr(
            self._rapidocr_engine.text_cls,
            "_mt_preprocess_backend",
            (config.get("preprocess_backends", {}) or {}).get("cls", "cpu"),
        )
        setattr(
            self._rapidocr_engine.text_rec,
            "_mt_preprocess_backend",
            (config.get("preprocess_backends", {}) or {}).get("rec", "cpu"),
        )
        execution_devices = self._collect_rapidocr_execution_devices(self._rapidocr_engine)
        self._validate_rapidocr_execution_devices(config, execution_devices)
        self._rapidocr_runtime_cfg = dict(config)
        self._rapidocr_runtime_cfg["execution_devices"] = execution_devices
        if config.get("det_limit_type") == "min":
            LOG.warning(
                "RapidOCR Det.limit_type=min will upscale small images and may increase latency."
            )
        LOG.info(
            "RapidOCR ready: config=%s device=%s stage_devices=%s stage_runtime_devices=%s exec_devices=%s preprocess_backends=%s opencl_device=%s hint=%s use_cls=%s max_side_len=%s "
            "det_limit=%s/%s rec_batch_num=%s cls_batch_num=%s ocr_stage_workers=%s",
            self.rapidocr_config_path,
            config.get("device_name"),
            {
                "det": config.get("det_device_name"),
                "cls": config.get("cls_device_name"),
                "rec": config.get("rec_device_name"),
            },
            {
                "det": config.get("det_runtime_device_name"),
                "cls": config.get("cls_runtime_device_name"),
                "rec": config.get("rec_runtime_device_name"),
            },
            execution_devices or "unknown",
            config.get("preprocess_backends"),
            (
                f"{self._ocr_opencl_device_name} ({self._ocr_opencl_device_vendor})"
                if self._ocr_opencl_device_name and self._ocr_opencl_device_vendor
                else "disabled"
            ),
            config.get("performance_hint"),
            config.get("use_cls"),
            config.get("max_side_len"),
            config.get("det_limit_type"),
            config.get("det_limit_side_len"),
            config.get("rec_batch_num"),
            config.get("cls_batch_num"),
            self._ocr_stage_worker_count,
        )

    def _build_openvino_preprocess_runner(
        self,
        runner_name: str,
        device_name: str,
        output_height: int,
        output_width: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> _OpenVinoPreprocessRunner:
        parameter = ov.opset13.parameter(
            ov.PartialShape([ov.Dimension.dynamic(), 3, int(output_height), int(output_width)]),
            ov.Type.f32,
            name=f"{runner_name}_input",
        )
        result = ov.opset13.result(parameter)
        result.set_friendly_name(f"{runner_name}_output")
        preprocess_model = ov.Model(
            [result],
            [parameter],
            f"{runner_name}_ppp",
        )
        ppp = ov.preprocess.PrePostProcessor(preprocess_model)
        ppp.input().tensor().set_shape([-1, -1, -1, 3]).set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")
        ).set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        ppp.input().preprocess().mean(mean_values)
        ppp.input().preprocess().scale(std_values)
        ppp.input().model().set_layout(ov.Layout("NCHW"))

        compiled = self.core.compile_model(
            ppp.build(),
            device_name,
            {
                "PERFORMANCE_HINT": "LATENCY",
            },
        )
        runner = _OpenVinoPreprocessRunner(
            compiled_model=compiled,
            input_port=compiled.input(0),
            output_port=compiled.output(0),
            runner_name=runner_name,
            input_height=int(output_height),
            input_width=int(output_width),
        )
        runner.validate()
        return runner

    @staticmethod
    def _enable_insightface_opencl_alignment() -> None:
        name, vendor = _ensure_intel_opencl_device("InsightFace alignment")
        LOG.info(
            "OpenCV OpenCL enabled for InsightFace alignment on Intel device: %s (%s).",
            name,
            vendor,
        )

    @staticmethod
    def _patch_insightface_norm_crop_opencl() -> None:
        from insightface.utils import face_align

        if getattr(face_align, "_mt_opencl_norm_crop_patch", False):
            return

        def _estimate_norm_no_warning(
            landmark: np.ndarray,
            image_size: int = 112,
            mode: str = "arcface",
        ) -> np.ndarray:
            _ = mode
            return _estimate_insightface_norm_matrix(
                landmark=landmark,
                image_size=int(image_size),
            )

        def _norm_crop_opencl(
            img: np.ndarray,
            landmark: np.ndarray,
            image_size: int = 112,
            mode: str = "arcface",
        ) -> np.ndarray:
            matrix = _estimate_norm_no_warning(landmark, image_size, mode)
            if matrix.shape != (2, 3):
                raise RuntimeError(
                    "InsightFace estimate_norm returned invalid affine matrix shape: "
                    f"{matrix.shape}"
                )
            source = _as_contiguous_bgr_uint8(img, context="InsightFace alignment")

            if not cv2.ocl.useOpenCL():
                raise RuntimeError(
                    "OpenCV OpenCL was disabled during InsightFace alignment. "
                    "No silent fallback is allowed."
                )
            try:
                warped_umat = cv2.warpAffine(
                    cv2.UMat(source),
                    matrix,
                    (int(image_size), int(image_size)),
                    borderValue=0.0,
                )
            except Exception as exc:
                raise RuntimeError(
                    "OpenCV OpenCL warpAffine failed in InsightFace alignment. "
                    "No silent fallback is allowed."
                ) from exc

            if isinstance(warped_umat, cv2.UMat):
                return _as_contiguous_bgr_uint8(warped_umat.get(), context="InsightFace alignment")
            return _as_contiguous_bgr_uint8(
                np.asarray(warped_umat),
                context="InsightFace alignment",
            )

        face_align.estimate_norm = _estimate_norm_no_warning
        face_align.norm_crop = _norm_crop_opencl
        face_align._mt_opencl_norm_crop_patch = True

    def _patch_face_detector_forward(
        self,
        det_model: Any,
        preprocess_runner: _OpenVinoPreprocessRunner,
    ) -> None:
        module = sys.modules.get(det_model.__class__.__module__)
        if module is None:
            raise RuntimeError(
                f"Cannot resolve module for detector class: {det_model.__class__.__module__}"
            )
        distance2bbox = getattr(module, "distance2bbox", None)
        distance2kps = getattr(module, "distance2kps", None)
        if distance2bbox is None:
            raise RuntimeError("InsightFace detector module missing distance2bbox helper.")
        if getattr(det_model, "use_kps", False) and distance2kps is None:
            raise RuntimeError("InsightFace detector module missing distance2kps helper.")

        def _forward_with_ppp(model_self: Any, img: np.ndarray, threshold: float) -> Any:
            scores_list: List[np.ndarray] = []
            bboxes_list: List[np.ndarray] = []
            kpss_list: List[np.ndarray] = []

            image_bgr = _as_contiguous_bgr_uint8(np.asarray(img), context="InsightFace detector")
            blob = preprocess_runner.run(image_bgr[np.newaxis, ...])
            net_outs = model_self.session.run(model_self.output_names, {model_self.input_name: blob})

            input_height = int(blob.shape[2])
            input_width = int(blob.shape[3])
            fmc = model_self.fmc
            batched_output = bool(getattr(model_self, "batched", False))

            for idx, stride in enumerate(model_self._feat_stride_fpn):
                if batched_output:
                    scores = net_outs[idx][0]
                    bbox_preds = net_outs[idx + fmc][0] * stride
                    if model_self.use_kps:
                        kps_preds = net_outs[idx + fmc * 2][0] * stride
                else:
                    scores = net_outs[idx]
                    bbox_preds = net_outs[idx + fmc] * stride
                    if model_self.use_kps:
                        kps_preds = net_outs[idx + fmc * 2] * stride

                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in model_self.center_cache:
                    anchor_centers = model_self.center_cache[key]
                else:
                    anchor_centers = np.stack(
                        np.mgrid[:height, :width][::-1], axis=-1
                    ).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    if model_self._num_anchors > 1:
                        anchor_centers = np.stack([anchor_centers] * model_self._num_anchors, axis=1)
                        anchor_centers = anchor_centers.reshape((-1, 2))
                    if len(model_self.center_cache) < 100:
                        model_self.center_cache[key] = anchor_centers

                pos_inds = np.where(scores >= threshold)[0]
                bboxes = distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                if model_self.use_kps:
                    kpss = distance2kps(anchor_centers, kps_preds)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)

            return scores_list, bboxes_list, kpss_list

        det_model.forward = types.MethodType(_forward_with_ppp, det_model)

    @staticmethod
    def _patch_face_recognition_get_feat(
        rec_model: Any,
        preprocess_runner: _OpenVinoPreprocessRunner,
    ) -> None:
        def _get_feat_with_ppp(model_self: Any, imgs: Any) -> np.ndarray:
            if not isinstance(imgs, list):
                imgs = [imgs]
            if not imgs:
                return np.empty((0, 0), dtype=np.float32)

            prepared = [
                _as_contiguous_bgr_uint8(np.asarray(item), context="InsightFace recognition")
                for item in imgs
            ]
            try:
                batch = np.stack(prepared, axis=0)
                blob = preprocess_runner.run(batch)
                return model_self.session.run(model_self.output_names, {model_self.input_name: blob})[0]
            except ValueError:
                features: List[np.ndarray] = []
                for item in prepared:
                    blob = preprocess_runner.run(item[np.newaxis, ...])
                    single = model_self.session.run(
                        model_self.output_names,
                        {model_self.input_name: blob},
                    )[0]
                    features.append(np.asarray(single))
                return np.concatenate(features, axis=0)

        rec_model.get_feat = types.MethodType(_get_feat_with_ppp, rec_model)

    def _attach_insightface_preprocess(self, face_app: FaceAnalysis, device_name: str) -> None:
        det_model = getattr(face_app, "det_model", None)
        rec_model = getattr(face_app, "models", {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("InsightFace detection/recognition model not found.")

        det_width, det_height = map(int, det_model.input_size)
        rec_width, rec_height = map(int, rec_model.input_size)

        self._face_det_ppp = self._build_openvino_preprocess_runner(
            runner_name="insightface_det",
            device_name=device_name,
            output_height=det_height,
            output_width=det_width,
            mean_values=_to_channel_triplet(getattr(det_model, "input_mean", 127.5)),
            std_values=_to_channel_triplet(getattr(det_model, "input_std", 127.5)),
        )
        self._face_rec_ppp = self._build_openvino_preprocess_runner(
            runner_name="insightface_rec",
            device_name=device_name,
            output_height=rec_height,
            output_width=rec_width,
            mean_values=_to_channel_triplet(getattr(rec_model, "input_mean", 127.5)),
            std_values=_to_channel_triplet(getattr(rec_model, "input_std", 127.5)),
        )

        self._patch_face_detector_forward(det_model, self._face_det_ppp)
        self._patch_face_recognition_get_feat(rec_model, self._face_rec_ppp)
        self._enable_insightface_opencl_alignment()
        self._patch_insightface_norm_crop_opencl()
        LOG.info("InsightFace preprocessing patched: OpenCV(OpenCL align) + OpenVINO PPP.")

    def _resolve_insightface_root(self) -> Path:
        candidate_roots = (self.insightface_model_root, self.insightface_root)
        for root in candidate_roots:
            if (root / MODEL_NAME).is_dir():
                return root
        raise FileNotFoundError(
            "InsightFace model directory missing for antelopev2. Checked paths: "
            f"{self.insightface_model_root / MODEL_NAME}, {self.insightface_root / MODEL_NAME}"
        )

    def _build_insightface_provider_options(self, provider_device: str) -> Dict[str, str]:
        provider_options: Dict[str, str] = {
            "device_type": provider_device,
            "enable_opencl_throttling": str(
                _as_bool(os.environ.get("INSIGHTFACE_OV_ENABLE_OPENCL_THROTTLING"), False)
            ).lower(),
        }
        if self.ov_cache_dir is not None:
            provider_options["cache_dir"] = str(self.ov_cache_dir)
        num_threads = _as_int(os.environ.get("INSIGHTFACE_OV_NUM_THREADS"), -1)
        if num_threads > 0:
            provider_options["num_of_threads"] = str(num_threads)
        return provider_options

    @staticmethod
    def _enforce_insightface_openvino_provider(
        face_app: FaceAnalysis,
        provider_options: Dict[str, str],
    ) -> None:
        models_map = getattr(face_app, "models", {})
        configured_sessions = 0
        for task_name, model in models_map.items():
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "set_providers"):
                continue
            try:
                session.set_providers(
                    ["OpenVINOExecutionProvider"],
                    [provider_options],
                )
            except TypeError:
                session.set_providers(
                    [("OpenVINOExecutionProvider", provider_options)]
                )
            except Exception as exc:
                raise RuntimeError(
                    "InsightFace failed to set OpenVINOExecutionProvider for task="
                    f"{task_name}. No silent fallback is allowed."
                ) from exc
            configured_sessions += 1

        if configured_sessions == 0:
            raise RuntimeError(
            "InsightFace session initialization missing; cannot enforce OpenVINOExecutionProvider."
            )

    @staticmethod
    def _is_insightface_init_kwargs_error(exc: Exception) -> bool:
        if not isinstance(exc, TypeError):
            return False
        message = str(exc)
        return any(
            keyword in message
            for keyword in ("providers", "provider_options", "allowed_modules")
        )

    def _instantiate_insightface_face_analysis(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> Tuple[FaceAnalysis, Path]:
        source_root = self._resolve_insightface_root()
        init_signature = inspect.signature(FaceAnalysis.__init__)
        init_parameters = init_signature.parameters
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in init_parameters.values()
        )
        supports_allowed_modules = "allowed_modules" in init_parameters
        supports_provider_kwargs = "providers" in init_parameters or supports_var_kwargs
        supports_provider_options = "provider_options" in init_parameters or supports_var_kwargs
        legacy_root: Optional[Path] = None

        def _build_kwargs(
            runtime_root: Path,
            *,
            include_provider_kwargs: bool,
            include_allowed_modules: bool,
        ) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "name": MODEL_NAME,
                "root": str(runtime_root),
            }
            if include_allowed_modules:
                kwargs["allowed_modules"] = ["detection", "recognition"]
            if include_provider_kwargs:
                kwargs["providers"] = list(provider_names)
                if supports_provider_options:
                    kwargs["provider_options"] = [dict(provider_options)]
            return kwargs

        attempts: List[Tuple[str, Path, Dict[str, Any]]] = [
            (
                "source-with-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=supports_provider_kwargs,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            (
                "source-without-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=False,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
        ]

        if legacy_root is None:
            legacy_root = self._prepare_legacy_insightface_runtime_root(source_root)
        attempts.extend(
            (
                (
                    "legacy-with-provider-kwargs",
                    legacy_root,
                    _build_kwargs(
                        legacy_root,
                        include_provider_kwargs=supports_provider_kwargs,
                        include_allowed_modules=supports_allowed_modules,
                    ),
                ),
                (
                    "legacy-minimal-kwargs",
                    legacy_root,
                    _build_kwargs(
                        legacy_root,
                        include_provider_kwargs=False,
                        include_allowed_modules=False,
                    ),
                ),
            )
        )

        seen_attempts: set[Tuple[str, str]] = set()
        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None
        for attempt_name, runtime_root, kwargs in attempts:
            attempt_key = (str(runtime_root), json.dumps(kwargs, sort_keys=True, ensure_ascii=True))
            if attempt_key in seen_attempts:
                continue
            seen_attempts.add(attempt_key)
            try:
                face_app = FaceAnalysis(**kwargs)
                if attempt_name != "source-with-provider-kwargs":
                    LOG.warning(
                        "InsightFace initialized via compatibility path %s (runtime_root=%s)",
                        attempt_name,
                        runtime_root,
                    )
                return face_app, runtime_root
            except Exception as exc:
                last_exc = exc
                attempt_errors.append(f"{attempt_name}: {exc}")
                if self._is_insightface_init_kwargs_error(exc):
                    LOG.warning(
                        "InsightFace init retry after unsupported kwargs on %s: %s",
                        attempt_name,
                        exc,
                    )
                else:
                    LOG.warning(
                        "InsightFace init attempt %s failed (runtime_root=%s): %s",
                        attempt_name,
                        runtime_root,
                        exc,
                    )

        summary = "; ".join(attempt_errors) or "no attempts executed"
        raise RuntimeError(
            "InsightFace initialization failed after compatibility retries: "
            f"{summary}"
        ) from last_exc

    def _prepare_legacy_insightface_runtime_root(self, source_root: Path) -> Path:
        source_model_dir = source_root / MODEL_NAME
        runtime_root = self.insightface_root / "_runtime_models"
        runtime_model_dir = runtime_root / "models" / MODEL_NAME
        runtime_model_dir.mkdir(parents=True, exist_ok=True)

        # insightface<=0.2.1 model router cannot parse all antelopev2 heads.
        # Keep only detection+recognition ONNX files required by /represent.
        required_files = ("scrfd_10g_bnkps.onnx", "glintr100.onnx")
        for filename in required_files:
            src = source_model_dir / filename
            dst = runtime_model_dir / filename
            if not src.is_file():
                raise FileNotFoundError(f"InsightFace required model missing: {src}")
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except Exception:
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
        return runtime_root

    @staticmethod
    def _validate_insightface_openvino_provider(
        face_app: FaceAnalysis,
        expected_device_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        models_map = getattr(face_app, "models", {})
        runtime_state: Dict[str, Dict[str, Any]] = {}
        normalized_expected = str(expected_device_type or "").strip().upper()
        for task_name, model in models_map.items():
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                continue
            providers = [str(item) for item in session.get_providers()]
            if not providers or providers[0] != "OpenVINOExecutionProvider":
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}, providers={providers}. "
                    "OpenVINOExecutionProvider must be the active primary provider. "
                    "No silent fallback is allowed."
                )
            task_state: Dict[str, Any] = {"providers": providers}
            get_provider_options = getattr(session, "get_provider_options", None)
            if callable(get_provider_options):
                try:
                    provider_options = get_provider_options()
                except Exception:
                    provider_options = None
                if isinstance(provider_options, dict):
                    openvino_options = provider_options.get("OpenVINOExecutionProvider")
                    if isinstance(openvino_options, dict):
                        normalized_options = {
                            str(key): str(value) for key, value in openvino_options.items()
                        }
                        task_state["openvino_options"] = normalized_options
                        actual_device = str(
                            normalized_options.get("device_type", "")
                        ).strip().upper()
                        if normalized_expected and actual_device and actual_device != normalized_expected:
                            raise RuntimeError(
                                "InsightFace provider validation failed for task="
                                f"{task_name}, expected device_type={normalized_expected} "
                                f"but got {actual_device}. No silent fallback is allowed."
                            )
            runtime_state[task_name] = task_state
        return runtime_state

    def _load_face_locked(self) -> None:
        configured_provider_device = _normalize_non_text_openvino_device(
            os.environ.get("INSIGHTFACE_OV_DEVICE", INFERENCE_DEVICE)
        )
        provider_device = _resolve_non_text_openvino_runtime_device(
            configured_provider_device,
            self.core.available_devices,
            consumer="ort_ep",
        )
        provider_options = self._build_insightface_provider_options(provider_device)
        provider_names = ["OpenVINOExecutionProvider"]
        try:
            face_app, root_for_runtime = self._instantiate_insightface_face_analysis(
                provider_names,
                provider_options,
            )
            # InsightFace<=0.2.1 ignores providers in __init__; enforce OpenVINO EP explicitly.
            self._enforce_insightface_openvino_provider(face_app, provider_options)
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            self._enforce_insightface_openvino_provider(face_app, provider_options)
            provider_runtime = self._validate_insightface_openvino_provider(
                face_app,
                expected_device_type=provider_device,
            )
            self._attach_insightface_preprocess(face_app, device_name=provider_device)
            ppp_execution_devices = {
                "det": _get_compiled_model_execution_devices(
                    getattr(self._face_det_ppp, "compiled_model", None)
                ),
                "rec": _get_compiled_model_execution_devices(
                    getattr(self._face_rec_ppp, "compiled_model", None)
                ),
            }
            self._face_engine = face_app
            LOG.info(
                "InsightFace loaded with providers=%s configured_device=%s runtime_device=%s provider_options=%s provider_runtime=%s ppp_execution_devices=%s (runtime_root=%s)",
                provider_names,
                configured_provider_device,
                provider_device,
                provider_options,
                provider_runtime,
                ppp_execution_devices,
                root_for_runtime,
            )
        except Exception as exc:
            self._face_engine = None
            self._face_det_ppp = None
            self._face_rec_ppp = None
            raise RuntimeError(
                "InsightFace must run with OpenVINOExecutionProvider + OpenCV OpenCL alignment. "
                "No silent fallback is allowed."
            ) from exc

    def _infer_clip_text_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._clip_text_model or not self._clip_text_request:
            raise RuntimeError("CLIP text model is not loaded.")
        if self._clip_text_input_names is None:
            raise RuntimeError("CLIP text input metadata is not initialized.")

        input_ids = _tokenize_for_clip(texts, context_length=CONTEXT_LENGTH)
        attention_mask = np.array(input_ids != _PAD_TOKEN_ID, dtype=np.int64)

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
            self._clip_text_request.set_input_tensor(
                0,
                ov.Tensor(np.ascontiguousarray(input_ids), shared_memory=True),
            )
            self._clip_text_request.set_input_tensor(
                1,
                ov.Tensor(np.ascontiguousarray(attention_mask), shared_memory=True),
            )
            self._clip_text_request.infer()
            embeddings = np.asarray(self._clip_text_request.get_output_tensor(0).data)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[-1] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid text embedding dims: expected={CLIP_EMBEDDING_DIMS}, got={embeddings.shape[-1]}"
            )
        return embeddings.astype(np.float32, copy=False).tolist()

    def _infer_clip_image_tensor_batch(self, tensors: List[np.ndarray]) -> List[List[float]]:
        if not self._clip_vision_model or not self._clip_vision_request:
            raise RuntimeError("CLIP vision model is not loaded.")
        if not tensors:
            return []

        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        self._clip_vision_request.set_input_tensor(0, ov.Tensor(batch, shared_memory=True))
        self._clip_vision_request.infer()
        embeddings = np.asarray(self._clip_vision_request.get_output_tensor(0).data)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape != (len(tensors), CLIP_EMBEDDING_DIMS):
            raise RuntimeError(
                "Invalid image embedding dims for CLIP vision batch: "
                f"expected={(len(tensors), CLIP_EMBEDDING_DIMS)}, got={tuple(embeddings.shape)}"
            )
        return [
            embeddings[index].astype(np.float32, copy=False).tolist()
            for index in range(len(tensors))
        ]

    def _infer_ocr(self, image: np.ndarray) -> OCRResult:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._infer_ocr_async(image))
        raise RuntimeError("Synchronous OCR inference cannot run inside an active event loop.")

    @staticmethod
    def _ocr_result_from_raw(raw_result: Any) -> OCRResult:
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

    @staticmethod
    def _run_in_executor(
        executor: ThreadPoolExecutor,
        func: Callable[..., Any],
        *args: Any,
    ) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(executor, partial(func, *args))

    def _rapidocr_detect_and_pad(
        self,
        engine: RapidOCR,
        image: np.ndarray,
        op_record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any], TextDetOutput]:
        padded, updated_record = apply_vertical_padding(
            image,
            op_record,
            engine.width_height_ratio,
            engine.min_height,
        )
        detector = engine.text_det
        preprocess_backend = str(getattr(detector, "_mt_preprocess_backend", "cpu")).strip().lower()
        if preprocess_backend == "opencl":
            return padded, updated_record, self._rapidocr_detect_opencl(detector, padded)
        return padded, updated_record, detector(padded)

    def _rapidocr_crop_regions(self, image: np.ndarray, det_boxes: np.ndarray) -> List[np.ndarray]:
        boxes = np.asarray(det_boxes, dtype=np.float32)
        if boxes.size == 0:
            return []
        futures = [
            self._shared_cpu_executor.submit(
                get_rotate_crop_image,
                image,
                np.array(box, dtype=np.float32, copy=True),
            )
            for box in boxes
        ]
        return [future.result() for future in futures]

    @staticmethod
    def _rapidocr_classify_single_image(classifier: Any, image: np.ndarray) -> Tuple[str, float]:
        norm_img = classifier.resize_norm_img(image)[np.newaxis, ...].astype(np.float32)
        prob_out = classifier.session(norm_img)
        cls_result = list(classifier.postprocess_op(prob_out))
        if len(cls_result) != 1:
            raise RapidOCRError(
                "RapidOCR cls single-image fallback output mismatch: "
                f"expected=1 got={len(cls_result)}"
            )
        label, score = cls_result[0]
        return str(label), float(score)

    @staticmethod
    def _rapidocr_build_sorted_batch_indices(
        images: List[np.ndarray],
        batch_num: int,
    ) -> List[List[int]]:
        if not images:
            return []

        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        indices = np.argsort(np.array(width_list))
        batch_size = max(1, int(batch_num))
        return [
            [int(indices[ino]) for ino in range(beg_img_no, min(len(images), beg_img_no + batch_size))]
            for beg_img_no in range(0, len(images), batch_size)
        ]

    @staticmethod
    def _rapidocr_classify_batch(
        engine: RapidOCR,
        images: List[np.ndarray],
        batch_indices: List[int],
    ) -> Tuple[List[int], List[np.ndarray], List[Tuple[str, float]]]:
        classifier = engine.text_cls
        batch_images = [images[original_index] for original_index in batch_indices]
        if not batch_images:
            return batch_indices, [], []

        norm_img_batch = np.stack(
            [classifier.resize_norm_img(image) for image in batch_images],
            axis=0,
        ).astype(np.float32)
        prob_out = classifier.session(norm_img_batch)
        cls_result = list(classifier.postprocess_op(prob_out))
        if len(cls_result) != len(batch_indices):
            LOG.warning(
                "RapidOCR cls output size mismatch: expected=%s got=%s; "
                "falling back to single-image classification.",
                len(batch_indices),
                len(cls_result),
            )
            cls_result = [
                AIModels._rapidocr_classify_single_image(classifier, image)
                for image in batch_images
            ]

        rotated_images = list(batch_images)
        normalized_results: List[Tuple[str, float]] = []
        for relative_index, (label, score) in enumerate(cls_result):
            label_str = str(label)
            score_float = float(score)
            normalized_results.append((label_str, score_float))
            if "180" in label_str and score_float > classifier.cls_thresh:
                rotated_images[relative_index] = cv2.rotate(
                    rotated_images[relative_index],
                    cv2.ROTATE_180,
                )
        return batch_indices, rotated_images, normalized_results

    @staticmethod
    def _rapidocr_recognize_single_batch(
        recognizer: Any,
        images: List[np.ndarray],
        *,
        return_word_box: bool,
    ) -> TextRecOutput:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            return TextRecOutput(imgs=[], txts=(), scores=[], word_results=(), elapse=0.0)

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        img_num = len(img_list)
        rec_res: List[Tuple[Tuple[str, float], Any]] = [(("", 0.0), None)] * img_num

        img_c, img_h, img_w = recognizer.rec_image_shape[:3]
        max_wh_ratio = img_w / img_h
        wh_ratio_list: List[float] = []
        ordered_images = [img_list[int(indices[ino])] for ino in range(img_num)]
        for image in ordered_images:
            h, w = image.shape[:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)

        target_width = int(max(img_w, round(img_h * max_wh_ratio)))
        preprocess_backend = str(getattr(recognizer, "_mt_preprocess_backend", "cpu")).strip().lower()
        norm_img_batch: List[np.ndarray] = []
        for image in ordered_images:
            if preprocess_backend == "opencl":
                ratio = image.shape[1] / float(image.shape[0])
                resized_w = min(target_width, int(np.ceil(img_h * ratio)))
                blob = AIModels._rapidocr_blob_from_image_opencl(
                    image,
                    target_width=resized_w,
                    target_height=int(img_h),
                    mean_values=[0.5, 0.5, 0.5],
                    std_values=[0.5, 0.5, 0.5],
                )[0]
                padding_im = np.zeros((img_c, img_h, target_width), dtype=np.float32)
                padding_im[:, :, :resized_w] = blob[:, :, :resized_w]
                norm_img_batch.append(padding_im)
            else:
                norm_img_batch.append(recognizer.resize_norm_img(image, max_wh_ratio))

        batch = np.stack(norm_img_batch, axis=0).astype(np.float32)
        preds = recognizer.session(batch)
        line_results, word_results = recognizer.postprocess_op(
            preds,
            return_word_box,
            wh_ratio_list=wh_ratio_list,
            max_wh_ratio=max_wh_ratio,
        )

        for result_index, one_res in enumerate(line_results):
            original_index = int(indices[result_index])
            if return_word_box:
                rec_res[original_index] = (one_res, word_results[result_index])
            else:
                rec_res[original_index] = (one_res, None)

        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        if recognizer.cfg.lang_type == LangRec.ARABIC:
            txts = reorder_bidi_for_display(txts)

        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            time.perf_counter() - start_time,
            viser=VisRes(lang_type=recognizer.cfg.lang_type, font_path=recognizer.cfg.font_path),
        )

    @staticmethod
    def _rapidocr_recognize(engine: RapidOCR, images: List[np.ndarray]) -> TextRecOutput:
        rec_res = AIModels._rapidocr_recognize_single_batch(
            engine.text_rec,
            images,
            return_word_box=engine.return_word_box,
        )
        if rec_res.txts is None:
            raise RapidOCRError("The text recognize result is empty")
        return rec_res

    @staticmethod
    def _rapidocr_recognize_batch(
        engine: RapidOCR,
        images: List[np.ndarray],
        batch_indices: List[int],
    ) -> Tuple[List[int], TextRecOutput]:
        batch_images = [images[original_index] for original_index in batch_indices]
        return batch_indices, AIModels._rapidocr_recognize(engine, batch_images)

    async def _rapidocr_cls_and_rotate_async(
        self,
        engine: RapidOCR,
        images: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], TextClsOutput]:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            output = TextClsOutput(img_list=[], cls_res=[], elapse=0.0)
            return [], output

        batch_indices_list = self._rapidocr_build_sorted_batch_indices(
            img_list,
            engine.text_cls.cls_batch_num,
        )
        batch_outputs = await asyncio.gather(
            *[
                self._run_in_executor(
                    self._ocr_cls_executor,
                    self._rapidocr_classify_batch,
                    engine,
                    img_list,
                    batch_indices,
                )
                for batch_indices in batch_indices_list
            ]
        )

        cls_res: List[Tuple[str, float]] = [("", 0.0)] * len(img_list)
        for batch_indices, rotated_images, batch_results in batch_outputs:
            if len(rotated_images) != len(batch_indices) or len(batch_results) != len(batch_indices):
                raise RapidOCRError(
                    "RapidOCR cls batch output mismatch: "
                    f"expected={len(batch_indices)} got_images={len(rotated_images)} "
                    f"got_results={len(batch_results)}"
                )
            for relative_index, original_index in enumerate(batch_indices):
                img_list[original_index] = rotated_images[relative_index]
                cls_res[original_index] = batch_results[relative_index]

        output = TextClsOutput(
            img_list=img_list,
            cls_res=cls_res,
            elapse=time.perf_counter() - start_time,
        )
        if output.img_list is None:
            raise RapidOCRError("The text classifier is empty")
        return output.img_list, output

    async def _rapidocr_recognize_async(
        self,
        engine: RapidOCR,
        images: List[np.ndarray],
    ) -> TextRecOutput:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            return TextRecOutput(imgs=[], txts=(), scores=[], word_results=(), elapse=0.0)

        batch_indices_list = self._rapidocr_build_sorted_batch_indices(
            img_list,
            engine.text_rec.rec_batch_num,
        )
        batch_outputs = await asyncio.gather(
            *[
                self._run_in_executor(
                    self._ocr_rec_executor,
                    self._rapidocr_recognize_batch,
                    engine,
                    img_list,
                    batch_indices,
                )
                for batch_indices in batch_indices_list
            ]
        )

        texts: List[str] = [""] * len(img_list)
        scores: List[float] = [0.0] * len(img_list)
        word_results: List[Any] = [None] * len(img_list)
        viser: Any = None
        for batch_indices, batch_output in batch_outputs:
            batch_texts = list(batch_output.txts or ())
            batch_scores = [float(score) for score in (batch_output.scores or ())]
            batch_word_results = list(batch_output.word_results or ())
            if (
                len(batch_texts) != len(batch_indices)
                or len(batch_scores) != len(batch_indices)
                or len(batch_word_results) != len(batch_indices)
            ):
                raise RapidOCRError(
                    "RapidOCR rec batch output mismatch: "
                    f"expected={len(batch_indices)} got_txts={len(batch_texts)} "
                    f"got_scores={len(batch_scores)} got_words={len(batch_word_results)}"
                )
            if viser is None:
                viser = batch_output.viser
            for relative_index, original_index in enumerate(batch_indices):
                texts[original_index] = str(batch_texts[relative_index])
                scores[original_index] = batch_scores[relative_index]
                word_results[original_index] = batch_word_results[relative_index]

        return TextRecOutput(
            imgs=img_list,
            txts=tuple(texts),
            scores=scores,
            word_results=tuple(word_results),
            elapse=time.perf_counter() - start_time,
            viser=viser,
        )

    async def _infer_ocr_async(self, image: np.ndarray) -> OCRResult:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")

        engine = self._rapidocr_engine
        total_started_at = time.perf_counter()
        ori_img = _as_contiguous_bgr_uint8(image, context="OCR")
        preprocess_ms = det_ms = crop_ms = cls_ms = rec_ms = assemble_ms = 0.0
        stage_started_at = time.perf_counter()
        img, op_record = await self._run_in_executor(
            self._shared_cpu_executor,
            self._rapidocr_preprocess_img,
            engine,
            ori_img,
        )
        preprocess_ms = (time.perf_counter() - stage_started_at) * 1000.0
        det_res, cls_res, rec_res = TextDetOutput(), TextClsOutput(), TextRecOutput()

        if engine.use_det:
            try:
                stage_started_at = time.perf_counter()
                img, op_record, det_res = await self._run_in_executor(
                    self._ocr_det_executor,
                    self._rapidocr_detect_and_pad,
                    engine,
                    img,
                    op_record,
                )
                det_ms = (time.perf_counter() - stage_started_at) * 1000.0
            except RapidOCRError as exc:
                LOG.warning(exc)
                return OCRResult(texts=[], scores=[], boxes=[])
            if det_res.boxes is None:
                return OCRResult(texts=[], scores=[], boxes=[])
            stage_started_at = time.perf_counter()
            cropped_img_list = await self._run_in_executor(
                self._shared_cpu_executor,
                self._rapidocr_crop_regions,
                img,
                det_res.boxes,
            )
            crop_ms = (time.perf_counter() - stage_started_at) * 1000.0
        else:
            cropped_img_list = [img]

        if engine.use_cls:
            try:
                stage_started_at = time.perf_counter()
                cls_img_list, cls_res = await self._rapidocr_cls_and_rotate_async(
                    engine,
                    cropped_img_list,
                )
                cls_ms = (time.perf_counter() - stage_started_at) * 1000.0
            except RapidOCRError as exc:
                LOG.warning(exc)
                return OCRResult(texts=[], scores=[], boxes=[])
        else:
            cls_img_list = cropped_img_list

        if engine.use_rec:
            try:
                stage_started_at = time.perf_counter()
                rec_res = await self._rapidocr_recognize_async(
                    engine,
                    cls_img_list,
                )
                rec_ms = (time.perf_counter() - stage_started_at) * 1000.0
            except RapidOCRError as exc:
                LOG.warning(exc)
                return OCRResult(texts=[], scores=[], boxes=[])

        stage_started_at = time.perf_counter()
        raw_result = await self._run_in_executor(
            self._shared_cpu_executor,
            engine.build_final_output,
            ori_img,
            det_res,
            cls_res,
            rec_res,
            cropped_img_list,
            op_record,
        )
        assemble_ms = (time.perf_counter() - stage_started_at) * 1000.0
        total_ms = (time.perf_counter() - total_started_at) * 1000.0
        slow_threshold_ms = max(
            1000.0,
            min(float(self._ocr_execution_timeout_seconds) * 500.0, 5000.0),
        )
        if total_ms >= slow_threshold_ms:
            runtime_cfg = self._rapidocr_runtime_cfg or {}
            box_count = 0 if det_res.boxes is None else int(len(det_res.boxes))
            LOG.warning(
                "RapidOCR slow request: total=%.1fms preprocess=%.1fms det=%.1fms crop=%.1fms cls=%.1fms rec=%.1fms assemble=%.1fms boxes=%s stage_runtime_devices=%s exec_devices=%s preprocess_backends=%s",
                total_ms,
                preprocess_ms,
                det_ms,
                crop_ms,
                cls_ms,
                rec_ms,
                assemble_ms,
                box_count,
                {
                    "det": runtime_cfg.get("det_runtime_device_name"),
                    "cls": runtime_cfg.get("cls_runtime_device_name"),
                    "rec": runtime_cfg.get("rec_runtime_device_name"),
                },
                runtime_cfg.get("execution_devices", "unknown"),
                runtime_cfg.get("preprocess_backends", "unknown"),
            )
        return self._ocr_result_from_raw(raw_result)

    def _infer_face(self, image: np.ndarray) -> List[RepresentResult]:
        if self._face_engine is None:
            raise RuntimeError("Face model is not loaded.")

        face_input = _as_contiguous_bgr_uint8(image, context="InsightFace")
        faces = self._face_engine.get(face_input)
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
        self._ensure_text_service_ready(preload=True)

    async def ensure_clip_text_model_loaded_async(self) -> None:
        await asyncio.to_thread(self.ensure_clip_text_model_loaded)

    def release_models(self) -> None:
        self._release_non_text_models_sync(reason="manual")

    def release_models_for_restart(self, text_restore_delay_seconds: Optional[float] = None) -> None:
        _ = text_restore_delay_seconds
        self._release_non_text_models_sync(reason="restart")

    def release_all_models(self) -> None:
        self._background_prewarm_cancel.set()
        self._release_non_text_models_sync(
            reason="shutdown",
            cancel_background_prewarm=False,
            join_background_prewarm=True,
        )

        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            pending_tasks = list(self._normal_queue)
            self._normal_queue.clear()
            self._condition.notify_all()

        for task in pending_tasks:
            self._safe_set_exception(task.future, RuntimeError("模型服务正在关闭"))

        self._worker.join()
        self._join_background_prewarm_thread(
            timeout_seconds=max(2.0, float(self._execution_timeout_seconds))
        )

        for executor in (
            self._shared_cpu_executor,
            self._ocr_det_executor,
            self._ocr_cls_executor,
            self._ocr_rec_executor,
            self._face_executor,
        ):
            executor.shutdown(wait=True, cancel_futures=True)

        self._shutdown_text_service()
        with self._model_lock:
            self._unload_everything_locked()
        LOG.info("All models released.")

    def get_text_embedding(self, text: str) -> List[float]:
        return self._request_text_embedding_remote(text)

    async def get_text_embedding_async(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.get_text_embedding, text)

    def get_image_embedding(self, image: np.ndarray, filename: str = "unknown") -> List[float]:
        _ = filename
        lease_bound = False
        self._acquire_non_text_family_lease("vision")
        try:
            self._ensure_clip_vision_loaded()
            payload = self._preprocess_clip_image_tensor(image)
            task = self._submit_task(kind="clip_img", payload=payload)
            self._bind_non_text_lease_to_future("vision", task.future)
            lease_bound = True
            return self._wait_task(task)
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("vision")

    async def get_image_embedding_async(
        self, image: np.ndarray, filename: str = "unknown"
    ) -> List[float]:
        _ = filename
        lease_bound = False
        await asyncio.to_thread(self._acquire_non_text_family_lease, "vision")
        try:
            await asyncio.to_thread(self._ensure_clip_vision_loaded)
            payload = await self._run_in_executor(
                self._shared_cpu_executor,
                self._preprocess_clip_image_tensor,
                image,
            )
            task = self._submit_task(kind="clip_img", payload=payload)
            self._bind_non_text_lease_to_future("vision", task.future)
            lease_bound = True
            return await self._await_task(task)
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("vision")

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self._acquire_non_text_family_lease("ocr")
        try:
            self._ensure_rapidocr_loaded()
            return self._infer_ocr(image)
        finally:
            self._release_non_text_family_lease("ocr")

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        lease_bound = False
        await asyncio.to_thread(self._acquire_non_text_family_lease, "ocr")
        try:
            await asyncio.to_thread(self._ensure_rapidocr_loaded)
            task = asyncio.create_task(self._infer_ocr_async(image))
            self._bind_non_text_lease_to_future("ocr", task)
            lease_bound = True
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self._ocr_execution_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                task.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="OCR task")
                )
                raise RuntimeError(
                    f"推理任务执行超时（>{self._ocr_execution_timeout_seconds}s）"
                ) from exc
            except asyncio.CancelledError:
                task.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="OCR task")
                )
                raise
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("ocr")

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        self._acquire_non_text_family_lease("face")
        try:
            self._ensure_face_loaded()
            return self._infer_face(image)
        finally:
            self._release_non_text_family_lease("face")

    async def get_face_representation_async(self, image: np.ndarray) -> List[RepresentResult]:
        lease_bound = False
        await asyncio.to_thread(self._acquire_non_text_family_lease, "face")
        try:
            await asyncio.to_thread(self._ensure_face_loaded)
            future = self._run_in_executor(self._face_executor, self._infer_face, image)
            self._bind_non_text_lease_to_future("face", future)
            lease_bound = True
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self._execution_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                future.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="Face task")
                )
                raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc
            except asyncio.CancelledError:
                future.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="Face task")
                )
                raise
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("face")
