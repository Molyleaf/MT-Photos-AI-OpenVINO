import asyncio
import os
import socketserver
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, cast

import cv2
import numpy as np
import openvino as ov

from .constants import (
    DEFAULT_NON_TEXT_OV_DEVICE,
    PROCESS_LOCK_POLL_SECONDS,
)

if os.name == "nt":
    import msvcrt  # type: ignore[attr-defined]

    fcntl = None  # type: ignore[assignment]
else:
    import fcntl  # type: ignore[attr-defined]

    msvcrt = None  # type: ignore[assignment]


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
        return DEFAULT_NON_TEXT_OV_DEVICE

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


def _ensure_intel_opencl_device(feature_name: str) -> Tuple[str, str]:
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.haveOpenCL() or not cv2.ocl.useOpenCL():
        raise RuntimeError(
            f"{feature_name} requires OpenCV OpenCL, but OpenCL is unavailable. "
            "No silent fallback is allowed."
        )

    device = cv2.ocl.Device.getDefault()
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


def _to_opencv_umat(image: np.ndarray) -> cv2.UMat:
    # OpenCV Python accepts ndarray-backed UMat construction, but current stubs miss that overload.
    return cv2.UMat(cast(Any, image))


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
                time.sleep(PROCESS_LOCK_POLL_SECONDS)

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
            request = self.compiled_model.create_infer_request()
            self._request_local.request = request
        return request

    def run(self, tensor: np.ndarray) -> np.ndarray:
        prepared = np.ascontiguousarray(tensor, dtype=np.uint8)
        request = self._get_request()
        request.set_tensor(self.input_port, ov.Tensor(prepared, shared_memory=True))
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


class _InferenceCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class _ManagedLease:
    label: str
    _callbacks: List[Callable[[], None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _released: bool = False

    def push(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._released:
                callback()
                return
            self._callbacks.append(callback)

    def bind_future(self, future: Any) -> None:
        future.add_done_callback(lambda _future: self.release())

    def release(self) -> None:
        callbacks: List[Callable[[], None]]
        with self._lock:
            if self._released:
                return
            self._released = True
            callbacks = list(reversed(self._callbacks))
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    async def release_async(self) -> None:
        await asyncio.to_thread(self.release)


class _AdmissionController:
    def __init__(self, name: str, capacity: int) -> None:
        self.name = str(name)
        self._capacity = max(1, int(capacity))
        self._active = 0
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        with self._condition:
            return self._capacity

    def resize(self, capacity: int) -> None:
        with self._condition:
            self._capacity = max(1, int(capacity))
            self._condition.notify_all()

    def acquire(self, timeout: Optional[float]) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._active >= self._capacity:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()
            self._active += 1
            return True

    async def acquire_async(self, timeout: Optional[float]) -> bool:
        return await asyncio.to_thread(self.acquire, timeout)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                return
            self._active -= 1
            self._condition.notify_all()


class _TextClipRpcServer(socketserver.TCPServer):
    allow_reuse_address = True


NonTextFamily = Literal["vision", "ocr", "face"]


@dataclass(slots=True)
class _ClipImageTask:
    payload: Any
    future: Future
    created_at: float
    started_at: Optional[float] = None
    started_event: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)


@dataclass(slots=True)
class _FaceInferenceTask:
    payload: Any
    future: Future
    created_at: float
    started_at: Optional[float] = None
    started_event: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
