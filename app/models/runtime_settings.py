import os
from dataclasses import dataclass
from pathlib import Path

from .common import _as_bool, _as_float, _as_int
from .constants import (
    CLIP_INFERENCE_DEVICE,
    EXEC_TIMEOUT_SECONDS,
    INSIGHTFACE_PREPROCESS_WORKERS,
    INSIGHTFACE_REQUEST_CAPACITY,
    MAX_PENDING_IMAGE_REQUESTS,
    PROJECT_ROOT,
    QUEUE_MAX_SIZE,
    QUEUE_TIMEOUT_SECONDS,
)


@dataclass(frozen=True, slots=True)
class RuntimePathSettings:
    model_base_path: Path
    insightface_root: Path
    insightface_model_root: Path
    qa_clip_path: Path
    ov_cache_dir: Path
    runtime_state_dir: Path


@dataclass(frozen=True, slots=True)
class ClipImageRuntimeSettings:
    inference_device: str
    batch_size: int
    batch_wait_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionControlSettings:
    configured_queue_capacity: int
    queue_capacity: int
    queue_timeout_seconds: int
    execution_timeout_seconds: int
    ocr_execution_timeout_seconds: int
    idle_release_timeout_seconds: float
    ocr_admission_capacity: int
    face_preprocess_worker_count: int
    face_batch_size: int
    face_batch_wait_seconds: float
    face_queue_capacity: int
    ocr_prewarm_enabled: bool
    ocr_prewarm_delay_seconds: float


def load_runtime_path_settings() -> RuntimePathSettings:
    model_base_path = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models")))
    ov_cache_dir_raw = str(os.environ.get("OV_CACHE_DIR", "")).strip()
    if ov_cache_dir_raw:
        ov_cache_dir = Path(ov_cache_dir_raw).expanduser().resolve()
    else:
        ov_cache_dir = (PROJECT_ROOT / "cache" / "openvino").resolve()
    ov_cache_dir.mkdir(parents=True, exist_ok=True)

    runtime_state_dir = (PROJECT_ROOT / "cache" / "runtime").resolve()
    runtime_state_dir.mkdir(parents=True, exist_ok=True)

    return RuntimePathSettings(
        model_base_path=model_base_path,
        insightface_root=model_base_path / "insightface",
        insightface_model_root=model_base_path / "insightface" / "models",
        qa_clip_path=model_base_path / "qa-clip" / "openvino",
        ov_cache_dir=ov_cache_dir,
        runtime_state_dir=runtime_state_dir,
    )


def load_clip_image_runtime_settings() -> ClipImageRuntimeSettings:
    return ClipImageRuntimeSettings(
        inference_device=CLIP_INFERENCE_DEVICE,
        batch_size=max(
            1,
            _as_int(
                os.environ.get("CLIP_IMAGE_BATCH", os.environ.get("CLIP_IMAGE_BATCH_SIZE")),
                8,
            ),
        ),
        batch_wait_seconds=max(
            0.0,
            _as_float(os.environ.get("CLIP_IMAGE_BATCH_WAIT_MS"), 5.0) / 1000.0,
        ),
    )


def load_execution_control_settings() -> ExecutionControlSettings:
    configured_queue_capacity = max(1, QUEUE_MAX_SIZE)
    queue_capacity = min(MAX_PENDING_IMAGE_REQUESTS, configured_queue_capacity)
    execution_timeout_seconds = max(1, EXEC_TIMEOUT_SECONDS)
    ocr_admission_capacity = max(
        1,
        min(
            queue_capacity,
            _as_int(os.environ.get("OCR_MAX_CONCURRENT_REQUESTS"), 4),
        ),
    )
    face_batch_size = max(1, min(queue_capacity, INSIGHTFACE_REQUEST_CAPACITY))

    return ExecutionControlSettings(
        configured_queue_capacity=configured_queue_capacity,
        queue_capacity=queue_capacity,
        queue_timeout_seconds=max(1, QUEUE_TIMEOUT_SECONDS),
        execution_timeout_seconds=execution_timeout_seconds,
        ocr_execution_timeout_seconds=max(
            1,
            _as_int(
                os.environ.get("OCR_EXEC_TIMEOUT"),
                max(30, execution_timeout_seconds),
            ),
        ),
        idle_release_timeout_seconds=max(
            0.0,
            _as_float(os.environ.get("NON_TEXT_IDLE_RELEASE_SECONDS"), 60.0),
        ),
        ocr_admission_capacity=ocr_admission_capacity,
        face_preprocess_worker_count=max(
            1,
            min(
                queue_capacity,
                INSIGHTFACE_REQUEST_CAPACITY,
                INSIGHTFACE_PREPROCESS_WORKERS,
                os.cpu_count() or INSIGHTFACE_PREPROCESS_WORKERS,
            ),
        ),
        face_batch_size=face_batch_size,
        face_batch_wait_seconds=max(
            0.0,
            _as_float(os.environ.get("INSIGHTFACE_BATCH_WAIT_MS"), 5.0) / 1000.0,
        ),
        face_queue_capacity=face_batch_size,
        ocr_prewarm_enabled=_as_bool(os.environ.get("OCR_PREWARM_ENABLED"), False),
        ocr_prewarm_delay_seconds=max(
            0.0,
            _as_float(os.environ.get("OCR_PREWARM_DELAY_SECONDS"), 1.0),
        ),
    )
