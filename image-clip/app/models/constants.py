import logging
import os
from pathlib import Path

import numpy as np


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
IMAGE_CLIP_ROOT = APP_DIR.parent
PROJECT_ROOT = IMAGE_CLIP_ROOT.parent

MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models"))
).expanduser().resolve()
HF_CACHE_DIR = Path(
    os.environ.get("HF_CACHE_DIR", str(PROJECT_ROOT / "cache" / "huggingface"))
).expanduser().resolve()

IMAGE_CLIP_DEVICE = str(os.environ.get("IMAGE_CLIP_DEVICE", "cuda")).strip() or "cuda"
IMAGE_CLIP_USE_FP16 = _env_flag("IMAGE_CLIP_USE_FP16", False)
HF_LOCAL_FILES_ONLY = _env_flag("HF_LOCAL_FILES_ONLY", False)

CLIP_EMBEDDING_DIMS = 768
CLIP_IMAGE_RESOLUTION = 224
MAX_PENDING_IMAGE_REQUESTS = 10

_CLIP_IMAGE_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
_CLIP_IMAGE_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)

QUEUE_MAX_SIZE = int(
    os.environ.get("INFERENCE_QUEUE_MAX_SIZE", str(MAX_PENDING_IMAGE_REQUESTS))
)
TASK_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_TASK_TIMEOUT", "10"))
QUEUE_TIMEOUT_SECONDS = int(
    os.environ.get("INFERENCE_QUEUE_TIMEOUT", str(TASK_TIMEOUT_SECONDS))
)
EXEC_TIMEOUT_SECONDS = int(
    os.environ.get("INFERENCE_EXEC_TIMEOUT", str(max(30, TASK_TIMEOUT_SECONDS)))
)

LOG = logging.getLogger("mt_photos_ai.image_clip")
