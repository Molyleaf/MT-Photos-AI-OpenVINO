import logging
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
TEXT_CLIP_ROOT = APP_DIR.parent
PROJECT_ROOT = TEXT_CLIP_ROOT.parent

QA_CLIP_CLIP_PATH_CANDIDATES = (
    PACKAGE_DIR / "QA-CLIP" / "clip",
    PROJECT_ROOT / "app" / "models" / "QA-CLIP" / "clip",
)

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models"))
).expanduser().resolve()
OV_CACHE_DIR = Path(
    os.environ.get("OV_CACHE_DIR", str(PROJECT_ROOT / "cache" / "openvino"))
).expanduser().resolve()

TEXT_CLIP_DEVICE = "CPU"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

LOG = logging.getLogger("mt_photos_ai.text_clip")
