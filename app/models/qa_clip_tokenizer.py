import importlib.util
import sys
from pathlib import Path
from typing import Any

from .constants import QA_CLIP_CLIP_ROOT

_TOKENIZER_MODULE_NAME = "mt_photos_ai_qa_clip_bert_tokenizer"


def _load_tokenizer_module() -> Any:
    existing = sys.modules.get(_TOKENIZER_MODULE_NAME)
    if existing is not None:
        return existing

    tokenizer_path = Path(QA_CLIP_CLIP_ROOT) / "bert_tokenizer.py"
    spec = importlib.util.spec_from_file_location(_TOKENIZER_MODULE_NAME, tokenizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load QA-CLIP tokenizer module from {tokenizer_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_TOKENIZER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def create_full_tokenizer() -> Any:
    module = _load_tokenizer_module()
    tokenizer_cls = getattr(module, "FullTokenizer", None)
    if tokenizer_cls is None:
        raise RuntimeError("QA-CLIP tokenizer module does not expose FullTokenizer")
    return tokenizer_cls()
