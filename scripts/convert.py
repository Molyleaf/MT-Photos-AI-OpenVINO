import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
EMBEDDING_DIMS = 768
INPUT_RESOLUTION = 224
CONTEXT_LENGTH = 77


def _resolve_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    convert_dir = Path(__file__).resolve().parent
    candidates = [convert_dir.parent, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "app").exists() and (candidate / "README.md").exists():
            return candidate
    return convert_dir.parent


PROJECT_ROOT = _resolve_project_root()
MODEL_BASE_PATH = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models")))
OV_SAVE_PATH = MODEL_BASE_PATH / "qa-clip" / "openvino"
CACHE_PATH = Path(os.environ.get("HF_CACHE_DIR", str(PROJECT_ROOT / "cache" / "huggingface")))

ENABLE_NNCF_WEIGHT_COMPRESSION = os.environ.get(
    "QA_CLIP_ENABLE_NNCF_WEIGHT_COMPRESSION", "0"
) == "1"
NNCF_WEIGHT_MODE = os.environ.get("QA_CLIP_NNCF_WEIGHT_MODE", "INT8_ASYM")


def _prepare_hf_cache_env() -> None:
    cache_root = CACHE_PATH.resolve()
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _import_conversion_dependencies() -> Tuple[Any, Any, Any, Any, Optional[Any]]:
    try:
        import openvino as ov
        import torch
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError as exc:
        install_cmd = "pip install openvino torch transformers"
        logging.error(
            "Missing conversion dependencies. Install manually with: %s%s",
            install_cmd,
            " nncf" if ENABLE_NNCF_WEIGHT_COMPRESSION else "",
        )
        raise SystemExit(1) from exc

    nncf_module: Optional[Any] = None
    if ENABLE_NNCF_WEIGHT_COMPRESSION:
        try:
            import nncf as nncf_module
        except ImportError as exc:
            logging.error(
                "NNCF weight compression is enabled but nncf is missing. Install manually with: "
                "pip install nncf"
            )
            raise SystemExit(1) from exc

    return ov, torch, nn, AutoModel, nncf_module


def _load_hf_model(auto_model_cls: Any) -> Any:
    logging.info("Loading model from Hugging Face: %s", MODEL_ID)
    model = auto_model_cls.from_pretrained(MODEL_ID, cache_dir=str(CACHE_PATH))
    model.eval()
    return model


def _build_ignored_scope(nncf_module: Any, branch: str) -> Optional[Any]:
    critical_patterns = [
        "visual_projection",
        "text_projection",
        "layer_norm",
        "LayerNorm",
        "ln_",
        "norm",
        "embeddings",
    ]
    branch_specific = {
        "vision": [
            "vision_model.embeddings",
            "vision_model.pre_layrnorm",
            "vision_model.post_layernorm",
        ],
        "text": ["text_model.embeddings", "text_model.encoder.layer.*.output.LayerNorm"],
    }
    patterns = critical_patterns + branch_specific.get(branch, [])

    for field_name in ("patterns", "names"):
        try:
            return nncf_module.IgnoredScope(**{field_name: patterns})
        except TypeError:
            continue
    return None


def _resolve_nncf_mode(nncf_module: Any) -> Optional[Any]:
    mode_enum = getattr(nncf_module, "CompressWeightsMode", None)
    if mode_enum is None:
        return None

    if hasattr(mode_enum, NNCF_WEIGHT_MODE):
        return getattr(mode_enum, NNCF_WEIGHT_MODE)

    fallback_order = ["INT8_ASYM", "INT8_SYM", "NF4", "E2M1"]
    for candidate in fallback_order:
        if hasattr(mode_enum, candidate):
            logging.warning(
                "Requested NNCF mode %s unavailable, fallback to %s",
                NNCF_WEIGHT_MODE,
                candidate,
            )
            return getattr(mode_enum, candidate)

    return None


def _apply_optional_nncf_weight_compression(
    ov_model: Any,
    branch: str,
    nncf_module: Optional[Any],
) -> Any:
    if not ENABLE_NNCF_WEIGHT_COMPRESSION:
        logging.info(
            "NNCF policy prepared for %s branch (critical layers excluded). "
            "Weight compression is disabled; export keeps FP16 path only.",
            branch,
        )
        return ov_model

    if nncf_module is None:
        raise RuntimeError("NNCF compression requested but nncf module is unavailable.")

    ignored_scope = _build_ignored_scope(nncf_module, branch)
    compress_fn = getattr(nncf_module, "compress_weights", None)
    if compress_fn is None:
        raise RuntimeError("NNCF compress_weights is unavailable in current nncf package.")

    kwargs: Dict[str, Any] = {}
    if ignored_scope is not None:
        kwargs["ignored_scope"] = ignored_scope

    mode_value = _resolve_nncf_mode(nncf_module)
    if mode_value is not None:
        kwargs["mode"] = mode_value

    logging.info("Applying NNCF weight compression to %s branch with args=%s", branch, kwargs)
    return compress_fn(ov_model, **kwargs)


def _convert_vision_branch(model: Any, ov: Any, torch: Any, nn: Any, nncf_module: Optional[Any]) -> None:
    logging.info("Converting vision branch...")

    class VisionModelWrapper(nn.Module):
        def __init__(self, loaded_model: Any):
            super().__init__()
            self.vision_model_base = loaded_model.vision_model
            self.visual_projection = loaded_model.visual_projection

        def forward(self, pixel_values: Any) -> Any:
            vision_outputs = self.vision_model_base(pixel_values=pixel_values)
            pooled_output = vision_outputs[1]
            return self.visual_projection(pooled_output)

    vision_wrapper = VisionModelWrapper(model)
    dummy_input = torch.randn(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)

    vision_path = OV_SAVE_PATH / "openvino_image_fp16.xml"
    ov_model = ov.convert_model(vision_wrapper, example_input=dummy_input)
    ov_model = _apply_optional_nncf_weight_compression(ov_model, branch="vision", nncf_module=nncf_module)
    ov.save_model(ov_model, vision_path, compress_to_fp16=True)
    logging.info("Vision branch saved to %s", vision_path)

    del ov_model
    del dummy_input
    del vision_wrapper
    gc.collect()


def _resolve_vocab_size(model: Any) -> int:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    token_embedding = getattr(getattr(getattr(model, "text_model", None), "embeddings", None), "token_embedding", None)
    num_embeddings = getattr(token_embedding, "num_embeddings", None)
    if isinstance(num_embeddings, int) and num_embeddings > 0:
        return num_embeddings

    fallback_vocab_size = 21128
    logging.warning("Unable to detect vocab size from model, fallback to %s", fallback_vocab_size)
    return fallback_vocab_size


def _convert_text_branch(model: Any, ov: Any, torch: Any, nn: Any, nncf_module: Optional[Any]) -> None:
    logging.info("Converting text branch...")

    class TextModelWrapper(nn.Module):
        def __init__(self, loaded_model: Any):
            super().__init__()
            self.text_model_base = loaded_model.text_model
            self.text_projection = loaded_model.text_projection

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            text_outputs = self.text_model_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_state = text_outputs[0]
            pooled_output = last_hidden_state[:, 0, :]
            return self.text_projection(pooled_output)

    text_wrapper = TextModelWrapper(model)
    vocab_size = _resolve_vocab_size(model)
    dummy_inputs = {
        "input_ids": torch.randint(0, vocab_size, (1, CONTEXT_LENGTH), dtype=torch.long),
        "attention_mask": torch.ones(1, CONTEXT_LENGTH, dtype=torch.long),
    }

    text_path = OV_SAVE_PATH / "openvino_text_fp16.xml"
    ov_model = ov.convert_model(text_wrapper, example_input=dummy_inputs)
    ov_model = _apply_optional_nncf_weight_compression(ov_model, branch="text", nncf_module=nncf_module)
    ov.save_model(ov_model, text_path, compress_to_fp16=True)
    logging.info("Text branch saved to %s", text_path)

    del ov_model
    del dummy_inputs
    del text_wrapper
    gc.collect()


def _verify_models(ov: Any) -> None:
    core = ov.Core()
    vision_path = OV_SAVE_PATH / "openvino_image_fp16.xml"
    text_path = OV_SAVE_PATH / "openvino_text_fp16.xml"

    vision_model = core.read_model(vision_path)
    vision_dim = vision_model.output(0).get_partial_shape()[1].get_length()
    if vision_dim != EMBEDDING_DIMS:
        raise RuntimeError(f"Vision output dim mismatch: expected={EMBEDDING_DIMS}, got={vision_dim}")
    if len(vision_model.inputs) != 1:
        raise RuntimeError(f"Vision input count mismatch: expected=1, got={len(vision_model.inputs)}")
    del vision_model
    gc.collect()

    text_model = core.read_model(text_path)
    text_dim = text_model.output(0).get_partial_shape()[1].get_length()
    if text_dim != EMBEDDING_DIMS:
        raise RuntimeError(f"Text output dim mismatch: expected={EMBEDDING_DIMS}, got={text_dim}")
    if len(text_model.inputs) != 2:
        raise RuntimeError(f"Text input count mismatch: expected=2, got={len(text_model.inputs)}")
    del text_model
    gc.collect()

    logging.info("Model verification passed: both branches output %s dimensions.", EMBEDDING_DIMS)


def convert_models() -> None:
    OV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    _prepare_hf_cache_env()

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("OpenVINO output directory: %s", OV_SAVE_PATH)
    logging.info("Hugging Face cache directory: %s", CACHE_PATH)
    logging.info("Conversion does not require a top-level scripts/ directory.")

    ov = torch = nn = auto_model_cls = nncf_module = None
    model = None
    try:
        ov, torch, nn, auto_model_cls, nncf_module = _import_conversion_dependencies()
        model = _load_hf_model(auto_model_cls)
        _convert_vision_branch(model=model, ov=ov, torch=torch, nn=nn, nncf_module=nncf_module)
        gc.collect()
        _convert_text_branch(model=model, ov=ov, torch=torch, nn=nn, nncf_module=nncf_module)
        gc.collect()
        _verify_models(ov=ov)
    except Exception as exc:
        logging.error("QA-CLIP conversion failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc
    finally:
        if model is not None:
            del model
            gc.collect()

    logging.info("QA-CLIP conversion completed successfully.")


if __name__ == "__main__":
    convert_models()
