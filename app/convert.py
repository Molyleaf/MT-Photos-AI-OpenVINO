import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import nncf
    import openvino as ov
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoProcessor
except ImportError as exc:
    logging.error(
        "Missing conversion dependencies. Install with: "
        "pip install openvino nncf torch transformers"
    )
    raise SystemExit(1) from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
EMBEDDING_DIMS = 768
INPUT_RESOLUTION = 224
CONTEXT_LENGTH = 77

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OV_SAVE_PATH = PROJECT_ROOT / "models" / "qa-clip" / "openvino"
CACHE_PATH = PROJECT_ROOT / "cache" / "huggingface"

ENABLE_NNCF_WEIGHT_COMPRESSION = os.environ.get(
    "QA_CLIP_ENABLE_NNCF_WEIGHT_COMPRESSION", "0"
) == "1"
NNCF_WEIGHT_MODE = os.environ.get("QA_CLIP_NNCF_WEIGHT_MODE", "INT8_ASYM")


class VisionModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.vision_model_base = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model_base(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]
        return self.visual_projection(pooled_output)


class TextModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.text_model_base = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = text_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        return self.text_projection(pooled_output)


def _load_hf_model() -> nn.Module:
    logging.info("Loading model from Hugging Face: %s", MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
    model.eval()
    return model


def _build_ignored_scope(branch: str) -> Optional[Any]:
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
        "vision": ["vision_model.embeddings", "vision_model.pre_layrnorm", "vision_model.post_layernorm"],
        "text": ["text_model.embeddings", "text_model.encoder.layer.*.output.LayerNorm"],
    }
    patterns = critical_patterns + branch_specific.get(branch, [])

    for field_name in ("patterns", "names"):
        try:
            return nncf.IgnoredScope(**{field_name: patterns})
        except TypeError:
            continue
    return None


def _resolve_nncf_mode() -> Optional[Any]:
    mode_enum = getattr(nncf, "CompressWeightsMode", None)
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


def _apply_optional_nncf_weight_compression(ov_model: ov.Model, branch: str) -> ov.Model:
    ignored_scope = _build_ignored_scope(branch)
    if not ENABLE_NNCF_WEIGHT_COMPRESSION:
        logging.info(
            "NNCF policy prepared for %s branch (critical layers excluded). "
            "Weight compression is disabled; export keeps FP16 path only.",
            branch,
        )
        return ov_model

    compress_fn = getattr(nncf, "compress_weights", None)
    if compress_fn is None:
        raise RuntimeError("NNCF compress_weights is unavailable in current nncf package.")

    kwargs: Dict[str, Any] = {}
    if ignored_scope is not None:
        kwargs["ignored_scope"] = ignored_scope

    mode_value = _resolve_nncf_mode()
    if mode_value is not None:
        kwargs["mode"] = mode_value

    logging.info("Applying NNCF weight compression to %s branch with args=%s", branch, kwargs)
    return compress_fn(ov_model, **kwargs)


def _convert_vision_branch() -> None:
    logging.info("Converting vision branch...")
    model = _load_hf_model()
    vision_wrapper = VisionModelWrapper(model)
    dummy_input = torch.randn(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)

    vision_path = OV_SAVE_PATH / "openvino_image_fp16.xml"
    ov_model = ov.convert_model(vision_wrapper, example_input=dummy_input)
    ov_model = _apply_optional_nncf_weight_compression(ov_model, branch="vision")
    ov.save_model(ov_model, vision_path, compress_to_fp16=True)
    logging.info("Vision branch saved to %s", vision_path)

    del ov_model
    del dummy_input
    del vision_wrapper
    del model
    gc.collect()


def _convert_text_branch() -> None:
    logging.info("Converting text branch...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_PATH,
        use_fast=True,
    )
    model = _load_hf_model()
    text_wrapper = TextModelWrapper(model)

    dummy_inputs = {
        "input_ids": torch.randint(0, processor.tokenizer.vocab_size, (1, CONTEXT_LENGTH)),
        "attention_mask": torch.ones(1, CONTEXT_LENGTH, dtype=torch.long),
    }

    text_path = OV_SAVE_PATH / "openvino_text_fp16.xml"
    ov_model = ov.convert_model(text_wrapper, example_input=dummy_inputs)
    ov_model = _apply_optional_nncf_weight_compression(ov_model, branch="text")
    ov.save_model(ov_model, text_path, compress_to_fp16=True)
    logging.info("Text branch saved to %s", text_path)

    del ov_model
    del dummy_inputs
    del text_wrapper
    del model
    del processor
    gc.collect()


def _verify_models() -> None:
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

    logging.info("OpenVINO output directory: %s", OV_SAVE_PATH)
    logging.info("Hugging Face cache directory: %s", CACHE_PATH)

    try:
        _convert_vision_branch()
        gc.collect()
        _convert_text_branch()
        gc.collect()
        _verify_models()
    except Exception as exc:
        logging.error("QA-CLIP conversion failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc

    logging.info("QA-CLIP conversion completed successfully.")


if __name__ == "__main__":
    convert_models()
