import gc
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
EMBEDDING_DIMS = 768
INPUT_RESOLUTION = 224
CONTEXT_LENGTH = 77


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


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
HF_SAVE_PATH = MODEL_BASE_PATH / "qa-clip" / "huggingface"
CACHE_PATH = Path(os.environ.get("HF_CACHE_DIR", str(PROJECT_ROOT / "cache" / "huggingface")))
OPENVINO_CACHE_PATH = Path(os.environ.get("OV_CACHE_DIR", str(PROJECT_ROOT / "cache" / "openvino")))


def _prepare_hf_cache_env() -> None:
    cache_root = CACHE_PATH.resolve()
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _import_conversion_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import openvino as ov
        import torch
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError as exc:
        install_cmd = "pip install openvino torch transformers"
        logging.error("Missing conversion dependencies. Install manually with: %s", install_cmd)
        raise SystemExit(1) from exc

    return ov, torch, nn, AutoModel


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        logging.info("Removed directory: %s", path)
        return
    path.unlink()
    logging.info("Removed file: %s", path)


def _reset_conversion_artifacts() -> None:
    removable_paths = [OPENVINO_CACHE_PATH, OV_SAVE_PATH]
    if _env_flag("QACLIP_RESET_HF_SNAPSHOT", False):
        removable_paths.append(HF_SAVE_PATH)
    if _env_flag("QACLIP_RESET_HF_CACHE", False):
        removable_paths.append(CACHE_PATH)

    for path in removable_paths:
        _remove_path(path)

    OV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    HF_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _cleanup_hf_cache() -> None:
    if not _env_flag("QACLIP_FORCE_CLEANUP_HF_CACHE", False):
        return

    gc.collect()
    for attempt in range(3):
        try:
            _remove_path(CACHE_PATH)
            return
        except PermissionError as exc:
            if attempt == 2:
                logging.warning(
                    "Best-effort cleanup skipped for %s because a file is still locked: %s",
                    CACHE_PATH,
                    exc,
                )
                return
            time.sleep(0.5 * (attempt + 1))


def _snapshot_is_populated(snapshot_path: Path) -> bool:
    if not snapshot_path.exists():
        return False
    if not (snapshot_path / "config.json").exists():
        return False
    direct_weight_files = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    if any((snapshot_path / file_name).exists() for file_name in direct_weight_files):
        return True
    if any(snapshot_path.glob("model-*.safetensors")):
        return True
    if any(snapshot_path.glob("pytorch_model-*.bin")):
        return True
    return False


def _load_hf_model(auto_model_cls: Any) -> tuple[Any, str]:
    if _snapshot_is_populated(HF_SAVE_PATH) and not _env_flag("QACLIP_FORCE_HF_DOWNLOAD", False):
        logging.info("Loading model from local snapshot: %s", HF_SAVE_PATH)
        model = auto_model_cls.from_pretrained(
            str(HF_SAVE_PATH),
            local_files_only=True,
            force_download=False,
        )
        model.eval()
        return model, "local_snapshot"

    if not _env_flag("QACLIP_FORCE_HF_DOWNLOAD", False):
        try:
            logging.info("Loading model from local Hugging Face cache: %s", CACHE_PATH)
            model = auto_model_cls.from_pretrained(
                MODEL_ID,
                cache_dir=str(CACHE_PATH),
                local_files_only=True,
                force_download=False,
            )
            model.eval()
            return model, "local_cache"
        except OSError:
            logging.info("Local Hugging Face cache miss detected. Falling back to remote download.")

    logging.info("Loading model from Hugging Face Hub: %s", MODEL_ID)
    model = auto_model_cls.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_PATH),
        force_download=False,
        local_files_only=False,
    )
    model.eval()
    return model, "remote"


def _export_hf_snapshot(model: Any, load_source: str) -> None:
    if load_source == "local_snapshot" and _snapshot_is_populated(HF_SAVE_PATH):
        logging.info("Local Hugging Face snapshot already present, skipping save_pretrained.")
        return
    if _snapshot_is_populated(HF_SAVE_PATH) and not _env_flag("QACLIP_OVERWRITE_HF_SNAPSHOT", False):
        logging.info("Hugging Face snapshot already present, keeping existing files at %s", HF_SAVE_PATH)
        return

    logging.info("Saving Hugging Face model snapshot to %s", HF_SAVE_PATH)
    model.save_pretrained(HF_SAVE_PATH, safe_serialization=True)


def _save_original_precision_ir(*, branch_name: str, ov: Any, ov_model: Any, output_model_path: Path) -> None:
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output_model_path, compress_to_fp16=False)
    logging.info(
        "%s branch saved to %s with original precision and structure.",
        branch_name,
        output_model_path,
    )


def _convert_vision_branch(model: Any, ov: Any, torch: Any, nn: Any) -> None:
    logging.info("Converting vision branch with original precision and structure...")

    class VisionModelWrapper(nn.Module):
        def __init__(self, loaded_model: Any):
            super().__init__()
            self.vision_model_base = loaded_model.vision_model
            self.visual_projection = loaded_model.visual_projection

        def forward(self, pixel_values: Any) -> Any:
            vision_outputs = self.vision_model_base(pixel_values=pixel_values)
            pooled_output = vision_outputs[1]
            return self.visual_projection(pooled_output)

    vision_wrapper = VisionModelWrapper(model).eval()
    dummy_input = torch.randn(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    ov_model = ov.convert_model(vision_wrapper, example_input=dummy_input)
    _save_original_precision_ir(
        branch_name="vision",
        ov=ov,
        ov_model=ov_model,
        output_model_path=OV_SAVE_PATH / "openvino_image.xml",
    )
    del ov_model
    del dummy_input
    del vision_wrapper
    gc.collect()


def _resolve_vocab_size(model: Any) -> int:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    token_embedding = getattr(
        getattr(getattr(model, "text_model", None), "embeddings", None),
        "token_embedding",
        None,
    )
    num_embeddings = getattr(token_embedding, "num_embeddings", None)
    if isinstance(num_embeddings, int) and num_embeddings > 0:
        return num_embeddings

    fallback_vocab_size = 21128
    logging.warning("Unable to detect vocab size from model, fallback to %s", fallback_vocab_size)
    return fallback_vocab_size


def _convert_text_branch(model: Any, ov: Any, torch: Any, nn: Any) -> None:
    logging.info("Converting text branch with original precision and structure...")

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
            pooled_output = text_outputs[0][:, 0, :]
            return self.text_projection(pooled_output)

    text_wrapper = TextModelWrapper(model).eval()
    vocab_size = _resolve_vocab_size(model)
    dummy_inputs = {
        "input_ids": torch.randint(0, vocab_size, (1, CONTEXT_LENGTH), dtype=torch.long),
        "attention_mask": torch.ones(1, CONTEXT_LENGTH, dtype=torch.long),
    }
    ov_model = ov.convert_model(text_wrapper, example_input=dummy_inputs)
    _save_original_precision_ir(
        branch_name="text",
        ov=ov,
        ov_model=ov_model,
        output_model_path=OV_SAVE_PATH / "openvino_text.xml",
    )
    del ov_model
    del dummy_inputs
    del text_wrapper
    gc.collect()


def _verify_models(ov: Any) -> None:
    core = ov.Core()
    vision_path = OV_SAVE_PATH / "openvino_image.xml"
    text_path = OV_SAVE_PATH / "openvino_text.xml"

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
    del core
    gc.collect()

    logging.info("Model verification passed: both branches output %s dimensions.", EMBEDDING_DIMS)


def convert_models() -> None:
    _reset_conversion_artifacts()
    _prepare_hf_cache_env()

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("OpenVINO output directory: %s", OV_SAVE_PATH)
    logging.info("Local Hugging Face snapshot directory: %s", HF_SAVE_PATH)
    logging.info("Hugging Face cache directory: %s", CACHE_PATH)
    logging.info("OpenVINO cache directory cleaned: %s", OPENVINO_CACHE_PATH)
    logging.info("Original precision/structure export enabled: True")
    logging.info(
        "Hugging Face cache reset enabled: cache=%s snapshot=%s force_download=%s",
        _env_flag("QACLIP_RESET_HF_CACHE", False),
        _env_flag("QACLIP_RESET_HF_SNAPSHOT", False),
        _env_flag("QACLIP_FORCE_HF_DOWNLOAD", False),
    )

    ov = torch = nn = auto_model_cls = None
    model = None
    load_source = "unknown"
    try:
        ov, torch, nn, auto_model_cls = _import_conversion_dependencies()
        model, load_source = _load_hf_model(auto_model_cls)
        logging.info("Hugging Face model load source: %s", load_source)
        _export_hf_snapshot(model, load_source)
        _convert_vision_branch(model=model, ov=ov, torch=torch, nn=nn)
        gc.collect()
        _convert_text_branch(model=model, ov=ov, torch=torch, nn=nn)
        gc.collect()
        _verify_models(ov=ov)
    except Exception as exc:
        logging.error("QA-CLIP conversion failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc
    finally:
        if model is not None:
            del model
            gc.collect()
        _cleanup_hf_cache()

    logging.info("QA-CLIP conversion completed successfully.")


if __name__ == "__main__":
    convert_models()
