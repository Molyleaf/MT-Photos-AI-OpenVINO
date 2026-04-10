import gc
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
EMBEDDING_DIMS = 768
INPUT_RESOLUTION = 224
CONTEXT_LENGTH = 77
CLIP_IMAGE_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
CLIP_IMAGE_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
VALIDATION_SAMPLE_COUNT = max(8, int(os.environ.get("QACLIP_VALIDATION_SAMPLES", "12")))
MIN_FIDELITY_SCORE = float(os.environ.get("QACLIP_MIN_FIDELITY_SCORE", "0.985"))
COLLAPSE_SIMILARITY_THRESHOLD = float(os.environ.get("QACLIP_COLLAPSE_SIMILARITY_THRESHOLD", "0.999"))
COLLAPSE_RANK_THRESHOLD = max(2, int(os.environ.get("QACLIP_COLLAPSE_RANK_THRESHOLD", "2")))
LOW_BIT_CONSTANT_TYPES = frozenset({"i4", "u4", "i8", "u8"})


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


def _import_conversion_dependencies() -> Tuple[Any, Any, Any, Any]:
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


def _fp16_compression_enabled() -> bool:
    return _env_flag("QACLIP_SAVE_FP16", True)


def _resolve_torch_export_device(torch: Any) -> tuple[Any, str]:
    requested_device = os.environ.get("QACLIP_TORCH_DEVICE", "AUTO").strip().upper()
    if requested_device not in {"AUTO", "CPU", "CUDA"}:
        raise ValueError(f"Unsupported QACLIP_TORCH_DEVICE: {requested_device}")

    if requested_device == "CPU":
        return torch.device("cpu"), requested_device

    if torch.cuda.is_available():
        return torch.device("cuda"), requested_device

    if requested_device == "CUDA":
        logging.warning("QACLIP_TORCH_DEVICE=CUDA requested but CUDA is unavailable. Falling back to CPU.")

    return torch.device("cpu"), requested_device


def _move_torch_inputs_to_device(example_input: Any, device: Any) -> Any:
    if isinstance(example_input, dict):
        return {key: value.to(device) for key, value in example_input.items()}
    if isinstance(example_input, (tuple, list)):
        moved = [value.to(device) for value in example_input]
        return tuple(moved) if isinstance(example_input, tuple) else moved
    return example_input.to(device)


def _run_torch_forward(module: Any, example_input: Any) -> Any:
    if isinstance(example_input, dict):
        return module(**example_input)
    if isinstance(example_input, (tuple, list)):
        return module(*example_input)
    return module(example_input)


def _cleanup_torch_cuda(torch: Any) -> None:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_module_for_openvino_export(
    *,
    branch_name: str,
    module: Any,
    example_input: Any,
    torch: Any,
) -> tuple[Any, Any, dict[str, str]]:
    export_device, requested_device = _resolve_torch_export_device(torch)
    warmup_device = "CPU"

    if export_device.type == "cuda":
        logging.info(
            "Attempting CUDA warmup for %s branch before OpenVINO conversion. requested_device=%s",
            branch_name,
            requested_device,
        )
        try:
            module = module.to(export_device)
            example_input = _move_torch_inputs_to_device(example_input, export_device)
            with torch.inference_mode():
                _run_torch_forward(module, example_input)
            warmup_device = "CUDA"
        except Exception as exc:
            logging.warning(
                "CUDA warmup for %s branch failed, falling back to CPU-only export: %s",
                branch_name,
                exc,
            )
        finally:
            module = module.to("cpu")
            example_input = _move_torch_inputs_to_device(example_input, torch.device("cpu"))
            _cleanup_torch_cuda(torch)

    return module, example_input, {
        "requested_torch_device": requested_device,
        "warmup_torch_device": warmup_device,
        "openvino_conversion_device": "CPU",
    }


def _normalize_clip_rgb(rgb_image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    normalized = (rgb - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)


def _build_synthetic_vision_samples(sample_count: int, *, seed_offset: int = 0) -> tuple[np.ndarray, ...]:
    y_coords, x_coords = np.indices((INPUT_RESOLUTION, INPUT_RESOLUTION))
    samples: list[np.ndarray] = []
    for index in range(sample_count):
        sample_index = index + seed_offset
        red = ((x_coords * (sample_index + 3) * 11) + (y_coords * 7)) % 256
        green = ((y_coords * (sample_index + 5) * 13) + (x_coords * 5)) % 256
        blue = (((x_coords + y_coords) * (sample_index + 7) * 3) + (sample_index * 29)) % 256
        rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
        if sample_index % 2 == 0:
            mask = ((x_coords - y_coords + (sample_index * 17)) % 29) < 14
            rgb[mask] = 255 - rgb[mask]
        if sample_index % 3 == 0:
            rgb[: INPUT_RESOLUTION // 2, : INPUT_RESOLUTION // 3, :] = (sample_index * 19) % 256
        samples.append(_normalize_clip_rgb(rgb))
    return tuple(samples)


def _build_synthetic_text_samples(
    vocab_size: int,
    sample_count: int,
    *,
    seed_offset: int = 0,
) -> tuple[dict[str, np.ndarray], ...]:
    valid_vocab_size = max(vocab_size, 256)
    samples: list[dict[str, np.ndarray]] = []
    for index in range(sample_count):
        sample_index = index + seed_offset
        sequence_length = min(CONTEXT_LENGTH, 8 + (sample_index * 5))
        input_ids = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        attention_mask = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        token_body = (
            ((np.arange(sequence_length, dtype=np.int64) * (sample_index + 11) * 17) + (sample_index * 23))
            % (valid_vocab_size - 32)
        ) + 16
        input_ids[0, :sequence_length] = token_body
        input_ids[0, 0] = 101 % valid_vocab_size
        input_ids[0, sequence_length - 1] = 102 % valid_vocab_size
        attention_mask[0, :sequence_length] = 1
        samples.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return tuple(samples)


def _normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _is_representation_collapsed(embeddings: np.ndarray) -> bool:
    normalized = _normalized_embeddings(embeddings)
    if normalized.shape[0] < 2:
        return False
    if np.linalg.matrix_rank(normalized, tol=1e-3) < COLLAPSE_RANK_THRESHOLD:
        return True
    similarity_matrix = normalized @ normalized.T
    off_diagonal = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    return bool(np.mean(off_diagonal) >= COLLAPSE_SIMILARITY_THRESHOLD)


def _embedding_fidelity_score(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> float:
    reference = _normalized_embeddings(reference_embeddings)
    candidate = _normalized_embeddings(candidate_embeddings)
    if reference.shape != candidate.shape:
        raise ValueError(f"Embedding shape mismatch: expected {reference.shape}, got {candidate.shape}")
    if _is_representation_collapsed(candidate):
        return 0.0

    direct_similarity = float(np.mean(np.sum(reference * candidate, axis=1)))
    reference_structure = reference @ reference.T
    candidate_structure = candidate @ candidate.T
    structure_score = float(
        np.clip(
            1.0 - (np.mean(np.abs(reference_structure - candidate_structure)) / 2.0),
            0.0,
            1.0,
        )
    )
    return float(np.clip((0.7 * direct_similarity) + (0.3 * structure_score), 0.0, 1.0))


def _set_infer_request_inputs(ov: Any, compiled_model: Any, infer_request: Any, sample: Any) -> None:
    if isinstance(sample, dict):
        for input_port in compiled_model.inputs:
            infer_request.set_input_tensor(
                input_port,
                ov.Tensor(np.ascontiguousarray(sample[input_port.get_any_name()])),
            )
        return

    if isinstance(sample, (tuple, list)):
        for index, value in enumerate(sample):
            infer_request.set_input_tensor(index, ov.Tensor(np.ascontiguousarray(value)))
        return

    infer_request.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(sample)))


def _infer_embeddings(ov: Any, model: Any, samples: Sequence[Any]) -> np.ndarray:
    core = ov.Core()
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    embeddings: list[np.ndarray] = []
    for sample in samples:
        _set_infer_request_inputs(ov=ov, compiled_model=compiled_model, infer_request=infer_request, sample=sample)
        infer_request.infer()
        output_tensor = infer_request.get_output_tensor(0)
        embeddings.append(np.array(output_tensor.data, dtype=np.float32, copy=True).reshape(-1))
    del infer_request
    del compiled_model
    del core
    gc.collect()
    return np.stack(embeddings, axis=0)


def _element_type_name(element_type: Any) -> str:
    if hasattr(element_type, "get_type_name"):
        return str(element_type.get_type_name()).lower()
    return str(element_type).lower().replace("<type:", "").replace(">", "").strip()


def _summarize_model_precision(ov: Any, model_path: Path) -> dict[str, Any]:
    model = ov.Core().read_model(str(model_path))
    constant_type_counts: dict[str, int] = {}
    for operation in model.get_ops():
        if operation.get_type_name() != "Constant":
            continue
        element_type_name = _element_type_name(operation.output(0).get_element_type())
        constant_type_counts[element_type_name] = constant_type_counts.get(element_type_name, 0) + 1

    low_bit_constant_type_counts = {
        key: value for key, value in constant_type_counts.items() if key in LOW_BIT_CONSTANT_TYPES
    }
    summary = {
        "constant_type_counts": constant_type_counts,
        "input_type_names": sorted({_element_type_name(input_port.get_element_type()) for input_port in model.inputs}),
        "output_type_names": sorted({_element_type_name(output_port.get_element_type()) for output_port in model.outputs}),
        "low_bit_constant_type_counts": low_bit_constant_type_counts,
    }
    del model
    gc.collect()
    return summary


def _validate_exported_branch_model(
    *,
    branch_name: str,
    baseline_model_path: Path,
    output_model_path: Path,
    validation_samples: Sequence[Any],
    ov: Any,
) -> dict[str, Any]:
    baseline_model = ov.Core().read_model(str(baseline_model_path))
    exported_model = ov.Core().read_model(str(output_model_path))
    baseline_embeddings = _infer_embeddings(ov=ov, model=baseline_model, samples=validation_samples)
    exported_embeddings = _infer_embeddings(ov=ov, model=exported_model, samples=validation_samples)
    fidelity_score = _embedding_fidelity_score(baseline_embeddings, exported_embeddings)
    representation_collapsed = _is_representation_collapsed(exported_embeddings)
    precision_summary = _summarize_model_precision(ov=ov, model_path=output_model_path)
    del baseline_model
    del exported_model
    gc.collect()

    if precision_summary["low_bit_constant_type_counts"]:
        raise RuntimeError(
            f"{branch_name} branch exported low-bit constants unexpectedly: "
            f"{precision_summary['low_bit_constant_type_counts']}"
        )
    if representation_collapsed:
        raise RuntimeError(f"{branch_name} branch representation collapsed after export")
    if fidelity_score < MIN_FIDELITY_SCORE:
        raise RuntimeError(
            f"{branch_name} branch fidelity too low after export: "
            f"{fidelity_score:.6f} < {MIN_FIDELITY_SCORE:.6f}"
        )

    return {
        "fidelity_score": fidelity_score,
        "representation_collapsed": representation_collapsed,
        "precision_summary": precision_summary,
    }


def _write_precision_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Precision metadata saved to %s", metadata_path)


def _export_branch_models(
    *,
    branch_name: str,
    ov_model: Any,
    final_model_path: Path,
    metadata_path: Path,
    validation_samples: Sequence[Any],
    ov: Any,
    export_context: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"qaclip_{branch_name}_", dir=str(OV_SAVE_PATH)) as temp_dir:
        baseline_model_path = Path(temp_dir) / f"{final_model_path.stem}.baseline.xml"
        ov.save_model(ov_model, baseline_model_path, compress_to_fp16=False)

        compress_to_fp16 = _fp16_compression_enabled()
        ov.save_model(ov_model, final_model_path, compress_to_fp16=compress_to_fp16)

        validation = _validate_exported_branch_model(
            branch_name=branch_name,
            baseline_model_path=baseline_model_path,
            output_model_path=final_model_path,
            validation_samples=validation_samples,
            ov=ov,
        )
        baseline_precision_summary = _summarize_model_precision(ov=ov, model_path=baseline_model_path)
        metadata = {
            "branch_name": branch_name,
            "model_id": MODEL_ID,
            "conversion_policy": "fp16_weights_fp32_compute" if compress_to_fp16 else "fp32_full_precision",
            "compress_to_fp16": compress_to_fp16,
            "baseline_model_path": str(baseline_model_path),
            "output_model_path": str(final_model_path),
            "validation_sample_count": len(validation_samples),
            "fidelity_score": validation["fidelity_score"],
            "representation_collapsed": validation["representation_collapsed"],
            "baseline_precision_summary": baseline_precision_summary,
            "exported_precision_summary": validation["precision_summary"],
            "torch_export": export_context,
        }
        _write_precision_metadata(metadata_path, metadata)
        logging.info(
            "%s branch saved to %s. conversion_policy=%s fidelity=%.6f low_bit_constants=%s",
            branch_name,
            final_model_path,
            metadata["conversion_policy"],
            validation["fidelity_score"],
            validation["precision_summary"]["low_bit_constant_type_counts"],
        )


def _convert_vision_branch(model: Any, ov: Any, torch: Any, nn: Any) -> None:
    logging.info("Converting vision branch with FP32 baseline and FP16 weight compression export...")

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
    vision_wrapper, dummy_input, export_context = _prepare_module_for_openvino_export(
        branch_name="vision",
        module=vision_wrapper,
        example_input=dummy_input,
        torch=torch,
    )
    ov_model = ov.convert_model(vision_wrapper, example_input=dummy_input)
    _export_branch_models(
        branch_name="vision",
        ov_model=ov_model,
        final_model_path=OV_SAVE_PATH / "openvino_image.xml",
        metadata_path=OV_SAVE_PATH / "openvino_image.precision.json",
        validation_samples=_build_synthetic_vision_samples(VALIDATION_SAMPLE_COUNT),
        ov=ov,
        export_context=export_context,
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
    logging.info("Converting text branch with FP32 baseline and FP16 weight compression export...")

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
    text_wrapper, dummy_inputs, export_context = _prepare_module_for_openvino_export(
        branch_name="text",
        module=text_wrapper,
        example_input=dummy_inputs,
        torch=torch,
    )
    ov_model = ov.convert_model(text_wrapper, example_input=dummy_inputs)
    _export_branch_models(
        branch_name="text",
        ov_model=ov_model,
        final_model_path=OV_SAVE_PATH / "openvino_text.xml",
        metadata_path=OV_SAVE_PATH / "openvino_text.precision.json",
        validation_samples=_build_synthetic_text_samples(vocab_size, VALIDATION_SAMPLE_COUNT),
        ov=ov,
        export_context=export_context,
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
    logging.info("FP16 compression for exported IR enabled: %s", _fp16_compression_enabled())
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
        if torch is not None:
            _cleanup_torch_cuda(torch)
        _cleanup_hf_cache()

    logging.info("QA-CLIP conversion completed successfully.")


if __name__ == "__main__":
    convert_models()
