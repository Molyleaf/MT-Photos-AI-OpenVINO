import gc
import inspect
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
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
DEFAULT_WEIGHT_CANDIDATES = "INT4_ASYM:0.75,INT4_SYM:0.75,INT4_ASYM:0.5,INT4_SYM:0.5"
CALIBRATION_SAMPLE_COUNT = max(8, int(os.environ.get("QACLIP_CALIBRATION_SAMPLES", "16")))
VALIDATION_SAMPLE_COUNT = max(8, int(os.environ.get("QACLIP_VALIDATION_SAMPLES", "12")))
WEIGHT_SUBSET_SIZE = max(8, int(os.environ.get("QACLIP_WEIGHT_SUBSET_SIZE", "32")))
QUANTIZATION_SUBSET_SIZE = max(8, int(os.environ.get("QACLIP_QUANTIZATION_SUBSET_SIZE", "32")))
MAX_ACCURACY_DROP = float(os.environ.get("QACLIP_MAX_ACCURACY_DROP", "0.01"))
MIN_FIDELITY_SCORE = float(os.environ.get("QACLIP_MIN_FIDELITY_SCORE", "0.985"))
WEIGHT_GROUP_SIZE = int(os.environ.get("QACLIP_WEIGHT_GROUP_SIZE", "128"))
WEIGHT_COMPRESSION_BACKUP_MODE = os.environ.get("QACLIP_WEIGHT_BACKUP_MODE", "INT8_ASYM").upper()
COLLAPSE_SIMILARITY_THRESHOLD = float(os.environ.get("QACLIP_COLLAPSE_SIMILARITY_THRESHOLD", "0.999"))
COLLAPSE_RANK_THRESHOLD = max(2, int(os.environ.get("QACLIP_COLLAPSE_RANK_THRESHOLD", "2")))


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


@dataclass(frozen=True)
class CompressionCandidate:
    name: str
    weight_mode: str
    weight_ratio: float
    backup_mode: str = WEIGHT_COMPRESSION_BACKUP_MODE
    group_size: int = WEIGHT_GROUP_SIZE
    all_layers: bool = False
    awq: bool = True
    scale_estimation: bool = True
    quantization_preset: str = "MIXED"

    @property
    def estimated_weight_bits(self) -> float:
        primary_bits = 4.0 if "INT4" in self.weight_mode else 8.0
        backup_bits = 8.0 if "INT8" in self.backup_mode else 32.0
        return (self.weight_ratio * primary_bits) + ((1.0 - self.weight_ratio) * backup_bits)


@dataclass
class CandidateEvaluation:
    name: str
    weight_mode: str
    weight_ratio: float
    backup_mode: str
    estimated_weight_bits: float
    fidelity_score: float
    representation_collapsed: bool
    selected: bool = False
    error: str | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: CompressionCandidate,
        *,
        fidelity_score: float,
        representation_collapsed: bool,
        selected: bool = False,
        error: str | None = None,
    ) -> "CandidateEvaluation":
        return cls(
            name=candidate.name,
            weight_mode=candidate.weight_mode,
            weight_ratio=candidate.weight_ratio,
            backup_mode=candidate.backup_mode,
            estimated_weight_bits=candidate.estimated_weight_bits,
            fidelity_score=fidelity_score,
            representation_collapsed=representation_collapsed,
            selected=selected,
            error=error,
        )


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


def _import_conversion_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import openvino as ov
        import nncf
        import torch
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError as exc:
        install_cmd = "pip install openvino nncf torch transformers"
        logging.error("Missing conversion dependencies. Install manually with: %s", install_cmd)
        raise SystemExit(1) from exc

    return ov, nncf, torch, nn, AutoModel


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
    for path in (CACHE_PATH, OPENVINO_CACHE_PATH, HF_SAVE_PATH, OV_SAVE_PATH):
        _remove_path(path)

    OV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    HF_SAVE_PATH.mkdir(parents=True, exist_ok=True)


def _cleanup_hf_cache() -> None:
    if os.name == "nt" and not _env_flag("QACLIP_FORCE_CLEANUP_HF_CACHE", False):
        logging.info(
            "Skipping in-process Hugging Face cache cleanup on Windows. "
            "huggingface_hub may keep .locks handles briefly; the next run will clean it at startup."
        )
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


def _load_hf_model(auto_model_cls: Any) -> Any:
    logging.info("Loading model from Hugging Face: %s", MODEL_ID)
    model = auto_model_cls.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_PATH),
        force_download=True,
    )
    model.eval()
    return model


def _export_hf_snapshot(model: Any) -> None:
    logging.info("Saving original Hugging Face model snapshot to %s", HF_SAVE_PATH)
    model.save_pretrained(HF_SAVE_PATH, safe_serialization=True)


def _fp16_compression_enabled() -> bool:
    return _env_flag("QACLIP_SAVE_FP16", True)


def _parse_weight_candidates() -> tuple[CompressionCandidate, ...]:
    raw_candidates = os.environ.get("QACLIP_WEIGHT_CANDIDATES", DEFAULT_WEIGHT_CANDIDATES)
    candidates: list[CompressionCandidate] = []
    for raw_item in raw_candidates.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid QACLIP_WEIGHT_CANDIDATES item: {item}")
        weight_mode = parts[0].strip().upper()
        weight_ratio = float(parts[1].strip())
        if not 0.0 < weight_ratio < 1.0:
            raise ValueError(f"QACLIP weight ratio must be in (0, 1): {item}")
        candidates.append(
            CompressionCandidate(
                name=f"{weight_mode.lower()}_{str(weight_ratio).replace('.', '_')}",
                weight_mode=weight_mode,
                weight_ratio=weight_ratio,
            )
        )
    if not candidates:
        raise ValueError("QACLIP_WEIGHT_CANDIDATES resolved to an empty candidate set")
    return tuple(sorted(candidates, key=lambda candidate: candidate.estimated_weight_bits))


def _resolve_enum(nncf: Any, enum_group_name: str, value_name: str) -> Any:
    enum_group = getattr(nncf, enum_group_name, None)
    if enum_group is None:
        raise AttributeError(f"nncf missing enum group: {enum_group_name}")
    try:
        return getattr(enum_group, value_name)
    except AttributeError as exc:
        raise AttributeError(f"nncf missing enum value: {enum_group_name}.{value_name}") from exc


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
        samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
    return tuple(samples)


def _build_nncf_dataset(nncf: Any, samples: Sequence[Any]) -> Any:
    dataset_ctor = nncf.Dataset
    identity_transform = lambda sample: sample
    try:
        parameters = inspect.signature(dataset_ctor).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "transform_func" in parameters:
        return dataset_ctor(list(samples), transform_func=identity_transform)
    if "transform_fn" in parameters:
        return dataset_ctor(list(samples), transform_fn=identity_transform)
    return dataset_ctor(list(samples), identity_transform)


def _normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _is_representation_collapsed(embeddings: np.ndarray) -> bool:
    normalized = _normalized_embeddings(embeddings)
    if normalized.shape[0] < 2:
        return False
    matrix_rank = np.linalg.matrix_rank(normalized, tol=1e-3)
    if matrix_rank < COLLAPSE_RANK_THRESHOLD:
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


def _build_validation_context(
    *,
    ov: Any,
    baseline_model_path: Path,
    validation_samples: Sequence[Any],
) -> tuple[np.ndarray, Any]:
    baseline_model = ov.Core().read_model(str(baseline_model_path))
    reference_embeddings = _infer_embeddings(ov=ov, model=baseline_model, samples=validation_samples)
    del baseline_model
    gc.collect()

    def _validation_fn(model: Any, _: Any) -> tuple[float, None]:
        candidate_embeddings = _infer_embeddings(ov=ov, model=model, samples=validation_samples)
        return _embedding_fidelity_score(reference_embeddings, candidate_embeddings), None

    return reference_embeddings, _validation_fn


def _write_compression_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("Compression metadata saved to %s", metadata_path)


def _optimize_and_save_model(
    *,
    branch_name: str,
    baseline_model_path: Path,
    output_model_path: Path,
    metadata_path: Path,
    calibration_samples: Sequence[Any],
    validation_samples: Sequence[Any],
    ov: Any,
    nncf: Any,
) -> None:
    core = ov.Core()
    reference_embeddings, validation_fn = _build_validation_context(
        ov=ov,
        baseline_model_path=baseline_model_path,
        validation_samples=validation_samples,
    )
    calibration_dataset = _build_nncf_dataset(nncf, calibration_samples)
    validation_dataset = _build_nncf_dataset(nncf, validation_samples)
    candidate_evaluations: list[CandidateEvaluation] = []
    selected_model = None
    selected_candidate = None
    selected_fidelity_score = None

    for candidate in _parse_weight_candidates():
        logging.info(
            "Evaluating %s candidate: mode=%s ratio=%.2f backup=%s estimated_weight_bits=%.2f",
            branch_name,
            candidate.weight_mode,
            candidate.weight_ratio,
            candidate.backup_mode,
            candidate.estimated_weight_bits,
        )
        model = core.read_model(str(baseline_model_path))
        compressed_model = None
        quantized_model = None
        try:
            compressed_model = nncf.compress_weights(
                model,
                mode=_resolve_enum(nncf, "CompressWeightsMode", candidate.weight_mode),
                ratio=candidate.weight_ratio,
                group_size=candidate.group_size,
                all_layers=candidate.all_layers,
                dataset=calibration_dataset,
                subset_size=min(WEIGHT_SUBSET_SIZE, len(calibration_samples)),
                awq=candidate.awq,
                scale_estimation=candidate.scale_estimation,
                backup_mode=_resolve_enum(nncf, "BackupMode", candidate.backup_mode),
            )
            quantized_model = nncf.quantize_with_accuracy_control(
                compressed_model,
                calibration_dataset=calibration_dataset,
                validation_dataset=validation_dataset,
                validation_fn=validation_fn,
                max_drop=MAX_ACCURACY_DROP,
                drop_type=_resolve_enum(nncf, "DropType", "ABSOLUTE"),
                preset=_resolve_enum(nncf, "QuantizationPreset", candidate.quantization_preset),
                model_type=_resolve_enum(nncf, "ModelType", "TRANSFORMER"),
                subset_size=min(QUANTIZATION_SUBSET_SIZE, len(calibration_samples)),
                fast_bias_correction=False,
            )
            candidate_embeddings = _infer_embeddings(ov=ov, model=quantized_model, samples=validation_samples)
            fidelity_score = _embedding_fidelity_score(reference_embeddings, candidate_embeddings)
            collapsed = _is_representation_collapsed(candidate_embeddings)
            evaluation = CandidateEvaluation.from_candidate(
                candidate,
                fidelity_score=fidelity_score,
                representation_collapsed=collapsed,
            )
            candidate_evaluations.append(evaluation)
            if not collapsed and fidelity_score >= MIN_FIDELITY_SCORE:
                evaluation.selected = True
                selected_model = quantized_model
                selected_candidate = candidate
                selected_fidelity_score = fidelity_score
                break
        except Exception as exc:
            candidate_evaluations.append(
                CandidateEvaluation.from_candidate(
                    candidate,
                    fidelity_score=0.0,
                    representation_collapsed=True,
                    error=str(exc),
                )
            )
            logging.warning("Candidate %s failed for %s branch: %s", candidate.name, branch_name, exc)
        finally:
            if quantized_model is not None and quantized_model is not selected_model:
                del quantized_model
            if compressed_model is not None:
                del compressed_model
            del model
            gc.collect()

    if selected_model is None or selected_candidate is None or selected_fidelity_score is None:
        raise RuntimeError(
            f"{branch_name} branch could not satisfy fidelity>={MIN_FIDELITY_SCORE:.4f} "
            f"with max_drop<={MAX_ACCURACY_DROP:.4f}; refusing to export a collapsed model"
        )

    compress_to_fp16 = _fp16_compression_enabled()
    ov.save_model(selected_model, output_model_path, compress_to_fp16=compress_to_fp16)
    _write_compression_metadata(
        metadata_path,
        {
            "branch_name": branch_name,
            "model_id": MODEL_ID,
            "baseline_model_path": str(baseline_model_path),
            "output_model_path": str(output_model_path),
            "compress_to_fp16": compress_to_fp16,
            "max_accuracy_drop": MAX_ACCURACY_DROP,
            "min_fidelity_score": MIN_FIDELITY_SCORE,
            "calibration_sample_count": len(calibration_samples),
            "validation_sample_count": len(validation_samples),
            "selected_candidate": asdict(selected_candidate),
            "selected_fidelity_score": selected_fidelity_score,
            "candidate_evaluations": [asdict(item) for item in candidate_evaluations],
        },
    )
    logging.info(
        "%s branch saved to %s using candidate %s with fidelity %.6f",
        branch_name,
        output_model_path,
        selected_candidate.name,
        selected_fidelity_score,
    )

    del selected_model
    del core
    gc.collect()


def _convert_vision_branch(model: Any, ov: Any, nncf: Any, torch: Any, nn: Any) -> None:
    logging.info("Converting vision branch with NNCF mixed precision search...")

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
    vision_path = OV_SAVE_PATH / "openvino_image.xml"
    metadata_path = OV_SAVE_PATH / "openvino_image.compression.json"

    with tempfile.TemporaryDirectory(prefix="qaclip_vision_", dir=str(OV_SAVE_PATH)) as temp_dir:
        temporary_model_path = Path(temp_dir) / "openvino_image_fp32.xml"
        ov_model = ov.convert_model(vision_wrapper, example_input=dummy_input)
        ov.save_model(ov_model, temporary_model_path, compress_to_fp16=False)
        _optimize_and_save_model(
            branch_name="vision",
            baseline_model_path=temporary_model_path,
            output_model_path=vision_path,
            metadata_path=metadata_path,
            calibration_samples=_build_synthetic_vision_samples(CALIBRATION_SAMPLE_COUNT, seed_offset=0),
            validation_samples=_build_synthetic_vision_samples(
                VALIDATION_SAMPLE_COUNT,
                seed_offset=CALIBRATION_SAMPLE_COUNT,
            ),
            ov=ov,
            nncf=nncf,
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


def _convert_text_branch(model: Any, ov: Any, nncf: Any, torch: Any, nn: Any) -> None:
    logging.info("Converting text branch with NNCF mixed precision search...")

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
    text_path = OV_SAVE_PATH / "openvino_text.xml"
    metadata_path = OV_SAVE_PATH / "openvino_text.compression.json"

    with tempfile.TemporaryDirectory(prefix="qaclip_text_", dir=str(OV_SAVE_PATH)) as temp_dir:
        temporary_model_path = Path(temp_dir) / "openvino_text_fp32.xml"
        ov_model = ov.convert_model(text_wrapper, example_input=dummy_inputs)
        ov.save_model(ov_model, temporary_model_path, compress_to_fp16=False)
        _optimize_and_save_model(
            branch_name="text",
            baseline_model_path=temporary_model_path,
            output_model_path=text_path,
            metadata_path=metadata_path,
            calibration_samples=_build_synthetic_text_samples(vocab_size, CALIBRATION_SAMPLE_COUNT, seed_offset=0),
            validation_samples=_build_synthetic_text_samples(
                vocab_size,
                VALIDATION_SAMPLE_COUNT,
                seed_offset=CALIBRATION_SAMPLE_COUNT,
            ),
            ov=ov,
            nncf=nncf,
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
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    _prepare_hf_cache_env()

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("OpenVINO output directory: %s", OV_SAVE_PATH)
    logging.info("Local Hugging Face snapshot directory: %s", HF_SAVE_PATH)
    logging.info("Hugging Face cache directory: %s", CACHE_PATH)
    logging.info("OpenVINO cache directory cleaned: %s", OPENVINO_CACHE_PATH)
    logging.info("FP16 compression for exported IR enabled: %s", _fp16_compression_enabled())

    ov = nncf = torch = nn = auto_model_cls = None
    model = None
    try:
        ov, nncf, torch, nn, auto_model_cls = _import_conversion_dependencies()
        model = _load_hf_model(auto_model_cls)
        _export_hf_snapshot(model)
        _convert_vision_branch(model=model, ov=ov, nncf=nncf, torch=torch, nn=nn)
        gc.collect()
        _convert_text_branch(model=model, ov=ov, nncf=nncf, torch=torch, nn=nn)
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
