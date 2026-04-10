import gc
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EMBEDDING_DIMS = 768
INPUT_RESOLUTION = 224
CONTEXT_LENGTH = 77
CLIP_IMAGE_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
CLIP_IMAGE_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
LOW_BIT_CONSTANT_TYPES = frozenset({"i4", "u4", "i8", "u8"})


def resolve_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    scripts_dir = Path(__file__).resolve().parent
    candidates = [scripts_dir.parent, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "app").exists() and (candidate / "README.md").exists():
            return candidate
    return scripts_dir.parent


def resolve_model_base_path(project_root: Path | None = None) -> Path:
    resolved_project_root = project_root or resolve_project_root()
    return Path(os.environ.get("MODEL_PATH", str(resolved_project_root / "models"))).resolve()


def normalize_clip_rgb(rgb_image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    normalized = (rgb - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)


def build_synthetic_vision_samples(sample_count: int, *, seed_offset: int = 0) -> tuple[np.ndarray, ...]:
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
        samples.append(normalize_clip_rgb(rgb))
    return tuple(samples)


def build_synthetic_text_samples(
    sample_count: int,
    *,
    token_upper_bound: int = 256,
    seed_offset: int = 0,
) -> tuple[dict[str, np.ndarray], ...]:
    valid_token_upper_bound = max(128, token_upper_bound)
    samples: list[dict[str, np.ndarray]] = []
    for index in range(sample_count):
        sample_index = index + seed_offset
        sequence_length = min(CONTEXT_LENGTH, 8 + (sample_index * 5))
        input_ids = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        attention_mask = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        token_body = (
            ((np.arange(sequence_length, dtype=np.int64) * (sample_index + 11) * 17) + (sample_index * 23))
            % (valid_token_upper_bound - 32)
        ) + 16
        input_ids[0, :sequence_length] = token_body
        input_ids[0, 0] = 101 % valid_token_upper_bound
        input_ids[0, sequence_length - 1] = 102 % valid_token_upper_bound
        attention_mask[0, :sequence_length] = 1
        samples.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return tuple(samples)


def normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def is_representation_collapsed(
    embeddings: np.ndarray,
    *,
    rank_threshold: int = 2,
    similarity_threshold: float = 0.999,
) -> bool:
    normalized = normalized_embeddings(embeddings)
    if normalized.shape[0] < 2:
        return False
    if np.linalg.matrix_rank(normalized, tol=1e-3) < max(2, rank_threshold):
        return True
    similarity_matrix = normalized @ normalized.T
    off_diagonal = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    return bool(np.mean(off_diagonal) >= similarity_threshold)


def mean_embedding_cosine_similarity(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> float:
    reference = normalized_embeddings(reference_embeddings)
    candidate = normalized_embeddings(candidate_embeddings)
    if reference.shape != candidate.shape:
        raise ValueError(f"Embedding shape mismatch: expected {reference.shape}, got {candidate.shape}")
    return float(np.mean(np.sum(reference * candidate, axis=1)))


def structure_delta_l1(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> float:
    reference = normalized_embeddings(reference_embeddings)
    candidate = normalized_embeddings(candidate_embeddings)
    reference_structure = reference @ reference.T
    candidate_structure = candidate @ candidate.T
    return float(np.mean(np.abs(reference_structure - candidate_structure)))


def embedding_fidelity_score(reference_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> float:
    if is_representation_collapsed(candidate_embeddings):
        return 0.0
    direct_similarity = mean_embedding_cosine_similarity(reference_embeddings, candidate_embeddings)
    structure_score = float(np.clip(1.0 - (structure_delta_l1(reference_embeddings, candidate_embeddings) / 2.0), 0.0, 1.0))
    return float(np.clip((0.7 * direct_similarity) + (0.3 * structure_score), 0.0, 1.0))


def _set_infer_request_inputs(ov: Any, compiled_model: Any, infer_request: Any, sample: Any) -> None:
    if isinstance(sample, dict):
        for index, input_port in enumerate(compiled_model.inputs):
            infer_request.set_input_tensor(
                index,
                ov.Tensor(np.ascontiguousarray(sample[input_port.get_any_name()])),
            )
        return

    if isinstance(sample, (tuple, list)):
        for index, value in enumerate(sample):
            infer_request.set_input_tensor(index, ov.Tensor(np.ascontiguousarray(value)))
        return

    infer_request.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(sample)))


def infer_embeddings(ov: Any, model: Any, samples: Sequence[Any], *, device_name: str = "CPU") -> np.ndarray:
    core = ov.Core()
    compiled_model = core.compile_model(model, device_name)
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


def element_type_name(element_type: Any) -> str:
    if hasattr(element_type, "get_type_name"):
        return str(element_type.get_type_name()).lower()
    return str(element_type).lower().replace("<type:", "").replace(">", "").strip()


def summarize_model_precision(ov: Any, model_path: Path) -> dict[str, Any]:
    core = ov.Core()
    model = core.read_model(str(model_path))
    constant_type_counts: dict[str, int] = {}
    for operation in model.get_ops():
        if operation.get_type_name() != "Constant":
            continue
        type_name = element_type_name(operation.output(0).get_element_type())
        constant_type_counts[type_name] = constant_type_counts.get(type_name, 0) + 1

    low_bit_constant_type_counts = {
        key: value for key, value in constant_type_counts.items() if key in LOW_BIT_CONSTANT_TYPES
    }
    summary = {
        "constant_type_counts": constant_type_counts,
        "input_type_names": sorted({element_type_name(input_port.get_element_type()) for input_port in model.inputs}),
        "output_type_names": sorted({element_type_name(output_port.get_element_type()) for output_port in model.outputs}),
        "low_bit_constant_type_counts": low_bit_constant_type_counts,
    }
    del model
    del core
    gc.collect()
    return summary


def model_artifact_size_bytes(model_path: Path) -> int:
    total_size = 0
    for artifact_path in (model_path, model_path.with_suffix(".bin")):
        if artifact_path.exists():
            total_size += artifact_path.stat().st_size
    return total_size


def evaluate_model_pair(
    *,
    ov: Any,
    baseline_model_path: Path,
    candidate_model_path: Path,
    samples: Sequence[Any],
    device_name: str = "CPU",
) -> dict[str, Any]:
    core = ov.Core()
    baseline_model = core.read_model(str(baseline_model_path))
    candidate_model = core.read_model(str(candidate_model_path))
    baseline_embeddings = infer_embeddings(ov=ov, model=baseline_model, samples=samples, device_name=device_name)
    candidate_embeddings = infer_embeddings(ov=ov, model=candidate_model, samples=samples, device_name=device_name)
    direct_similarity = mean_embedding_cosine_similarity(baseline_embeddings, candidate_embeddings)
    structure_delta = structure_delta_l1(baseline_embeddings, candidate_embeddings)
    fidelity_score = embedding_fidelity_score(baseline_embeddings, candidate_embeddings)
    representation_collapsed = is_representation_collapsed(candidate_embeddings)
    baseline_precision_summary = summarize_model_precision(ov=ov, model_path=baseline_model_path)
    candidate_precision_summary = summarize_model_precision(ov=ov, model_path=candidate_model_path)
    baseline_size_bytes = model_artifact_size_bytes(baseline_model_path)
    candidate_size_bytes = model_artifact_size_bytes(candidate_model_path)
    del baseline_model
    del candidate_model
    del core
    gc.collect()
    return {
        "baseline_model_path": str(baseline_model_path),
        "candidate_model_path": str(candidate_model_path),
        "sample_count": len(samples),
        "mean_cosine_similarity": direct_similarity,
        "structure_delta_l1": structure_delta,
        "fidelity_score": fidelity_score,
        "representation_collapsed": representation_collapsed,
        "baseline_precision_summary": baseline_precision_summary,
        "candidate_precision_summary": candidate_precision_summary,
        "baseline_size_bytes": baseline_size_bytes,
        "candidate_size_bytes": candidate_size_bytes,
        "size_ratio": (candidate_size_bytes / baseline_size_bytes) if baseline_size_bytes else 1.0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
