import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import cv2
import numpy as np


def _resolve_app_dir() -> Path:
    env_app_dir = os.environ.get("APP_DIR")
    if env_app_dir:
        return Path(env_app_dir).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "app",
        script_dir.parent,
        Path.cwd().resolve() / "app",
        Path.cwd().resolve(),
    ]
    for candidate in candidates:
        if (candidate / "server.py").is_file() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError("Unable to resolve app directory for smoke_insightface.py")


APP_DIR = _resolve_app_dir()
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test strict InsightFace OpenVINO path.")
    parser.add_argument(
        "--device",
        choices=("CPU", "GPU"),
        default="CPU",
        help="OpenVINOExecutionProvider device_type for InsightFace inference.",
    )
    return parser.parse_args()


def _configure_env(device: str) -> None:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("INFERENCE_DEVICE", "CPU")
    os.environ.setdefault("CLIP_INFERENCE_DEVICE", "CPU")
    os.environ.setdefault("RAPIDOCR_DEVICE", "CPU")
    os.environ["INSIGHTFACE_OV_DEVICE"] = str(device).strip().upper()
    os.environ.setdefault("NON_TEXT_IDLE_RELEASE_SECONDS", "0")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    default_model_path = Path("/models") if Path("/models").is_dir() else (PROJECT_ROOT / "models")
    os.environ.setdefault("MODEL_PATH", str(default_model_path))


ARGS = _parse_args()
_configure_env(ARGS.device)

from models.runtime import AIModels  # noqa: E402


def _read_rss_kb() -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return np.ascontiguousarray(image)


def _resolve_sample_images() -> List[Path]:
    import insightface

    data_dir = Path(insightface.__file__).resolve().parent / "data" / "images"
    candidates = [
        data_dir / "t1.jpg",
        data_dir / "Tom_Hanks_54745.png",
    ]
    resolved = [path for path in candidates if path.is_file()]
    if not resolved:
        raise RuntimeError(
            "InsightFace package sample images are missing; expected at "
            f"{data_dir}"
        )
    return resolved


def _build_smoke_samples() -> List[tuple[str, np.ndarray]]:
    sample_paths = _resolve_sample_images()
    base_samples = [(path.name, _load_bgr(path)) for path in sample_paths]
    samples = list(base_samples)
    augment_index = 0
    while len(samples) < 4:
        source_name, source_image = base_samples[augment_index % len(base_samples)]
        source_path = Path(source_name)
        flip_code = 1 if augment_index % 2 == 0 else 0
        suffix = "hflip" if flip_code == 1 else "vflip"
        samples.append(
            (
                f"{source_path.stem}_{suffix}{source_path.suffix}",
                np.ascontiguousarray(cv2.flip(source_image, flip_code)),
            )
        )
        augment_index += 1
    return samples[:4]


async def _collect_local_results_concurrent(models: AIModels, images: Sequence[np.ndarray]) -> List[Any]:
    coroutines = [
        models.get_face_representation_async(image.copy())
        for image in images
    ]
    return list(await asyncio.gather(*coroutines))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise RuntimeError("Encountered zero-norm embedding during smoke test.")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _sort_native_faces(faces: Iterable[Any]) -> List[Any]:
    return sorted(
        list(faces),
        key=lambda face: (
            float(face.bbox[0]),
            float(face.bbox[1]),
            float(face.bbox[2]),
            float(face.bbox[3]),
        ),
    )


def _sort_local_results(results: Iterable[Any]) -> List[Any]:
    return sorted(
        list(results),
        key=lambda item: (
            int(item.facial_area.x),
            int(item.facial_area.y),
            int(item.facial_area.w),
            int(item.facial_area.h),
        ),
    )


def _compare_native_vs_local(image_name: str, local_results: Sequence[Any], native_faces: Sequence[Any]) -> List[dict[str, Any]]:
    local_sorted = _sort_local_results(local_results)
    native_sorted = _sort_native_faces(native_faces)
    if len(local_sorted) != len(native_sorted):
        raise RuntimeError(
            f"{image_name}: local/native face count mismatch "
            f"local={len(local_sorted)} native={len(native_sorted)}"
        )

    comparisons: List[dict[str, Any]] = []
    for index, (local_result, native_face) in enumerate(zip(local_sorted, native_sorted)):
        local_embedding = np.asarray(local_result.embedding, dtype=np.float32)
        native_embedding = np.asarray(native_face.embedding, dtype=np.float32)
        cosine = _cosine_similarity(local_embedding, native_embedding)
        if cosine < 0.999:
            raise RuntimeError(
                f"{image_name}: embedding cosine drift too large at face={index}, cosine={cosine:.6f}"
            )

        native_bbox = np.asarray(native_face.bbox, dtype=np.float32).reshape(-1)
        local_bbox = np.array(
            [
                local_result.facial_area.x,
                local_result.facial_area.y,
                local_result.facial_area.x + local_result.facial_area.w,
                local_result.facial_area.y + local_result.facial_area.h,
            ],
            dtype=np.float32,
        )
        bbox_delta = float(np.max(np.abs(local_bbox - native_bbox)))
        if bbox_delta > 2.5:
            raise RuntimeError(
                f"{image_name}: bbox drift too large at face={index}, max_abs_delta={bbox_delta:.4f}"
            )

        score_delta = abs(float(local_result.face_confidence) - float(native_face.det_score))
        if score_delta > 1e-4:
            raise RuntimeError(
                f"{image_name}: score drift too large at face={index}, abs_delta={score_delta:.6f}"
            )

        comparisons.append(
            {
                "image": image_name,
                "face_index": index,
                "cosine": round(cosine, 8),
                "bbox_max_abs_delta": round(bbox_delta, 6),
                "score_abs_delta": round(score_delta, 8),
            }
        )
    return comparisons


def _collect_aligned_faces(
    models: AIModels,
    det_model: Any,
    rec_model: Any,
    images: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    aligned_faces: List[np.ndarray] = []
    detection_checks: List[dict[str, Any]] = []
    rec_image_size = int(rec_model.input_size[0])
    for index, image in enumerate(images):
        detections, kpss = models._detect_faces_native(
            det_model,
            image.copy(),
            max_num=0,
            metric="default",
        )
        if detections.shape[0] == 0:
            detection_checks.append({"image_index": index, "faces": 0})
            continue
        if kpss is None:
            raise RuntimeError(
                f"Detection returned faces without landmarks during smoke test at image={index}."
            )
        for face_index in range(detections.shape[0]):
            aligned_faces.append(
                models._align_face(
                    image,
                    np.asarray(kpss[face_index]),
                    rec_image_size,
                )
            )
        detection_checks.append(
            {
                "image_index": index,
                "faces": int(detections.shape[0]),
            }
        )
    return aligned_faces, detection_checks


def _compare_represent_microbatch(
    sample_names: Sequence[str],
    sequential_results: Sequence[Any],
    concurrent_results: Sequence[Any],
) -> List[dict[str, Any]]:
    comparisons: List[dict[str, Any]] = []
    for sample_name, sequential_result, concurrent_result in zip(
        sample_names,
        sequential_results,
        concurrent_results,
    ):
        sequential_sorted = _sort_local_results(sequential_result)
        concurrent_sorted = _sort_local_results(concurrent_result)
        if len(sequential_sorted) != len(concurrent_sorted):
            raise RuntimeError(
                f"{sample_name}: represent microbatch face count mismatch "
                f"sequential={len(sequential_sorted)} concurrent={len(concurrent_sorted)}"
            )

        bbox_delta = 0.0
        score_delta = 0.0
        min_cosine = 1.0
        for sequential_face, concurrent_face in zip(sequential_sorted, concurrent_sorted):
            sequential_embedding = np.asarray(sequential_face.embedding, dtype=np.float32)
            concurrent_embedding = np.asarray(concurrent_face.embedding, dtype=np.float32)
            min_cosine = min(
                min_cosine,
                _cosine_similarity(sequential_embedding, concurrent_embedding),
            )
            sequential_bbox = np.array(
                [
                    sequential_face.facial_area.x,
                    sequential_face.facial_area.y,
                    sequential_face.facial_area.x + sequential_face.facial_area.w,
                    sequential_face.facial_area.y + sequential_face.facial_area.h,
                ],
                dtype=np.float32,
            )
            concurrent_bbox = np.array(
                [
                    concurrent_face.facial_area.x,
                    concurrent_face.facial_area.y,
                    concurrent_face.facial_area.x + concurrent_face.facial_area.w,
                    concurrent_face.facial_area.y + concurrent_face.facial_area.h,
                ],
                dtype=np.float32,
            )
            bbox_delta = max(
                bbox_delta,
                float(np.max(np.abs(sequential_bbox - concurrent_bbox))),
            )
            score_delta = max(
                score_delta,
                abs(float(sequential_face.face_confidence) - float(concurrent_face.face_confidence)),
            )

        if bbox_delta > 1e-4:
            raise RuntimeError(
                f"{sample_name}: represent microbatch bbox drift too large, max_abs_delta={bbox_delta:.6f}"
            )
        if score_delta > 1e-4:
            raise RuntimeError(
                f"{sample_name}: represent microbatch score drift too large, max_abs_delta={score_delta:.6f}"
            )
        if min_cosine < 0.999999:
            raise RuntimeError(
                f"{sample_name}: represent microbatch embedding drift too large, min_cosine={min_cosine:.8f}"
            )

        comparisons.append(
            {
                "image": sample_name,
                "faces": int(len(sequential_sorted)),
                "bbox_max_abs_delta": round(bbox_delta, 8),
                "score_max_abs_delta": round(score_delta, 8),
                "embedding_min_cosine": round(min_cosine, 8),
            }
        )
    return comparisons


def _compare_recognition_batch(
    rec_model: Any,
    aligned_faces: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not aligned_faces:
        raise RuntimeError("Recognition smoke test found no aligned faces in the sample set.")

    batch_embeddings = np.asarray(rec_model.get_feat(list(aligned_faces)), dtype=np.float32)
    if batch_embeddings.ndim == 1:
        batch_embeddings = batch_embeddings.reshape(1, -1)

    single_embeddings_list: List[np.ndarray] = []
    for face in aligned_faces:
        single_embedding = np.asarray(rec_model.get_feat(face), dtype=np.float32)
        if single_embedding.ndim == 1:
            single_embedding = single_embedding.reshape(1, -1)
        single_embeddings_list.append(single_embedding)
    single_embeddings = np.vstack(single_embeddings_list)
    if batch_embeddings.shape != single_embeddings.shape:
        raise RuntimeError(
            "Recognition batch/single shape mismatch: "
            f"batch={batch_embeddings.shape} single={single_embeddings.shape}"
        )

    max_abs_delta = float(np.max(np.abs(batch_embeddings - single_embeddings)))
    if max_abs_delta > 1e-4:
        raise RuntimeError(
            f"Recognition batch/single drift too large: max_abs_delta={max_abs_delta:.6f}"
        )

    min_cosine = 1.0
    for batch_embedding, single_embedding in zip(batch_embeddings, single_embeddings):
        cosine = _cosine_similarity(
            np.asarray(batch_embedding, dtype=np.float32),
            np.asarray(single_embedding, dtype=np.float32),
        )
        min_cosine = min(min_cosine, cosine)
    if min_cosine < 0.999999:
        raise RuntimeError(
            f"Recognition batch/single cosine drift too large: min_cosine={min_cosine:.8f}"
        )

    return {
        "faces": int(len(aligned_faces)),
        "embedding_max_abs_delta": round(max_abs_delta, 8),
        "embedding_min_cosine": round(min_cosine, 8),
    }


async def _collect_local_results(models: AIModels, images: Sequence[np.ndarray]) -> List[Any]:
    results: List[Any] = []
    for image in images:
        results.append(await models.get_face_representation_async(image.copy()))
    return results


def _verify_runtime_layout(face_app: Any) -> dict[str, str]:
    runtime_model_dir = Path(face_app.model_dir).resolve()
    runtime_root = runtime_model_dir.parent.parent
    if runtime_model_dir.parent.name != "models" or runtime_model_dir.name != "antelopev2":
        raise RuntimeError(
            "InsightFace runtime root layout is invalid: "
            f"model_dir={runtime_model_dir}"
        )
    if (runtime_root / "antelopev2").exists():
        raise RuntimeError(
            "Legacy compatibility runtime directory still exists under strict path: "
            f"{runtime_root / 'antelopev2'}"
        )
    return {
        "runtime_root": str(runtime_root),
        "runtime_model_dir": str(runtime_model_dir),
    }


def main() -> int:
    smoke_samples = _build_smoke_samples()
    sample_names = [name for name, _ in smoke_samples]
    sample_images = [image for _, image in smoke_samples]
    models = AIModels()
    try:
        models._ensure_face_loaded()
        if models._face_engine is None:
            raise RuntimeError("Face engine failed to load during smoke test.")

        face_app = models._face_engine
        det_model = face_app.det_model
        rec_model = face_app.models.get("recognition")
        if rec_model is None:
            raise RuntimeError("Recognition model missing during smoke test.")

        runtime_layout = _verify_runtime_layout(face_app)
        provider_runtime = models._validate_insightface_openvino_provider(
            face_app,
            expected_device_type=ARGS.device,
        )
        if models._face_preprocess_device != "CPU":
            raise RuntimeError(
                "InsightFace preprocessing must stay on CPU, "
                f"got preprocess_device={models._face_preprocess_device!r}"
            )
        local_results = asyncio.run(_collect_local_results(models, sample_images))
        concurrent_results = asyncio.run(_collect_local_results_concurrent(models, sample_images))
        native_results = [face_app.get(image.copy(), max_num=0) for image in sample_images]

        semantic_checks: List[dict[str, Any]] = []
        semantic_sample_count = 0
        for sample_name, local_result, native_result in zip(sample_names, local_results, native_results):
            if not native_result:
                continue
            semantic_sample_count += 1
            semantic_checks.extend(
                _compare_native_vs_local(sample_name, local_result, native_result)
            )
        if semantic_sample_count == 0:
            raise RuntimeError("No InsightFace sample image produced native faces during semantic smoke test.")

        aligned_faces, detection_checks = _collect_aligned_faces(
            models,
            det_model,
            rec_model,
            sample_images,
        )
        represent_microbatch_checks = _compare_represent_microbatch(
            sample_names,
            local_results,
            concurrent_results,
        )
        recognition_check = _compare_recognition_batch(rec_model, aligned_faces)

        rss_before_release_kb = _read_rss_kb()
        models.release_models_for_restart()
        gc.collect()
        rss_after_release_kb = _read_rss_kb()
        released_families = models.get_loaded_runtime_families()
        if released_families:
            raise RuntimeError(
                "Runtime families were not fully released by release_models_for_restart(): "
                f"{released_families}"
            )

        reload_results = asyncio.run(_collect_local_results(models, [sample_images[0]]))[0]
        rss_after_reload_kb = _read_rss_kb()
        if not reload_results:
            raise RuntimeError("Face reload smoke test returned no faces after release/reload.")
        loaded_families_after_reload = models.get_loaded_runtime_families()

        summary = {
            "device": ARGS.device,
            "provider_runtime": provider_runtime,
            "preprocess_device": models._face_preprocess_device,
            "face_runtime": {
                "lane": 1,
                "preprocess_workers": models._face_preprocess_worker_count,
                "batch_size": models._face_batch_size,
                "admission": models._face_admission.capacity,
            },
            "runtime_layout": runtime_layout,
            "semantic_checks": semantic_checks,
            "detection_checks": detection_checks,
            "represent_microbatch_checks": represent_microbatch_checks,
            "recognition_check": recognition_check,
            "release_check": {
                "rss_before_release_kb": rss_before_release_kb,
                "rss_after_release_kb": rss_after_release_kb,
                "rss_after_reload_kb": rss_after_reload_kb,
                "released_families_after_release": released_families,
                "loaded_families_after_reload": loaded_families_after_reload,
                "reload_face_count": len(reload_results),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        models.release_all_models()


if __name__ == "__main__":
    raise SystemExit(main())
