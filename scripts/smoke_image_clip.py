import argparse
import asyncio
import gc
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _resolve_image_clip_app_dir() -> Path:
    env_app_dir = os.environ.get("IMAGE_CLIP_APP_DIR")
    if env_app_dir:
        candidate = Path(env_app_dir).expanduser().resolve()
        if (candidate / "server.py").is_file() and (candidate / "models").is_dir():
            return candidate
        raise RuntimeError(f"IMAGE_CLIP_APP_DIR is invalid: {candidate}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "image-clip" / "app",
        Path.cwd().resolve() / "image-clip" / "app",
    ]
    for candidate in candidates:
        if (candidate / "server.py").is_file() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError("Unable to resolve image-clip/app directory for smoke_image_clip.py")


IMAGE_CLIP_APP_DIR = _resolve_image_clip_app_dir()
PROJECT_ROOT = IMAGE_CLIP_APP_DIR.parent.parent
if str(IMAGE_CLIP_APP_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_CLIP_APP_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test local Windows/CUDA Image-CLIP runtime.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device expression for Image-CLIP, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path. Can be passed multiple times. If omitted, synthetic samples are used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="CLIP_IMAGE_BATCH value used during the smoke test.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10,
        help="INFERENCE_QUEUE_MAX_SIZE value used during the smoke test.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require Hugging Face assets to be available locally.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run Image-CLIP with IMAGE_CLIP_USE_FP16=1.",
    )
    return parser.parse_args()


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["IMAGE_CLIP_DEVICE"] = str(args.device).strip()
    os.environ["CLIP_IMAGE_BATCH"] = str(max(1, int(args.batch_size)))
    os.environ["INFERENCE_QUEUE_MAX_SIZE"] = str(max(1, int(args.queue_size)))
    os.environ.setdefault("CLIP_IMAGE_BATCH_WAIT_MS", "5")
    os.environ.setdefault("INFERENCE_TASK_TIMEOUT", "10")
    os.environ.setdefault("INFERENCE_QUEUE_TIMEOUT", "10")
    os.environ.setdefault("INFERENCE_EXEC_TIMEOUT", "30")
    os.environ.setdefault("LOG_LEVEL", "INFO")

    default_model_path = Path("/models") if Path("/models").is_dir() else (PROJECT_ROOT / "models")
    os.environ.setdefault("MODEL_PATH", str(default_model_path))
    os.environ.setdefault("HF_CACHE_DIR", str(PROJECT_ROOT / "cache" / "huggingface"))
    if args.local_files_only:
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"
    if args.fp16:
        os.environ["IMAGE_CLIP_USE_FP16"] = "1"


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return np.ascontiguousarray(image)


def _build_synthetic_base(height: int = 480, width: int = 640) -> np.ndarray:
    y_coords, x_coords = np.indices((height, width))
    blue = np.asarray((x_coords * 255) / max(1, width - 1), dtype=np.uint8)
    green = np.asarray((y_coords * 255) / max(1, height - 1), dtype=np.uint8)
    red = np.asarray(((x_coords + y_coords) * 255) / max(1, width + height - 2), dtype=np.uint8)
    image = np.dstack((blue, green, red))

    cv2.rectangle(image, (40, 60), (260, 220), (20, 40, 220), thickness=-1)
    cv2.circle(image, (460, 150), 90, (220, 200, 30), thickness=-1)
    cv2.line(image, (20, 430), (620, 250), (255, 255, 255), thickness=6)
    cv2.putText(
        image,
        "QA-CLIP",
        (120, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SMOKE",
        (115, 355),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(image)


def _build_synthetic_samples() -> List[Tuple[str, np.ndarray]]:
    base = _build_synthetic_base()
    flipped = np.ascontiguousarray(cv2.flip(base, 1))
    rotated = np.ascontiguousarray(cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE))
    rotated = cv2.resize(rotated, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    tinted = cv2.addWeighted(base, 0.72, np.full_like(base, (35, 90, 10)), 0.28, 0.0)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=2.4)
    return [
        ("synthetic_base", np.ascontiguousarray(base)),
        ("synthetic_flip", np.ascontiguousarray(flipped)),
        ("synthetic_rotate", np.ascontiguousarray(rotated)),
        ("synthetic_tint_blur", np.ascontiguousarray(cv2.addWeighted(tinted, 0.7, blurred, 0.3, 0.0))),
    ]


def _resolve_samples(image_args: Sequence[str]) -> Tuple[List[str], List[np.ndarray], bool]:
    if image_args:
        sample_paths = [Path(item).expanduser().resolve() for item in image_args]
        sample_names = [path.name for path in sample_paths]
        images = [_load_bgr(path) for path in sample_paths]
        return sample_names, images, False

    synthetic_samples = _build_synthetic_samples()
    sample_names = [name for name, _ in synthetic_samples]
    images = [image for _, image in synthetic_samples]
    return sample_names, images, True


def _as_embedding(array_like: Sequence[float], label: str) -> np.ndarray:
    embedding = np.asarray(array_like, dtype=np.float32)
    if embedding.shape != (768,):
        raise RuntimeError(f"{label}: embedding dims mismatch, expected=(768,), got={embedding.shape}")
    if not np.all(np.isfinite(embedding)):
        raise RuntimeError(f"{label}: embedding contains non-finite values")
    norm = float(np.linalg.norm(embedding))
    if norm <= 0.0:
        raise RuntimeError(f"{label}: embedding norm is zero")
    return embedding


def _embedding_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    max_abs_delta = float(np.max(np.abs(left - right)))
    mean_abs_delta = float(np.mean(np.abs(left - right)))
    return {
        "cosine": cosine,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
    }


def _assert_close(
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_cosine: float,
    max_abs_delta: float,
) -> dict[str, float]:
    metrics = _embedding_metrics(left, right)
    if metrics["cosine"] < min_cosine:
        raise RuntimeError(f"{label}: cosine drift too large, cosine={metrics['cosine']}")
    if metrics["max_abs_delta"] > max_abs_delta:
        raise RuntimeError(
            f"{label}: max_abs_delta drift too large, max_abs_delta={metrics['max_abs_delta']}"
        )
    return {
        "cosine": round(metrics["cosine"], 8),
        "max_abs_delta": round(metrics["max_abs_delta"], 8),
        "mean_abs_delta": round(metrics["mean_abs_delta"], 8),
    }


def _assert_generated_samples_diverge(
    sample_names: Sequence[str],
    embeddings: Sequence[np.ndarray],
) -> List[dict[str, float | str]]:
    comparisons: List[dict[str, float | str]] = []
    distinct_pairs = 0
    for left_index, right_index in combinations(range(len(embeddings)), 2):
        metrics = _embedding_metrics(embeddings[left_index], embeddings[right_index])
        comparisons.append(
            {
                "left": sample_names[left_index],
                "right": sample_names[right_index],
                "cosine": round(metrics["cosine"], 8),
                "max_abs_delta": round(metrics["max_abs_delta"], 8),
                "mean_abs_delta": round(metrics["mean_abs_delta"], 8),
            }
        )
        if metrics["max_abs_delta"] > 1e-6:
            distinct_pairs += 1
    if distinct_pairs == 0:
        raise RuntimeError("Synthetic sample embeddings are unexpectedly identical across all pairs.")
    return comparisons


async def _embed_many_sequential(runtime: "ImageClipRuntime", images: Sequence[np.ndarray]) -> List[np.ndarray]:
    embeddings: List[np.ndarray] = []
    for index, image in enumerate(images):
        result = await runtime.get_image_embedding_async(image.copy())
        embeddings.append(_as_embedding(result, f"sequential[{index}]"))
    return embeddings


async def _embed_many_concurrent(runtime: "ImageClipRuntime", images: Sequence[np.ndarray]) -> List[np.ndarray]:
    coroutines = [runtime.get_image_embedding_async(image.copy()) for image in images]
    results = await asyncio.gather(*coroutines)
    return [
        _as_embedding(result, f"concurrent[{index}]")
        for index, result in enumerate(results)
    ]


def main() -> int:
    args = _parse_args()
    _configure_env(args)

    from models.runtime import ImageClipRuntime  # noqa: WPS433

    sample_names, sample_images, using_synthetic_samples = _resolve_samples(args.image)
    if not sample_images:
        raise RuntimeError("No sample images were resolved for the Image-CLIP smoke test.")

    runtime = ImageClipRuntime()
    reload_runtime = None
    try:
        runtime.load()
        if runtime._device is None or runtime._device.type != "cuda":
            raise RuntimeError(
                "Image-CLIP runtime did not initialize on CUDA. "
                f"runtime_device={runtime.runtime_device_label}"
            )

        sequential_embeddings = asyncio.run(_embed_many_sequential(runtime, sample_images))
        repeated_embeddings = asyncio.run(
            _embed_many_sequential(runtime, [sample_images[0], sample_images[0]])
        )
        identical_batch_images = [sample_images[0] for _ in range(max(2, min(args.batch_size, 4)))]
        identical_batch_embeddings = asyncio.run(
            _embed_many_concurrent(runtime, identical_batch_images)
        )
        concurrent_embeddings = asyncio.run(_embed_many_concurrent(runtime, sample_images))

        repeat_check = _assert_close(
            "sequential repeat",
            repeated_embeddings[0],
            repeated_embeddings[1],
            min_cosine=0.999999,
            max_abs_delta=1e-6,
        )
        identical_batch_checks = [
            {
                "repeat_index": index,
                **_assert_close(
                    f"identical batch item {index}",
                    sequential_embeddings[0],
                    embedding,
                    min_cosine=0.99999,
                    max_abs_delta=5e-4,
                ),
            }
            for index, embedding in enumerate(identical_batch_embeddings)
        ]
        mixed_batch_checks = [
            {
                "sample": sample_name,
                **_assert_close(
                    f"mixed batch {sample_name}",
                    sequential_embedding,
                    concurrent_embedding,
                    min_cosine=0.99999,
                    max_abs_delta=5e-4,
                ),
            }
            for sample_name, sequential_embedding, concurrent_embedding in zip(
                sample_names,
                sequential_embeddings,
                concurrent_embeddings,
            )
        ]

        distinct_sample_checks: List[dict[str, float | str]] = []
        if using_synthetic_samples and len(sequential_embeddings) >= 2:
            distinct_sample_checks = _assert_generated_samples_diverge(
                sample_names,
                sequential_embeddings,
            )

        runtime.release()
        gc.collect()

        reload_runtime = ImageClipRuntime()
        reload_runtime.load()
        if reload_runtime._device is None or reload_runtime._device.type != "cuda":
            raise RuntimeError(
                "Reloaded Image-CLIP runtime did not initialize on CUDA. "
                f"runtime_device={reload_runtime.runtime_device_label}"
            )
        reload_embedding = asyncio.run(
            _embed_many_sequential(reload_runtime, [sample_images[0]])
        )[0]
        reload_check = _assert_close(
            "reload",
            sequential_embeddings[0],
            reload_embedding,
            min_cosine=0.99999,
            max_abs_delta=5e-4,
        )

        summary = {
            "device_requested": args.device,
            "runtime_device": runtime.runtime_device_label,
            "reload_runtime_device": reload_runtime.runtime_device_label,
            "model_source": runtime._model_source,
            "batch_size": max(1, int(args.batch_size)),
            "queue_size": max(1, int(args.queue_size)),
            "sample_names": sample_names,
            "using_synthetic_samples": using_synthetic_samples,
            "repeat_check": repeat_check,
            "identical_batch_checks": identical_batch_checks,
            "mixed_batch_checks": mixed_batch_checks,
            "distinct_sample_checks": distinct_sample_checks,
            "reload_check": reload_check,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        runtime.release()
        if reload_runtime is not None:
            reload_runtime.release()


if __name__ == "__main__":
    raise SystemExit(main())
