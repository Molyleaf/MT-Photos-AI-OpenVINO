import gc
import logging
import shutil
from pathlib import Path
from typing import Any

try:
    from scripts.qaclip_precision_utils import (
        EMBEDDING_DIMS,
        model_artifact_size_bytes,
        resolve_model_base_path,
        resolve_project_root,
        summarize_model_precision,
    )
except ImportError:
    from qaclip_precision_utils import (
        EMBEDDING_DIMS,
        model_artifact_size_bytes,
        resolve_model_base_path,
        resolve_project_root,
        summarize_model_precision,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = resolve_project_root()
MODEL_BASE_PATH = resolve_model_base_path(PROJECT_ROOT)
SOURCE_DIR = MODEL_BASE_PATH / "qa-clip" / "openvino"
TARGET_DIR = MODEL_BASE_PATH / "qa-clip" / "openvino-fp16"
MODEL_FILENAMES = ("openvino_image.xml", "openvino_text.xml")


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        logging.error("Missing conversion dependency. Install manually with: pip install openvino")
        raise SystemExit(1) from exc
    return ov


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        logging.info("Removed directory: %s", path)
        return
    path.unlink()
    logging.info("Removed file: %s", path)


def _verify_qaclip_ir_shapes(ov: Any, model_dir: Path) -> None:
    core = ov.Core()
    vision_model = core.read_model(str(model_dir / "openvino_image.xml"))
    text_model = core.read_model(str(model_dir / "openvino_text.xml"))

    vision_dim = vision_model.output(0).get_partial_shape()[1].get_length()
    text_dim = text_model.output(0).get_partial_shape()[1].get_length()
    if vision_dim != EMBEDDING_DIMS:
        raise RuntimeError(f"Vision output dim mismatch: expected={EMBEDDING_DIMS}, got={vision_dim}")
    if text_dim != EMBEDDING_DIMS:
        raise RuntimeError(f"Text output dim mismatch: expected={EMBEDDING_DIMS}, got={text_dim}")

    del vision_model
    del text_model
    del core
    gc.collect()


def _compress_model_to_fp16(*, ov: Any, source_model_path: Path, target_model_path: Path) -> dict[str, Any]:
    model = ov.Core().read_model(str(source_model_path))
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, target_model_path, compress_to_fp16=True)
    del model
    gc.collect()
    return {
        "source_model_path": str(source_model_path),
        "target_model_path": str(target_model_path),
        "source_size_bytes": model_artifact_size_bytes(source_model_path),
        "target_size_bytes": model_artifact_size_bytes(target_model_path),
        "target_precision_summary": summarize_model_precision(ov=ov, model_path=target_model_path),
    }


def compress_models(*, source_dir: Path | None = None, target_dir: Path | None = None, ov: Any | None = None) -> dict[str, Any]:
    openvino_module = ov or _import_openvino()
    resolved_source_dir = Path(source_dir or SOURCE_DIR)
    resolved_target_dir = Path(target_dir or TARGET_DIR)

    for model_filename in MODEL_FILENAMES:
        model_path = resolved_source_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing source IR file: {model_path}. Run `py -3.12 scripts/convert.py` first."
            )

    _remove_path(resolved_target_dir)
    resolved_target_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "source_dir": str(resolved_source_dir),
        "target_dir": str(resolved_target_dir),
        "branches": {},
    }
    for model_filename in MODEL_FILENAMES:
        branch_name = "vision" if "image" in model_filename else "text"
        source_model_path = resolved_source_dir / model_filename
        target_model_path = resolved_target_dir / model_filename
        logging.info("Compressing %s branch to FP16 with OpenVINO native save_model...", branch_name)
        branch_report = _compress_model_to_fp16(
            ov=openvino_module,
            source_model_path=source_model_path,
            target_model_path=target_model_path,
        )
        logging.info(
            "%s branch compressed: source_bytes=%s target_bytes=%s constant_types=%s",
            branch_name,
            branch_report["source_size_bytes"],
            branch_report["target_size_bytes"],
            branch_report["target_precision_summary"]["constant_type_counts"],
        )
        report["branches"][branch_name] = branch_report

    _verify_qaclip_ir_shapes(openvino_module, resolved_target_dir)
    logging.info("FP16 QA-CLIP IR saved to %s", resolved_target_dir)
    return report


if __name__ == "__main__":
    try:
        compress_models()
    except Exception as exc:
        logging.error("QA-CLIP FP16 compression failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc
