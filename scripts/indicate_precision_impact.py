import argparse
import logging
import os
from pathlib import Path
from typing import Any

try:
    from scripts.qaclip_precision_utils import (
        build_synthetic_text_samples,
        build_synthetic_vision_samples,
        evaluate_model_pair,
        resolve_model_base_path,
        resolve_project_root,
        write_json,
    )
except ImportError:
    from qaclip_precision_utils import (
        build_synthetic_text_samples,
        build_synthetic_vision_samples,
        evaluate_model_pair,
        resolve_model_base_path,
        resolve_project_root,
        write_json,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = resolve_project_root()
MODEL_BASE_PATH = resolve_model_base_path(PROJECT_ROOT)
BASELINE_DIR = MODEL_BASE_PATH / "qa-clip" / "openvino"
CANDIDATE_DIR = MODEL_BASE_PATH / "qa-clip" / "openvino-fp16"
DEFAULT_REPORT_PATH = CANDIDATE_DIR / "precision_impact.json"
DEFAULT_SAMPLE_COUNT = max(8, int(os.environ.get("QACLIP_PRECISION_IMPACT_SAMPLES", "12")))
DEFAULT_MIN_FIDELITY = float(os.environ.get("QACLIP_FP16_MIN_FIDELITY", "0.995"))
DEFAULT_TEXT_TOKEN_UPPER_BOUND = int(os.environ.get("QACLIP_FP16_TOKEN_UPPER_BOUND", "256"))


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        logging.error("Missing analysis dependency. Install manually with: pip install openvino")
        raise SystemExit(1) from exc
    return ov


def analyze_precision_impact(
    *,
    baseline_dir: Path | None = None,
    candidate_dir: Path | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    min_fidelity: float = DEFAULT_MIN_FIDELITY,
    device_name: str = "CPU",
    token_upper_bound: int = DEFAULT_TEXT_TOKEN_UPPER_BOUND,
    ov: Any | None = None,
) -> dict[str, Any]:
    openvino_module = ov or _import_openvino()
    resolved_baseline_dir = Path(baseline_dir or BASELINE_DIR)
    resolved_candidate_dir = Path(candidate_dir or CANDIDATE_DIR)

    vision_report = evaluate_model_pair(
        ov=openvino_module,
        baseline_model_path=resolved_baseline_dir / "openvino_image.xml",
        candidate_model_path=resolved_candidate_dir / "openvino_image.xml",
        samples=build_synthetic_vision_samples(sample_count),
        device_name=device_name,
    )
    text_report = evaluate_model_pair(
        ov=openvino_module,
        baseline_model_path=resolved_baseline_dir / "openvino_text.xml",
        candidate_model_path=resolved_candidate_dir / "openvino_text.xml",
        samples=build_synthetic_text_samples(sample_count, token_upper_bound=token_upper_bound),
        device_name=device_name,
    )

    report = {
        "project_root": str(PROJECT_ROOT),
        "baseline_dir": str(resolved_baseline_dir),
        "candidate_dir": str(resolved_candidate_dir),
        "device_name": device_name,
        "sample_count": sample_count,
        "min_fidelity": min_fidelity,
        "vision": vision_report,
        "text": text_report,
    }

    failures: list[str] = []
    for branch_name, branch_report in (("vision", vision_report), ("text", text_report)):
        if branch_report["candidate_precision_summary"]["low_bit_constant_type_counts"]:
            failures.append(
                f"{branch_name} branch exported low-bit constants unexpectedly: "
                f"{branch_report['candidate_precision_summary']['low_bit_constant_type_counts']}"
            )
        if branch_report["representation_collapsed"]:
            failures.append(f"{branch_name} branch representation collapsed after FP16 compression")
        if branch_report["fidelity_score"] < min_fidelity:
            failures.append(
                f"{branch_name} branch fidelity too low after FP16 compression: "
                f"{branch_report['fidelity_score']:.6f} < {min_fidelity:.6f}"
            )

    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the precision impact of QA-CLIP FP16 OpenVINO IR.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR, help="Baseline FP32 IR directory.")
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR, help="Candidate FP16 IR directory.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Optional JSON report output path.",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device name used for evaluation.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Synthetic sample count per branch.")
    parser.add_argument(
        "--min-fidelity",
        type=float,
        default=DEFAULT_MIN_FIDELITY,
        help="Minimum acceptable embedding fidelity score.",
    )
    parser.add_argument(
        "--token-upper-bound",
        type=int,
        default=DEFAULT_TEXT_TOKEN_UPPER_BOUND,
        help="Upper bound used when generating synthetic token ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_precision_impact(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        sample_count=args.sample_count,
        min_fidelity=args.min_fidelity,
        device_name=args.device,
        token_upper_bound=args.token_upper_bound,
    )
    write_json(args.report_path, report)
    logging.info("Precision impact report saved to %s", args.report_path)
    for branch_name in ("vision", "text"):
        branch_report = report[branch_name]
        logging.info(
            "%s FP16 impact: fidelity=%.6f cosine=%.6f structure_delta=%.6f size_ratio=%.6f collapsed=%s",
            branch_name,
            branch_report["fidelity_score"],
            branch_report["mean_cosine_similarity"],
            branch_report["structure_delta_l1"],
            branch_report["size_ratio"],
            branch_report["representation_collapsed"],
        )
    if report["status"] != "pass":
        raise SystemExit("\n".join(report["failures"]))


if __name__ == "__main__":
    main()
