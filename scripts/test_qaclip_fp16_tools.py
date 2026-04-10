import tempfile
import unittest
from pathlib import Path

import openvino as ov
import torch

import scripts.convert_fp16 as convert_fp16_module
import scripts.indicate_precision_impact as indicate_precision_impact_module


class _SmallPrecisionVisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = torch.nn.Linear(6 * 4 * 4, 768)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.features(pixel_values)
        flattened = features.reshape(pixel_values.shape[0], -1)
        return self.head(flattened)


class _SmallPrecisionTextModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 32)
        self.projection = torch.nn.Linear(32, 768)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        masked = embeddings * attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        pooled = masked[:, 0, :] + masked.mean(dim=1)
        return self.projection(pooled)


def _export_baseline_ir(baseline_dir: Path) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)

    vision_model = _SmallPrecisionVisionModel().eval()
    vision_ov_model = ov.convert_model(vision_model, example_input=torch.randn(1, 3, 224, 224))
    ov.save_model(vision_ov_model, baseline_dir / "openvino_image.xml", compress_to_fp16=False)

    text_model = _SmallPrecisionTextModel().eval()
    text_ov_model = ov.convert_model(
        text_model,
        example_input={
            "input_ids": torch.randint(0, 128, (1, 77), dtype=torch.long),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        },
    )
    ov.save_model(text_ov_model, baseline_dir / "openvino_text.xml", compress_to_fp16=False)


class QaclipFp16ToolTests(unittest.TestCase):
    def test_fp16_compression_and_precision_impact_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            root = Path(temp_dir_name)
            baseline_dir = root / "openvino"
            candidate_dir = root / "openvino-fp16"
            _export_baseline_ir(baseline_dir)

            compression_report = convert_fp16_module.compress_models(
                source_dir=baseline_dir,
                target_dir=candidate_dir,
                ov=ov,
            )

            self.assertGreater(
                compression_report["branches"]["vision"]["target_precision_summary"]["constant_type_counts"].get("f16", 0),
                0,
            )
            self.assertGreater(
                compression_report["branches"]["text"]["target_precision_summary"]["constant_type_counts"].get("f16", 0),
                0,
            )

            impact_report = indicate_precision_impact_module.analyze_precision_impact(
                baseline_dir=baseline_dir,
                candidate_dir=candidate_dir,
                sample_count=8,
                min_fidelity=0.98,
                device_name="CPU",
                token_upper_bound=128,
                ov=ov,
            )

            self.assertEqual("pass", impact_report["status"])
            self.assertEqual([], impact_report["failures"])
            self.assertFalse(impact_report["vision"]["representation_collapsed"])
            self.assertFalse(impact_report["text"]["representation_collapsed"])
            self.assertGreater(impact_report["vision"]["fidelity_score"], 0.98)
            self.assertGreater(impact_report["text"]["fidelity_score"], 0.98)
            self.assertEqual({}, impact_report["vision"]["candidate_precision_summary"]["low_bit_constant_type_counts"])
            self.assertEqual({}, impact_report["text"]["candidate_precision_summary"]["low_bit_constant_type_counts"])
            self.assertLess(impact_report["vision"]["size_ratio"], 1.0)
            self.assertLess(impact_report["text"]["size_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
