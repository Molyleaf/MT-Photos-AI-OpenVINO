import tempfile
import unittest
from pathlib import Path

import openvino as ov
import torch

import scripts.indicate_precision_impact as indicate_precision_impact_module
import scripts.qaclip_precision_utils as precision_utils_module


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


def _export_precision_pair(*, baseline_dir: Path, candidate_dir: Path) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    vision_model = _SmallPrecisionVisionModel().eval()
    vision_ov_model = ov.convert_model(vision_model, example_input=torch.randn(1, 3, 224, 224))
    ov.save_model(vision_ov_model, baseline_dir / "openvino_image_fp32.xml", compress_to_fp16=False)
    ov.save_model(vision_ov_model, candidate_dir / "openvino_image_fp16.xml", compress_to_fp16=True)

    text_model = _SmallPrecisionTextModel().eval()
    text_ov_model = ov.convert_model(
        text_model,
        example_input={
            "input_ids": torch.randint(0, 128, (1, 77), dtype=torch.long),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        },
    )
    ov.save_model(text_ov_model, baseline_dir / "openvino_text_fp32.xml", compress_to_fp16=False)
    ov.save_model(text_ov_model, candidate_dir / "openvino_text_fp16.xml", compress_to_fp16=True)


class QaclipPrecisionToolTests(unittest.TestCase):
    def test_resolve_sample_value_for_input_falls_back_to_input_order_when_names_are_internal(self) -> None:
        input_ids = torch.randint(0, 128, (1, 77), dtype=torch.long).numpy()
        attention_mask = torch.ones((1, 77), dtype=torch.long).numpy()
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        class _FakeInputPort:
            def __init__(self, names: tuple[str, ...], any_name: str) -> None:
                self._names = names
                self._any_name = any_name

            def get_names(self) -> tuple[str, ...]:
                return self._names

            def get_any_name(self) -> str:
                return self._any_name

        first_port = _FakeInputPort(("49",), "49")
        second_port = _FakeInputPort(("50",), "50")

        resolved_input_ids = precision_utils_module._resolve_sample_value_for_input(0, first_port, sample)
        resolved_attention_mask = precision_utils_module._resolve_sample_value_for_input(1, second_port, sample)

        self.assertTrue((resolved_input_ids == input_ids).all())
        self.assertTrue((resolved_attention_mask == attention_mask).all())

    def test_precision_impact_report_uses_fp32_control_and_fp16_mainline_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            root = Path(temp_dir_name)
            baseline_dir = root / "openvino_fp32"
            candidate_dir = root / "openvino"
            _export_precision_pair(baseline_dir=baseline_dir, candidate_dir=candidate_dir)

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
            self.assertTrue(
                impact_report["vision"]["candidate_model_path"].endswith("openvino_image_fp16.xml")
            )
            self.assertTrue(
                impact_report["text"]["baseline_model_path"].endswith("openvino_text_fp32.xml")
            )


if __name__ == "__main__":
    unittest.main()
