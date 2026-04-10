import logging
import sys
import types
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import openvino as ov
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if find_spec("transitions") is None:
    fake_transitions = types.ModuleType("transitions")

    class _FakeMachine:
        def __init__(self, model, states, initial, auto_transitions=False) -> None:
            self.model = model
            self.model.state = initial

        def add_transition(self, trigger, sources, dest) -> None:
            def _transition() -> None:
                self.model.state = dest

            setattr(self.model, trigger, _transition)

    fake_transitions.Machine = _FakeMachine
    sys.modules["transitions"] = fake_transitions

if find_spec("rapidocr") is None:
    fake_rapidocr = types.ModuleType("rapidocr")

    class _FakeRapidOCR:
        pass

    class _FakeEngineType:
        OPENVINO = types.SimpleNamespace(value="openvino")

    fake_rapidocr.RapidOCR = _FakeRapidOCR
    fake_rapidocr.EngineType = _FakeEngineType
    fake_rapidocr_utils = types.ModuleType("rapidocr.utils")
    fake_rapidocr_log = types.ModuleType("rapidocr.utils.log")
    fake_rapidocr_log.logger = logging.getLogger("rapidocr")
    sys.modules["rapidocr"] = fake_rapidocr
    sys.modules["rapidocr.utils"] = fake_rapidocr_utils
    sys.modules["rapidocr.utils.log"] = fake_rapidocr_log

try:
    from insightface.app import FaceAnalysis as _InsightFaceAnalysis  # type: ignore[attr-defined]
except Exception:
    fake_insightface = types.ModuleType("insightface")
    fake_insightface_app = types.ModuleType("insightface.app")

    class _InsightFaceAnalysis:
        pass

    fake_insightface_app.FaceAnalysis = _InsightFaceAnalysis
    fake_insightface.app = fake_insightface_app
    sys.modules["insightface"] = fake_insightface
    sys.modules["insightface.app"] = fake_insightface_app

from models.constants import CLIP_IMAGE_RESOLUTION, _CLIP_IMAGE_MEAN, _CLIP_IMAGE_STD
from models.runtime import AIModels
import scripts.convert as convert_module


class _FakeVisionBackbone(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(2, 3))
        return (pixel_values, pooled)


class _FakeTextBackbone(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 768)
        hidden = hidden * attention_mask.to(dtype=torch.float32).unsqueeze(-1)
        return (hidden,)


class _FakeLoadedModel:
    def __init__(self) -> None:
        self.vision_model = _FakeVisionBackbone()
        self.visual_projection = torch.nn.Identity()
        self.text_model = _FakeTextBackbone()
        self.text_projection = torch.nn.Identity()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(vocab_size=21128)
        )


class ClipOpenvinoPipelineTests(unittest.TestCase):
    def test_openvino_ppp_matches_clip_manual_normalization(self) -> None:
        models = AIModels.__new__(AIModels)
        runner = None

        with patch.object(AIModels, "_ensure_openvino_runtime", return_value=ov.Core()):
            runner = AIModels._build_openvino_preprocess_runner(
                models,
                runner_name="clip_vision_test",
                device_name="CPU",
                output_height=CLIP_IMAGE_RESOLUTION,
                output_width=CLIP_IMAGE_RESOLUTION,
                mean_values=_CLIP_IMAGE_MEAN.tolist(),
                std_values=_CLIP_IMAGE_STD.tolist(),
            )

        try:
            image = np.zeros(
                (CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION, 3),
                dtype=np.uint8,
            )
            image[..., 0] = 12
            image[..., 1] = 34
            image[..., 2] = 200

            actual = runner.run(image[np.newaxis, ...])[0]
            rgb = image[:, :, ::-1].astype(np.float32) / 255.0
            expected = np.transpose((rgb - _CLIP_IMAGE_MEAN) / _CLIP_IMAGE_STD, (2, 0, 1))

            self.assertEqual((3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION), actual.shape)
            self.assertLess(float(np.max(np.abs(actual - expected))), 1e-5)
        finally:
            if runner is not None:
                runner.release()

    def test_convert_script_saves_ir_without_fp16_weight_compression(self) -> None:
        fake_model = _FakeLoadedModel()
        fake_ov = Mock()
        fake_ov.convert_model.return_value = object()

        convert_module._convert_vision_branch(
            model=fake_model,
            ov=fake_ov,
            torch=torch,
            nn=torch.nn,
        )
        convert_module._convert_text_branch(
            model=fake_model,
            ov=fake_ov,
            torch=torch,
            nn=torch.nn,
        )

        self.assertEqual(2, fake_ov.save_model.call_count)
        for call in fake_ov.save_model.call_args_list:
            self.assertIn("compress_to_fp16", call.kwargs)
            self.assertFalse(call.kwargs["compress_to_fp16"])


if __name__ == "__main__":
    unittest.main()
