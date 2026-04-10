import logging
import math
import sys
import tempfile
import types
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
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
        self.config = types.SimpleNamespace(text_config=types.SimpleNamespace(vocab_size=21128))


class _SyntheticVisionModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(pixel_values, kernel_size=16, stride=16)
        flattened = pooled.reshape(pixel_values.shape[0], -1)
        channel_stats = pixel_values.mean(dim=(2, 3))
        features = torch.cat((flattened, channel_stats), dim=1)
        repeat_factor = math.ceil(convert_module.EMBEDDING_DIMS / features.shape[1])
        tiled = features.repeat(1, repeat_factor)
        return tiled[:, : convert_module.EMBEDDING_DIMS]


class ClipOpenvinoPipelineTests(unittest.TestCase):
    @staticmethod
    def _build_sample_images() -> dict[str, np.ndarray]:
        height = 480
        width = 640
        y_coords, x_coords = np.indices((height, width))
        blue = np.asarray((x_coords * 255) / max(1, width - 1), dtype=np.uint8)
        green = np.asarray((y_coords * 255) / max(1, height - 1), dtype=np.uint8)
        red = np.asarray(
            ((x_coords + y_coords) * 255) / max(1, width + height - 2),
            dtype=np.uint8,
        )
        base = np.dstack((blue, green, red))
        cv2.rectangle(base, (40, 60), (260, 220), (20, 40, 220), thickness=-1)
        cv2.circle(base, (460, 150), 90, (220, 200, 30), thickness=-1)
        cv2.line(base, (20, 430), (620, 250), (255, 255, 255), thickness=6)
        cv2.putText(
            base,
            "QA-CLIP",
            (120, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            base,
            "SMOKE",
            (115, 355),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        return {
            "base": np.ascontiguousarray(base),
            "black": np.zeros_like(base),
            "white": np.full_like(base, 255),
        }

    @staticmethod
    def _resize_and_center_crop(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = float(CLIP_IMAGE_RESOLUTION) / float(min(height, width))
        resized_width = max(CLIP_IMAGE_RESOLUTION, int(round(width * scale)))
        resized_height = max(CLIP_IMAGE_RESOLUTION, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        top = max(0, (resized_height - CLIP_IMAGE_RESOLUTION) // 2)
        left = max(0, (resized_width - CLIP_IMAGE_RESOLUTION) // 2)
        cropped = resized[
            top : top + CLIP_IMAGE_RESOLUTION,
            left : left + CLIP_IMAGE_RESOLUTION,
        ]
        return np.ascontiguousarray(cropped, dtype=np.uint8)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))

    def _ensure_test_vision_model(self) -> tuple[Path, object | None]:
        repo_model_path = PROJECT_ROOT / "models" / "qa-clip" / "openvino" / "openvino_image.xml"
        if repo_model_path.exists():
            return repo_model_path, None

        temp_dir = tempfile.TemporaryDirectory()
        temp_model_path = Path(temp_dir.name) / "synthetic_openvino_image.xml"
        model = _SyntheticVisionModel().eval()
        ov_model = ov.convert_model(
            model,
            example_input=torch.randn(1, 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION),
        )
        ov.save_model(ov_model, temp_model_path, compress_to_fp16=False)
        return temp_model_path, temp_dir

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

    def test_current_ir_image_pipeline_preserves_embedding_separation(self) -> None:
        model_path, temp_dir = self._ensure_test_vision_model()
        models = AIModels.__new__(AIModels)
        runner = None

        with patch.object(AIModels, "_ensure_openvino_runtime", return_value=ov.Core()):
            runner = AIModels._build_openvino_preprocess_runner(
                models,
                runner_name="clip_vision_runtime_test",
                device_name="CPU",
                output_height=CLIP_IMAGE_RESOLUTION,
                output_width=CLIP_IMAGE_RESOLUTION,
                mean_values=_CLIP_IMAGE_MEAN.tolist(),
                std_values=_CLIP_IMAGE_STD.tolist(),
            )

        try:
            core = ov.Core()
            compiled_model = core.compile_model(str(model_path), "CPU")
            infer_request = compiled_model.create_infer_request()

            embeddings: dict[str, np.ndarray] = {}
            for name, image in self._build_sample_images().items():
                cropped = self._resize_and_center_crop(image)
                preprocessed = runner.run(cropped[np.newaxis, ...])
                infer_request.set_input_tensor(
                    0,
                    ov.Tensor(np.ascontiguousarray(preprocessed), shared_memory=True),
                )
                infer_request.infer()
                embeddings[name] = np.array(
                    infer_request.get_output_tensor(0).data,
                    dtype=np.float32,
                    copy=True,
                )[0]

            self.assertLess(self._cosine_similarity(embeddings["base"], embeddings["black"]), 0.9)
            self.assertLess(self._cosine_similarity(embeddings["base"], embeddings["white"]), 0.9)
        finally:
            if runner is not None:
                runner.release()
            if temp_dir is not None:
                temp_dir.cleanup()

    def test_load_hf_model_prefers_existing_local_snapshot(self) -> None:
        fake_model = Mock()
        fake_model.eval = Mock()
        auto_model_cls = Mock()
        auto_model_cls.from_pretrained.return_value = fake_model

        with tempfile.TemporaryDirectory() as temp_dir_name:
            snapshot_path = Path(temp_dir_name) / "huggingface"
            snapshot_path.mkdir(parents=True, exist_ok=True)
            (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
            (snapshot_path / "model.safetensors").write_bytes(b"stub")
            with patch.object(convert_module, "HF_SAVE_PATH", snapshot_path):
                model, source = convert_module._load_hf_model(auto_model_cls)

        self.assertIs(model, fake_model)
        self.assertEqual("local_snapshot", source)
        auto_model_cls.from_pretrained.assert_called_once()
        self.assertEqual(str(snapshot_path), auto_model_cls.from_pretrained.call_args.args[0])
        self.assertTrue(auto_model_cls.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertFalse(auto_model_cls.from_pretrained.call_args.kwargs["force_download"])
        fake_model.eval.assert_called_once()

    def test_load_hf_model_uses_local_cache_before_remote(self) -> None:
        fake_model = Mock()
        fake_model.eval = Mock()
        auto_model_cls = Mock()
        auto_model_cls.from_pretrained.return_value = fake_model

        with tempfile.TemporaryDirectory() as temp_dir_name:
            snapshot_path = Path(temp_dir_name) / "missing_snapshot"
            cache_path = Path(temp_dir_name) / "cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            with (
                patch.object(convert_module, "HF_SAVE_PATH", snapshot_path),
                patch.object(convert_module, "CACHE_PATH", cache_path),
            ):
                _, source = convert_module._load_hf_model(auto_model_cls)

        self.assertEqual("local_cache", source)
        auto_model_cls.from_pretrained.assert_called_once()
        self.assertEqual(convert_module.MODEL_ID, auto_model_cls.from_pretrained.call_args.args[0])
        self.assertEqual(str(cache_path), auto_model_cls.from_pretrained.call_args.kwargs["cache_dir"])
        self.assertTrue(auto_model_cls.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertFalse(auto_model_cls.from_pretrained.call_args.kwargs["force_download"])

    def test_reset_conversion_artifacts_preserves_hf_cache_and_snapshot_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            root = Path(temp_dir_name)
            ov_path = root / "openvino"
            hf_path = root / "huggingface"
            cache_path = root / "cache"
            ov_cache_path = root / "ov_cache"
            for path in (ov_path, hf_path, cache_path, ov_cache_path):
                path.mkdir(parents=True, exist_ok=True)
                (path / "sentinel.txt").write_text("sentinel", encoding="utf-8")

            with (
                patch.object(convert_module, "OV_SAVE_PATH", ov_path),
                patch.object(convert_module, "HF_SAVE_PATH", hf_path),
                patch.object(convert_module, "CACHE_PATH", cache_path),
                patch.object(convert_module, "OPENVINO_CACHE_PATH", ov_cache_path),
            ):
                convert_module._reset_conversion_artifacts()

            self.assertFalse((ov_path / "sentinel.txt").exists())
            self.assertFalse((ov_cache_path / "sentinel.txt").exists())
            self.assertTrue((hf_path / "sentinel.txt").exists())
            self.assertTrue((cache_path / "sentinel.txt").exists())

    def test_convert_vision_branch_saves_original_precision_ir(self) -> None:
        fake_ov = Mock()
        fake_ov_model = object()
        fake_ov.convert_model.return_value = fake_ov_model

        with tempfile.TemporaryDirectory() as temp_dir_name:
            with patch.object(convert_module, "OV_SAVE_PATH", Path(temp_dir_name)):
                convert_module._convert_vision_branch(
                    model=_FakeLoadedModel(),
                    ov=fake_ov,
                    torch=torch,
                    nn=torch.nn,
                )

        fake_ov.convert_model.assert_called_once()
        fake_ov.save_model.assert_called_once()
        args, kwargs = fake_ov.save_model.call_args
        self.assertIs(args[0], fake_ov_model)
        self.assertEqual(Path(temp_dir_name) / "openvino_image.xml", args[1])
        self.assertFalse(kwargs["compress_to_fp16"])

    def test_convert_text_branch_saves_original_precision_ir(self) -> None:
        fake_ov = Mock()
        fake_ov_model = object()
        fake_ov.convert_model.return_value = fake_ov_model

        with tempfile.TemporaryDirectory() as temp_dir_name:
            with patch.object(convert_module, "OV_SAVE_PATH", Path(temp_dir_name)):
                convert_module._convert_text_branch(
                    model=_FakeLoadedModel(),
                    ov=fake_ov,
                    torch=torch,
                    nn=torch.nn,
                )

        fake_ov.convert_model.assert_called_once()
        fake_ov.save_model.assert_called_once()
        args, kwargs = fake_ov.save_model.call_args
        self.assertIs(args[0], fake_ov_model)
        self.assertEqual(Path(temp_dir_name) / "openvino_text.xml", args[1])
        self.assertFalse(kwargs["compress_to_fp16"])


if __name__ == "__main__":
    unittest.main()
