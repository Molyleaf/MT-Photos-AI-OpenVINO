import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import server
from models.schemas import OCRBox, OCRResult


class _FakeManager:
    def __init__(self) -> None:
        self.mark_request_activity_calls = 0
        self.release_all_models_calls = 0
        self.release_models_for_restart_calls = 0
        self.images: list[np.ndarray] = []

    def mark_request_activity(self) -> None:
        self.mark_request_activity_calls += 1

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        self.images.append(image)
        return OCRResult(
            texts=["dummy-text"],
            scores=["0.99"],
            boxes=[OCRBox(x="1.0", y="2.0", width="3.0", height="4.0")],
        )

    def release_all_models(self) -> None:
        self.release_all_models_calls += 1

    def release_models_for_restart(self) -> None:
        self.release_models_for_restart_calls += 1


def _build_png_bytes() -> bytes:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode PNG test image.")
    return encoded.tobytes()


class ServerOCRRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_models_instance = server.models_instance
        self._env = patch.dict(os.environ, {"API_AUTH_KEY": "mt_photos_ai_extra"}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        server.models_instance = self._previous_models_instance
        self._env.stop()

    def test_ocr_route_success_uses_lifespan_manager_and_returns_contract(self) -> None:
        fake_manager = _FakeManager()
        with (
            patch.object(server, "NonTextProcessManager", return_value=fake_manager),
            patch.object(server, "startup_self_check_dri"),
            TestClient(server.app) as client,
        ):
            response = client.post(
                "/ocr",
                headers={"api-key": "mt_photos_ai_extra"},
                files={"file": ("sample.png", BytesIO(_build_png_bytes()), "image/png")},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "result": {
                    "texts": ["dummy-text"],
                    "scores": ["0.99"],
                    "boxes": [{"x": "1.0", "y": "2.0", "width": "3.0", "height": "4.0"}],
                }
            },
            response.json(),
        )
        self.assertEqual(1, fake_manager.mark_request_activity_calls)
        self.assertEqual(1, len(fake_manager.images))
        self.assertEqual((8, 10, 3), tuple(fake_manager.images[0].shape))
        self.assertEqual(1, fake_manager.release_all_models_calls)

    def test_ocr_route_decode_failure_keeps_response_contract(self) -> None:
        fake_manager = _FakeManager()
        with (
            patch.object(server, "NonTextProcessManager", return_value=fake_manager),
            patch.object(server, "startup_self_check_dri"),
            TestClient(server.app) as client,
        ):
            response = client.post(
                "/ocr",
                headers={"api-key": "mt_photos_ai_extra"},
                files={"file": ("broken.png", BytesIO(b"not-an-image"), "image/png")},
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual([], payload["result"])
        self.assertIn("无法被解码为图像", payload["msg"])
        self.assertEqual(1, fake_manager.mark_request_activity_calls)
        self.assertEqual([], fake_manager.images)
        self.assertEqual(1, fake_manager.release_all_models_calls)


if __name__ == "__main__":
    unittest.main()
