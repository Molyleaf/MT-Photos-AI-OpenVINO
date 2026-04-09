import logging
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

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

from models.common import _AdmissionController
from models.rapidocr_lib import RapidOCRMixin
from models.schemas import OCRResult


class _DummyRapidOCRHost(RapidOCRMixin):
    def __init__(self) -> None:
        self._rapidocr_load_lock = threading.Lock()
        self._rapidocr_run_lock = threading.Lock()
        self._rapidocr_engine = None
        self._rapidocr_runtime_cfg = None
        self._ocr_executor = ThreadPoolExecutor(max_workers=1)
        self._ocr_admission = _AdmissionController("ocr", 1)
        self._ocr_execution_timeout_seconds = 30

    def _load_family_serialized(self, family: str, loader: Any) -> None:
        loader()

    def _non_text_request_scope(self, **kwargs: Any):
        return nullcontext()

    def _non_text_request_scope_async(self, **kwargs: Any):
        raise NotImplementedError

    @staticmethod
    def _run_in_executor(executor: ThreadPoolExecutor, func: Any, *args: Any):
        raise NotImplementedError

    def _ensure_non_text_task_executors_ready(self) -> None:
        return None

    async def _await_with_timeout_and_cooperative_cancel(self, awaitable, **kwargs: Any):
        raise NotImplementedError


class _FakeRapidOCRResult:
    def __init__(self, boxes, txts, scores) -> None:
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


class OCRFunctionalTests(unittest.TestCase):
    def test_ocr_result_from_raw_object_normalizes_boxes_texts_scores(self) -> None:
        raw = _FakeRapidOCRResult(
            boxes=[
                [[1, 2], [11, 2], [11, 12], [1, 12]],
                [[5, 6], [25, 6], [25, 16], [5, 16]],
            ],
            txts=["hello", "world"],
            scores=[0.98, 0.87],
        )

        result = _DummyRapidOCRHost._ocr_result_from_raw(raw)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(["hello", "world"], result.texts)
        self.assertEqual(["0.98", "0.87"], result.scores)
        self.assertEqual("1.0", result.boxes[0].x)
        self.assertEqual("2.0", result.boxes[0].y)
        self.assertEqual("10.0", result.boxes[0].width)
        self.assertEqual("10.0", result.boxes[0].height)
        self.assertEqual("20.0", result.boxes[1].width)

    def test_ocr_result_from_tuple_handles_empty_payload(self) -> None:
        result = _DummyRapidOCRHost._ocr_result_from_raw(([], None))

        self.assertEqual([], result.texts)
        self.assertEqual([], result.scores)
        self.assertEqual([], result.boxes)


if __name__ == "__main__":
    unittest.main()
