import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from non_text_process import NonTextProcessManager


def _fake_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        request_id = str(message["request_id"])
        operation = str(message["operation"])
        if operation == "clip_img":
            result = [float(pid)]
        elif operation == "ocr":
            result = {
                "texts": [str(pid)],
                "scores": ["1.0"],
                "boxes": [{"x": "0", "y": "0", "width": "1", "height": "1"}],
            }
        elif operation == "represent":
            result = [
                {
                    "embedding": [float(pid)],
                    "facial_area": {"x": 0, "y": 0, "w": 1, "h": 1},
                    "face_confidence": 1.0,
                }
            ]
        else:
            response_queue.put(
                {"kind": "error", "request_id": request_id, "error": f"unsupported {operation}"}
            )
            request_queue.task_done()
            continue
        response_queue.put(
            {
                "kind": "result",
                "request_id": request_id,
                "operation": operation,
                "result": result,
            }
        )
        request_queue.task_done()
    response_queue.put({"kind": "stopped", "pid": pid})


def _slow_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        time.sleep(2.0)
        response_queue.put(
            {
                "kind": "result",
                "request_id": str(message["request_id"]),
                "operation": str(message["operation"]),
                "result": [float(pid)],
            }
        )
        request_queue.task_done()
    response_queue.put({"kind": "stopped", "pid": pid})


class NonTextProcessManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._env = patch.dict(
            os.environ,
            {
                "INFERENCE_QUEUE_MAX_SIZE": "4",
                "INFERENCE_QUEUE_TIMEOUT": "3",
                "INFERENCE_EXEC_TIMEOUT": "3",
                "OCR_EXEC_TIMEOUT": "3",
                "NON_TEXT_IDLE_RELEASE_SECONDS": "0",
            },
            clear=False,
        )
        self._env.start()
        self.manager = NonTextProcessManager(worker_target=_fake_non_text_worker)

    async def asyncTearDown(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self._env.stop()

    async def test_same_family_requests_reuse_worker_process(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        first = await self.manager.get_image_embedding_async(image)
        second = await self.manager.get_image_embedding_async(image)

        self.assertEqual(first, second)
        self.assertEqual("vision", self.manager.get_loaded_runtime_family())

    async def test_family_switch_restarts_worker_process(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        clip_result = await self.manager.get_image_embedding_async(image)
        ocr_result = await self.manager.get_ocr_results_async(image)
        face_result = await self.manager.get_face_representation_async(image)

        clip_pid = int(clip_result[0])
        ocr_pid = int(ocr_result.texts[0])
        face_pid = int(face_result[0].embedding[0])
        self.assertNotEqual(clip_pid, ocr_pid)
        self.assertNotEqual(ocr_pid, face_pid)
        self.assertEqual("face", self.manager.get_loaded_runtime_family())

    async def test_release_models_for_restart_drops_loaded_family(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        first = await self.manager.get_image_embedding_async(image)
        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            await asyncio.to_thread(self.manager.release_models_for_restart)
        self.assertIsNone(self.manager.get_loaded_runtime_family())
        memory_log_mock.assert_called_once_with("restart-worker-stop")
        second = await self.manager.get_image_embedding_async(image)

        self.assertNotEqual(first, second)

    async def test_request_timeout_releases_worker_and_clears_loaded_family(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_slow_non_text_worker)
        self.manager._execution_timeout_seconds = 1

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            with self.assertRaisesRegex(RuntimeError, "执行超时"):
                await self.manager.get_image_embedding_async(image)

        self.assertIsNone(self.manager.get_loaded_runtime_family())
        memory_log_mock.assert_called_once_with("vision-timeout-worker-stop")

    async def test_idle_release_stops_worker_process(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        with patch.dict(os.environ, {"NON_TEXT_IDLE_RELEASE_SECONDS": "0.2"}, clear=False):
            self.manager = NonTextProcessManager(worker_target=_fake_non_text_worker)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            await self.manager.get_image_embedding_async(image)

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self.manager.get_loaded_runtime_family() is None:
                    break
                await asyncio.sleep(0.1)
            else:
                self.fail("Idle release did not stop the worker process in time.")

        memory_log_mock.assert_any_call("idle-timeout-worker-stop")


if __name__ == "__main__":
    unittest.main()
