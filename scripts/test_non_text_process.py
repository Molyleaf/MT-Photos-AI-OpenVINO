import asyncio
import itertools
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


_ORDERED_OPERATIONS = ("clip_img", "ocr", "represent")
_ORDERED_FAMILIES = {
    "clip_img": "vision",
    "ocr": "ocr",
    "represent": "face",
}


def _build_operation_result(operation: str, pid: int):
    if operation == "clip_img":
        return [float(pid)]
    if operation == "ocr":
        return {
            "texts": [str(pid)],
            "scores": ["1.0"],
            "boxes": [{"x": "0", "y": "0", "width": "1", "height": "1"}],
        }
    if operation == "represent":
        return [
            {
                "embedding": [float(pid)],
                "facial_area": {"x": 0, "y": 0, "w": 1, "h": 1},
                "face_confidence": 1.0,
            }
        ]
    raise RuntimeError(f"unsupported {operation}")


def _extract_operation_pid(operation: str, payload) -> int:
    if operation == "clip_img":
        return int(payload[0])
    if operation == "ocr":
        return int(payload.texts[0])
    return int(payload[0].embedding[0])


async def _invoke_operation(manager: NonTextProcessManager, operation: str, image: np.ndarray):
    if operation == "clip_img":
        return await manager.get_image_embedding_async(image)
    if operation == "ocr":
        return await manager.get_ocr_results_async(image)
    return await manager.get_face_representation_async(image)


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
        try:
            result = _build_operation_result(operation, pid)
        except RuntimeError as exc:
            response_queue.put(
                {"kind": "error", "request_id": request_id, "error": str(exc)}
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


def _hung_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        time.sleep(60.0)
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


def _mixed_latency_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    operation_delay_seconds = {
        "clip_img": 0.05,
        "ocr": 0.08,
        "represent": 0.35,
    }
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        request_id = str(message["request_id"])
        operation = str(message["operation"])
        time.sleep(operation_delay_seconds.get(operation, 0.02))
        try:
            result = _build_operation_result(operation, pid)
        except RuntimeError as exc:
            response_queue.put(
                {"kind": "error", "request_id": request_id, "error": str(exc)}
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

    async def test_ocr_family_stays_in_worker_process_and_releases_on_restart(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        first = await self.manager.get_ocr_results_async(image)
        second = await self.manager.get_ocr_results_async(image)

        self.assertEqual(first.texts, second.texts)
        self.assertEqual("ocr", self.manager.get_loaded_runtime_family())

        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            await asyncio.to_thread(self.manager.release_models_for_restart)

        self.assertIsNone(self.manager.get_loaded_runtime_family())
        memory_log_mock.assert_called_once_with("restart-worker-stop")

        third = await self.manager.get_ocr_results_async(image)
        self.assertNotEqual(first.texts, third.texts)
        self.assertEqual("ocr", self.manager.get_loaded_runtime_family())

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

    async def test_idle_release_waits_for_active_request_completion(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        with patch.dict(os.environ, {"NON_TEXT_IDLE_RELEASE_SECONDS": "0.2"}, clear=False):
            self.manager = NonTextProcessManager(worker_target=_slow_non_text_worker)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            request_task = asyncio.create_task(self.manager.get_image_embedding_async(image))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if self.manager.get_loaded_runtime_family() == "vision":
                    break
                await asyncio.sleep(0.05)
            else:
                self.fail("Vision worker did not start in time.")

            await asyncio.sleep(0.35)
            self.assertFalse(request_task.done())
            self.assertEqual("vision", self.manager.get_loaded_runtime_family())
            memory_log_mock.assert_not_called()

            await request_task

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self.manager.get_loaded_runtime_family() is None:
                    break
                await asyncio.sleep(0.1)
            else:
                self.fail("Idle release did not stop the worker process after request completion.")

        memory_log_mock.assert_any_call("idle-timeout-worker-stop")

    async def test_hung_worker_timeout_does_not_block_on_joinable_queue(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_hung_non_text_worker)
        self.manager._execution_timeout_seconds = 1

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        started = time.monotonic()
        with patch("non_text_process._log_current_process_memory") as memory_log_mock:
            with self.assertRaisesRegex(RuntimeError, "执行超时"):
                await self.manager.get_image_embedding_async(image)

        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0)
        self.assertIsNone(self.manager.get_loaded_runtime_family())
        memory_log_mock.assert_called_once_with("vision-timeout-worker-stop")

    async def test_mixed_family_out_of_order_requests_do_not_deadlock(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_mixed_latency_non_text_worker)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        face_task = asyncio.create_task(_invoke_operation(self.manager, "represent", image))

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.manager.get_loaded_runtime_family() == "face":
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("Face request did not acquire the worker in time.")

        clip_task = asyncio.create_task(_invoke_operation(self.manager, "clip_img", image))
        face_result, clip_result = await asyncio.wait_for(
            asyncio.gather(face_task, clip_task),
            timeout=6.0,
        )

        face_pid = _extract_operation_pid("represent", face_result)
        clip_pid = _extract_operation_pid("clip_img", clip_result)
        self.assertNotEqual(face_pid, clip_pid)
        self.assertEqual("vision", self.manager.get_loaded_runtime_family())

    async def test_pairwise_family_rotation_is_symmetric(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_mixed_latency_non_text_worker)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        for first_operation, second_operation in itertools.permutations(_ORDERED_OPERATIONS, 2):
            with self.subTest(first=first_operation, second=second_operation):
                await asyncio.to_thread(self.manager.release_models_for_restart)

                first_task = asyncio.create_task(
                    _invoke_operation(self.manager, first_operation, image)
                )
                deadline = time.monotonic() + 3.0
                expected_first_family = _ORDERED_FAMILIES[first_operation]
                while time.monotonic() < deadline:
                    if self.manager.get_loaded_runtime_family() == expected_first_family:
                        break
                    await asyncio.sleep(0.02)
                else:
                    self.fail(f"{first_operation} did not acquire the worker in time.")

                second_task = asyncio.create_task(
                    _invoke_operation(self.manager, second_operation, image)
                )
                done, _pending = await asyncio.wait(
                    {first_task, second_task},
                    timeout=6.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                self.assertIn(first_task, done)
                self.assertNotIn(second_task, done)

                first_result = await asyncio.wait_for(first_task, timeout=6.0)
                second_result = await asyncio.wait_for(second_task, timeout=6.0)
                first_pid = _extract_operation_pid(first_operation, first_result)
                second_pid = _extract_operation_pid(second_operation, second_result)
                self.assertNotEqual(first_pid, second_pid)
                self.assertEqual(
                    _ORDERED_FAMILIES[second_operation],
                    self.manager.get_loaded_runtime_family(),
                )

    async def test_round_robin_three_family_requests_complete_without_deadlock(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_mixed_latency_non_text_worker)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        operation_order = ("represent", "clip_img", "ocr")
        tasks = []
        for index, operation in enumerate(operation_order):
            if index == 0:
                task = asyncio.create_task(_invoke_operation(self.manager, operation, image))
                tasks.append(task)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if self.manager.get_loaded_runtime_family() == _ORDERED_FAMILIES[operation]:
                        break
                    await asyncio.sleep(0.02)
                else:
                    self.fail(f"{operation} did not acquire the worker in time.")
                continue
            tasks.append(asyncio.create_task(_invoke_operation(self.manager, operation, image)))

        first_done, _pending = await asyncio.wait(
            set(tasks),
            timeout=6.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        self.assertIn(tasks[0], first_done)

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=12.0)
        pids = [
            _extract_operation_pid(operation, payload)
            for operation, payload in zip(operation_order, results)
        ]
        self.assertEqual(3, len(results))
        self.assertEqual(3, len(set(pids)))
        self.assertEqual("ocr", self.manager.get_loaded_runtime_family())

    async def test_mixed_family_burst_requests_complete_under_pressure(self) -> None:
        await asyncio.to_thread(self.manager.release_all_models)
        self.manager = NonTextProcessManager(worker_target=_mixed_latency_non_text_worker)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        operations = [
            ("represent", 0.00),
            ("clip_img", 0.01),
            ("ocr", 0.02),
            ("clip_img", 0.03),
            ("represent", 0.04),
            ("ocr", 0.05),
            ("clip_img", 0.06),
            ("represent", 0.07),
            ("ocr", 0.08),
        ]

        async def invoke(operation: str, delay_seconds: float):
            await asyncio.sleep(delay_seconds)
            return operation, await _invoke_operation(self.manager, operation, image)

        started = time.monotonic()
        results = await asyncio.wait_for(
            asyncio.gather(*(invoke(operation, delay) for operation, delay in operations)),
            timeout=12.0,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(len(operations), len(results))
        self.assertLess(elapsed, 12.0)

        clip_pids = []
        ocr_pids = []
        face_pids = []
        for operation, payload in results:
            operation_pid = _extract_operation_pid(operation, payload)
            if operation == "clip_img":
                clip_pids.append(operation_pid)
            elif operation == "ocr":
                ocr_pids.append(operation_pid)
            else:
                face_pids.append(operation_pid)

        self.assertTrue(clip_pids)
        self.assertTrue(ocr_pids)
        self.assertTrue(face_pids)
        self.assertGreaterEqual(len(set(clip_pids + ocr_pids + face_pids)), 2)


if __name__ == "__main__":
    unittest.main()
