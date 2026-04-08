import sys
import threading
import types
import unittest
from concurrent.futures import Future
from importlib.util import find_spec
import logging
from pathlib import Path
from unittest.mock import Mock, patch

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

    fake_rapidocr.RapidOCR = _FakeRapidOCR
    fake_rapidocr_utils = types.ModuleType("rapidocr.utils")
    fake_rapidocr_log = types.ModuleType("rapidocr.utils.log")
    fake_rapidocr_log.logger = logging.getLogger("rapidocr")
    fake_rapidocr_typings = types.ModuleType("rapidocr.utils.typings")
    fake_rapidocr_typings.EngineType = object
    sys.modules["rapidocr"] = fake_rapidocr
    sys.modules["rapidocr.utils"] = fake_rapidocr_utils
    sys.modules["rapidocr.utils.log"] = fake_rapidocr_log
    sys.modules["rapidocr.utils.typings"] = fake_rapidocr_typings

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

from models.common import _ClipImageTask, _FaceInferenceTask, _OpenVinoPreprocessRunner
from models.runtime import AIModels


class _FakeCompiledModel:
    def __init__(self) -> None:
        self.create_infer_request_calls = 0

    def create_infer_request(self) -> object:
        self.create_infer_request_calls += 1
        return object()


class _FakeThread:
    def __init__(self) -> None:
        self.join_calls: list[float | None] = []
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _FakeLoop:
    def __init__(self) -> None:
        self.stop_called = False

    def call_soon_threadsafe(self, callback) -> None:
        callback()

    def stop(self) -> None:
        self.stop_called = True


class _FakeExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _FakeLock:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _FakeConcurrentResult:
    def __init__(self, value) -> None:
        self._value = value

    def result(self, timeout: float | None = None):
        return self._value


class RuntimeCleanupTests(unittest.TestCase):
    def test_ai_models_init_failure_triggers_release_all_models(self) -> None:
        with (
            patch("models.runtime._prepare_windows_openvino_runtime"),
            patch.object(AIModels, "_initialize_paths"),
            patch.object(AIModels, "_initialize_openvino_runtime"),
            patch.object(AIModels, "_initialize_model_load_locks"),
            patch.object(AIModels, "_acquire_single_process_lock"),
            patch.object(AIModels, "_initialize_clip_image_state"),
            patch.object(AIModels, "_initialize_non_text_model_state"),
            patch.object(AIModels, "_initialize_execution_controls"),
            patch.object(AIModels, "_initialize_non_text_family_state"),
            patch.object(AIModels, "_start_clip_image_worker"),
            patch.object(AIModels, "_start_background_services"),
            patch.object(AIModels, "_log_ready"),
            patch.object(
                AIModels,
                "_start_face_batch_service",
                side_effect=RuntimeError("face worker boom"),
            ),
            patch.object(AIModels, "release_all_models", autospec=True) as release_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "face worker boom"):
                AIModels()

        release_mock.assert_called_once()

    def test_release_all_models_cleans_pending_tasks_and_shuts_down_resources(self) -> None:
        models = AIModels.__new__(AIModels)
        models._pid = 123
        models._stopping = False
        AIModels._initialize_release_defaults(models)
        models._execution_timeout_seconds = 3

        clip_task = _ClipImageTask(payload=object(), future=Future(), created_at=0.0)
        face_task = _FaceInferenceTask(payload=object(), future=Future(), created_at=0.0)
        clip_loop = _FakeLoop()
        clip_worker = _FakeThread()
        control_executor = _FakeExecutor()
        face_executor = _FakeExecutor()
        shared_executor = _FakeExecutor()
        ocr_executor = _FakeExecutor()
        process_lock = _FakeLock()

        models._clip_image_dispatch_loop = clip_loop
        models._clip_image_worker = clip_worker
        models._control_executor = control_executor
        models._face_preprocess_executor = face_executor
        models._shared_cpu_executor = shared_executor
        models._ocr_executor = ocr_executor
        models._single_process_lock = process_lock
        models._stop_face_batch_service = Mock(return_value=[face_task])
        models._unload_everything_locked = Mock()

        def fake_run_coroutine_threadsafe(coro, loop):
            self.assertIs(loop, clip_loop)
            coro.close()
            return _FakeConcurrentResult([clip_task])

        with patch(
            "models.runtime.asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ):
            models.release_all_models()

        self.assertTrue(models._stopping)
        self.assertIsNone(clip_task.payload)
        self.assertIsNone(face_task.payload)
        self.assertTrue(clip_task.future.done())
        self.assertTrue(face_task.future.done())
        self.assertEqual(1, len(clip_worker.join_calls))
        self.assertTrue(clip_loop.stop_called)
        self.assertEqual([(True, True)], control_executor.shutdown_calls)
        self.assertEqual([(True, True)], face_executor.shutdown_calls)
        self.assertEqual([(True, True)], shared_executor.shutdown_calls)
        self.assertEqual([(True, True)], ocr_executor.shutdown_calls)
        self.assertEqual(1, process_lock.release_calls)
        models._stop_face_batch_service.assert_called_once_with()
        models._unload_everything_locked.assert_called_once_with()

    def test_release_openvino_runtime_if_unused_drops_core_and_remote_context(self) -> None:
        models = AIModels.__new__(AIModels)
        models._stopping = False
        AIModels._initialize_release_defaults(models)
        models.core = object()
        models._clip_remote_context = object()
        models._clip_remote_context_device_name = "GPU"

        released = models._release_openvino_runtime_if_unused_locked(keep_family=None)

        self.assertTrue(released)
        self.assertIsNone(models.core)
        self.assertIsNone(models._clip_remote_context)
        self.assertIsNone(models._clip_remote_context_device_name)

    def test_openvino_preprocess_runner_release_drops_cached_requests(self) -> None:
        compiled_model = _FakeCompiledModel()
        runner = _OpenVinoPreprocessRunner(
            compiled_model=compiled_model,
            input_port="input",
            output_port="output",
            runner_name="clip_ppp",
            input_height=224,
            input_width=224,
        )

        first_request = runner._get_request()
        second_request = runner._get_request()

        self.assertIs(first_request, second_request)
        self.assertEqual(1, compiled_model.create_infer_request_calls)

        runner.release()

        self.assertEqual({}, runner._requests_by_thread)
        self.assertIsNone(runner.compiled_model)
        self.assertIsNone(runner.input_port)
        self.assertIsNone(runner.output_port)
        with self.assertRaisesRegex(RuntimeError, "released"):
            runner._get_request()


if __name__ == "__main__":
    unittest.main()
