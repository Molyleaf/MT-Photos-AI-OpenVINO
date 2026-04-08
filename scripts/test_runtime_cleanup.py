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


class _FakeOrtSession:
    def __init__(self) -> None:
        self._sess = object()
        self._sess_options = object()
        self._sess_options_initial = object()
        self._inputs_meta = object()
        self._outputs_meta = object()
        self._overridable_initializers = object()
        self._input_meminfos = object()
        self._output_meminfos = object()
        self._input_epdevices = object()
        self._model_meta = object()
        self._providers = ["OpenVINOExecutionProvider"]
        self._provider_options = {"OpenVINOExecutionProvider": {"device_type": "GPU"}}
        self._fallback_providers = ["CPUExecutionProvider"]
        self._profiling_start_time_ns = 1
        self._profiling_start_time = 1
        self._model_path = "model.onnx"
        self._model_bytes = b"model"


class _FakeInsightFaceModel:
    def __init__(self) -> None:
        self.center_cache = {"a": 1}
        self.session = _FakeOrtSession()
        self.input_name = "input"
        self.output_names = ["output"]
        self.input_shape = [1, 3, 112, 112]
        self.output_shape = [1, 512]
        self.output_shapes = [[1, 10]]


class _FakeFaceApp:
    def __init__(self) -> None:
        self.det_model = _FakeInsightFaceModel()
        self.models = {
            "detection": self.det_model,
            "recognition": _FakeInsightFaceModel(),
        }
        self.det_thresh = 0.5
        self.det_size = (640, 640)


class _CapturedFaceAnalysis:
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        _CapturedFaceAnalysis.last_kwargs = dict(kwargs)
        self.models = {
            "recognition": object(),
        }


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
        models._drop_non_text_filesystem_page_cache = Mock()

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
        models._drop_non_text_filesystem_page_cache.assert_called_once_with()
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

    def test_release_models_for_restart_recycles_support_workers_and_executors(self) -> None:
        models = AIModels.__new__(AIModels)
        models._pid = 123
        models._stopping = False
        AIModels._initialize_release_defaults(models)
        models._execution_timeout_seconds = 3

        clip_task = _ClipImageTask(payload=object(), future=Future(), created_at=0.0)
        face_task = _FaceInferenceTask(payload=object(), future=Future(), created_at=0.0)
        clip_loop = _FakeLoop()
        clip_worker = _FakeThread()
        face_loop = _FakeLoop()
        face_worker = _FakeThread()
        shared_executor = _FakeExecutor()
        ocr_executor = _FakeExecutor()
        face_executor = _FakeExecutor()

        models._clip_image_dispatch_loop = clip_loop
        models._clip_image_worker = clip_worker
        models._face_dispatch_loop = face_loop
        models._face_worker = face_worker
        models._shared_cpu_executor = shared_executor
        models._ocr_executor = ocr_executor
        models._face_preprocess_executor = face_executor
        models._non_text_state = Mock()
        models._unload_non_text_models = Mock(return_value=["face"])
        models._drop_non_text_filesystem_page_cache = Mock()

        def fake_run_coroutine_threadsafe(coro, loop):
            coro.close()
            if loop is clip_loop:
                return _FakeConcurrentResult([clip_task])
            if loop is face_loop:
                return _FakeConcurrentResult([face_task])
            raise AssertionError(f"unexpected loop: {loop}")

        with patch(
            "models.runtime.asyncio.run_coroutine_threadsafe",
            side_effect=fake_run_coroutine_threadsafe,
        ), patch("models.insightface.asyncio.run_coroutine_threadsafe", side_effect=fake_run_coroutine_threadsafe):
            models.release_models_for_restart()

        self.assertTrue(clip_task.future.done())
        self.assertTrue(face_task.future.done())
        self.assertIsNone(clip_task.payload)
        self.assertIsNone(face_task.payload)
        self.assertTrue(clip_loop.stop_called)
        self.assertTrue(face_loop.stop_called)
        self.assertEqual([(True, True)], shared_executor.shutdown_calls)
        self.assertEqual([(True, True)], ocr_executor.shutdown_calls)
        self.assertEqual([(True, True)], face_executor.shutdown_calls)
        self.assertIsNone(models._shared_cpu_executor)
        self.assertIsNone(models._ocr_executor)
        self.assertIsNone(models._face_preprocess_executor)
        models._non_text_state.begin_release.assert_called_once_with()
        models._non_text_state.wait_for_drain.assert_called_once_with()
        models._non_text_state.finish_release.assert_called_once_with()
        models._drop_non_text_filesystem_page_cache.assert_called_once_with()
        models._unload_non_text_models.assert_called_once_with()

    def test_release_non_text_models_sync_keeps_release_local_to_worker_runtime(self) -> None:
        models = AIModels.__new__(AIModels)
        models._pid = 123
        models._stopping = False
        AIModels._initialize_release_defaults(models)
        models._execution_timeout_seconds = 3
        models._non_text_state = Mock()
        models._unload_non_text_models = Mock(return_value=["face"])
        models._recycle_non_text_runtime_support_resources = Mock(return_value=False)
        models._drop_non_text_filesystem_page_cache = Mock()

        models._release_non_text_models_sync(reason="idle-timeout")

        models._non_text_state.begin_release.assert_called_once_with()
        models._non_text_state.wait_for_drain.assert_called_once_with()
        models._non_text_state.finish_release.assert_called_once_with()
        models._drop_non_text_filesystem_page_cache.assert_called_once_with()

    def test_dispose_insightface_face_analysis_clears_ort_runtime_refs(self) -> None:
        face_app = _FakeFaceApp()
        det_session = face_app.det_model.session
        rec_session = face_app.models["recognition"].session

        AIModels._dispose_insightface_face_analysis(face_app)

        self.assertIsNone(face_app.det_model)
        self.assertEqual({}, face_app.models)
        self.assertIsNone(face_app.det_thresh)
        self.assertIsNone(face_app.det_size)
        self.assertIsNone(det_session._sess)
        self.assertIsNone(det_session._provider_options)
        self.assertIsNone(rec_session._sess)
        self.assertIsNone(rec_session._providers)

        detached_model = _FakeInsightFaceModel()
        AIModels._dispose_insightface_model(detached_model)
        self.assertEqual({}, detached_model.center_cache)
        self.assertIsNone(detached_model.session)
        self.assertIsNone(detached_model.input_name)
        self.assertIsNone(detached_model.output_names)
        self.assertIsNone(detached_model.input_shape)
        self.assertIsNone(detached_model.output_shape)
        self.assertIsNone(detached_model.output_shapes)

    def test_instantiate_insightface_uses_low_residue_session_options(self) -> None:
        _CapturedFaceAnalysis.last_kwargs = None
        models = AIModels.__new__(AIModels)
        models._stopping = False
        AIModels._initialize_release_defaults(models)
        runtime_root = PROJECT_ROOT / "tests" / "runtime"
        source_root = PROJECT_ROOT / "tests" / "source"

        with (
            patch("models.insightface.FaceAnalysis", _CapturedFaceAnalysis),
            patch.object(AIModels, "_resolve_insightface_source_model_dir", return_value=source_root),
            patch.object(AIModels, "_prepare_insightface_runtime_root", return_value=runtime_root),
            patch.object(AIModels, "_validate_insightface_loaded_modules"),
            patch.object(AIModels, "_normalize_insightface_recognition_state"),
        ):
            _, actual_runtime_root = models._instantiate_insightface_face_analysis(
                ["OpenVINOExecutionProvider"],
                {"device_type": "GPU"},
            )

        self.assertEqual(runtime_root, actual_runtime_root)
        self.assertIsNotNone(_CapturedFaceAnalysis.last_kwargs)
        sess_options = _CapturedFaceAnalysis.last_kwargs["sess_options"]
        self.assertFalse(sess_options.enable_cpu_mem_arena)
        self.assertFalse(sess_options.enable_mem_pattern)
        self.assertEqual(1, sess_options.inter_op_num_threads)
        self.assertEqual(1, sess_options.intra_op_num_threads)


if __name__ == "__main__":
    unittest.main()
