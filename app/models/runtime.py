import asyncio
import gc
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

import openvino as ov
from transitions import Machine

from .clip_image import ClipImageMixin
from .common import (
    _AdmissionController,
    _FaceInferenceTask,
    _ManagedLease,
    NonTextFamily,
    _ClipImageTask,
    _InferenceCancelled,
    _InterProcessFileLock,
    _OpenVinoPreprocessRunner,
    _prepare_windows_openvino_runtime,
    _extract_explicit_gpu_devices,
    _get_openvino_gpu_devices,
    _drop_filesystem_page_cache,
    _normalize_openvino_devices,
    _summarize_exception,
    _trim_process_memory,
)
from .constants import (
    INSIGHTFACE_REQUEST_CAPACITY,
    INSIGHTFACE_SINGLE_LANE,
    LOG,
)
from .insightface import InsightFaceMixin
from .rapidocr_lib import RapidOCRMixin
from .runtime_settings import (
    load_clip_image_runtime_settings,
    load_execution_control_settings,
    load_runtime_path_settings,
)


class _NonTextFamilyStateModel:
    """`transitions` injects the trigger methods onto this model instance."""

    def __init__(self) -> None:
        self.state = "idle"


class _NonTextFamilyStateMachine:
    def __init__(
        self,
        *,
        unload_callback: Callable[[NonTextFamily], List[str]],
        is_stopping: Callable[[], bool],
    ) -> None:
        self._unload_callback = unload_callback
        self._is_stopping = is_stopping
        self._condition = threading.Condition()
        self._model = _NonTextFamilyStateModel()
        self._machine = Machine(
            model=self._model,
            states=["idle", "switching", "vision", "ocr", "face"],
            initial="idle",
            auto_transitions=False,
        )
        self._machine.add_transition("begin_switch", ["idle", "vision", "ocr", "face"], "switching")
        self._machine.add_transition(
            "release_to_idle",
            ["idle", "switching", "vision", "ocr", "face"],
            "idle",
        )
        self._machine.add_transition("activate_vision", ["idle", "switching", "vision"], "vision")
        self._machine.add_transition("activate_ocr", ["idle", "switching", "ocr"], "ocr")
        self._machine.add_transition("activate_face", ["idle", "switching", "face"], "face")
        self._inflight: Dict[NonTextFamily, int] = {
            "vision": 0,
            "ocr": 0,
            "face": 0,
        }

    def _raise_if_stopping(self) -> None:
        if self._is_stopping():
            raise RuntimeError("模型服务已关闭")

    def _active_family_locked(self) -> Optional[NonTextFamily]:
        if self._model.state in {"idle", "switching"}:
            return None
        return self._model.state  # type: ignore[return-value]

    def _activate_family_locked(self, family: NonTextFamily) -> None:
        if family == "vision":
            self._model.activate_vision()
        elif family == "ocr":
            self._model.activate_ocr()
        else:
            self._model.activate_face()

    def acquire(
        self,
        family: NonTextFamily,
        abort_event: Optional[threading.Event] = None,
    ) -> bool:
        while True:
            previous_family: Optional[NonTextFamily]
            with self._condition:
                while self._model.state == "switching":
                    if abort_event is not None and abort_event.is_set():
                        return False
                    self._raise_if_stopping()
                    self._condition.wait(timeout=0.1 if abort_event is not None else None)

                if abort_event is not None and abort_event.is_set():
                    return False
                self._raise_if_stopping()
                previous_family = self._active_family_locked()
                if previous_family is None or previous_family == family:
                    self._activate_family_locked(family)
                    self._inflight[family] += 1
                    return True

                if self._inflight[previous_family] > 0:
                    self._condition.wait(timeout=0.1 if abort_event is not None else None)
                    continue

                self._model.begin_switch()

            switch_exc: Optional[Exception] = None
            unloaded_families: List[str] = []
            try:
                unloaded_families = self._unload_callback(family)
            except Exception as exc:
                switch_exc = exc

            aborted = abort_event is not None and abort_event.is_set()
            with self._condition:
                if switch_exc is None and not aborted:
                    self._activate_family_locked(family)
                    self._inflight[family] += 1
                else:
                    self._model.release_to_idle()
                self._condition.notify_all()

            if switch_exc is not None:
                raise switch_exc
            if aborted:
                return False
            if unloaded_families:
                LOG.info(
                    "Switched runtime family from %s to %s; released=%s",
                    previous_family,
                    family,
                    ",".join(unloaded_families),
                )
            return True

    def release(self, family: NonTextFamily) -> None:
        with self._condition:
            inflight = self._inflight[family]
            if inflight <= 0:
                return
            self._inflight[family] = inflight - 1
            if self._inflight[family] == 0:
                self._condition.notify_all()

    def begin_release(self) -> None:
        with self._condition:
            while self._model.state == "switching":
                if self._is_stopping():
                    return
                self._condition.wait()
            self._model.begin_switch()

    def wait_for_drain(self) -> None:
        with self._condition:
            while any(self._inflight.values()):
                self._condition.wait()

    def finish_release(self) -> None:
        with self._condition:
            self._model.release_to_idle()
            self._condition.notify_all()


class AIModels(ClipImageMixin, RapidOCRMixin, InsightFaceMixin):
    """
    This runtime is loaded inside the non-text worker process. Image-CLIP uses
    a dedicated batch queue after standardized preprocessing, while runtime
    model families still lazy-load and switch synchronously inside that worker.
    """
    ov_cache_dir: Optional[Path]

    def _compile_clip_model(
        self,
        model_or_path: Any,
        performance_hint: str,
    ) -> ov.CompiledModel:
        return ClipImageMixin._compile_clip_model(self, model_or_path, performance_hint)

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._stopping = False
        self._initialize_release_defaults()
        _prepare_windows_openvino_runtime()
        try:
            self._initialize_paths()
            self._initialize_openvino_runtime()
            self._initialize_model_load_locks()
            self._acquire_single_process_lock()
            self._initialize_clip_image_state()
            self._initialize_non_text_model_state()
            self._initialize_execution_controls()
            self._initialize_non_text_family_state()
            self._start_clip_image_worker()
            self._start_face_batch_service()
            self._start_background_services()
            self._log_ready()
        except Exception:
            try:
                self.release_all_models()
            except Exception as cleanup_exc:
                LOG.warning(
                    "AIModels initialization cleanup failed after startup error: %s",
                    cleanup_exc,
                    exc_info=True,
                )
            raise

    def _initialize_release_defaults(self) -> None:
        self.core = None
        self.ov_cache_dir = None
        self._clip_remote_context_device_name = None
        self._clip_remote_context = None
        self._single_process_lock = None

        self._model_lock = threading.Lock()
        self._clip_vision_load_lock = threading.Lock()
        self._rapidocr_load_lock = threading.Lock()
        self._face_load_lock = threading.Lock()
        self._clip_image_worker_lock = threading.Lock()
        self._face_worker_lock = threading.Lock()
        self._executor_lock = threading.Lock()

        self._clip_vision_model = None
        self._clip_vision_ppp = None
        self._clip_vision_request = None
        self._clip_image_batch_size = 1
        self._clip_image_batch_wait_seconds = 0.0
        self._clip_image_dispatch_loop = None
        self._clip_image_queue = None
        self._clip_image_loop_ready = threading.Event()
        self._clip_image_worker = None

        self._rapidocr_engine = None
        self._rapidocr_engines = None
        self._rapidocr_engine_pool = None
        self._rapidocr_runtime_cfg = None

        self._face_engine = None
        self._face_dispatch_loop = None
        self._face_task_queue = None
        self._face_loop_ready = threading.Event()
        self._face_worker = None
        self._face_preprocess_device = None

        self._queue_capacity = 1
        self._queue_timeout_seconds = 30
        self._execution_timeout_seconds = 30
        self._ocr_execution_timeout_seconds = 30
        self._idle_release_timeout_seconds = 0
        self._ocr_prewarm_enabled = False
        self._ocr_prewarm_delay_seconds = 0.0

        self._image_admission = _AdmissionController("image", 1)
        self._ocr_worker_count = 1
        self._face_preprocess_worker_count = 1
        self._ocr_admission = _AdmissionController("ocr", 1)
        self._face_admission = _AdmissionController("face", 1)
        self._face_batch_size = 1
        self._face_batch_wait_seconds = 0.0
        self._face_queue_capacity = 1

        self._shared_cpu_executor = None
        self._control_executor = None
        self._ocr_executor = None
        self._face_preprocess_executor = None

        self._request_activity_lock = threading.Lock()
        self._last_request_activity_monotonic = time.monotonic()
        self._idle_release_stop = threading.Event()
        self._idle_release_wakeup = threading.Event()
        self._idle_release_thread = None
        self._background_prewarm_cancel = threading.Event()
        self._background_prewarm_thread = None

        self._non_text_state = _NonTextFamilyStateMachine(
            unload_callback=self._unload_non_text_models,
            is_stopping=lambda: self._stopping,
        )

    def _initialize_paths(self) -> None:
        self._path_settings = load_runtime_path_settings()
        self._clip_image_settings = load_clip_image_runtime_settings()
        self.model_base_path = self._path_settings.model_base_path
        self.insightface_root = self._path_settings.insightface_root
        self.insightface_model_root = self._path_settings.insightface_model_root
        self.qa_clip_path = self._path_settings.qa_clip_path
        self._clip_inference_device = self._clip_image_settings.inference_device
        self.ov_cache_dir = self._path_settings.ov_cache_dir
        self.rapidocr_config_path = self._path_settings.rapidocr_config_path
        self.rapidocr_model_dir = self._path_settings.rapidocr_model_dir
        self.rapidocr_model_dir_path = self._path_settings.rapidocr_model_dir_path
        self.rapidocr_font_path = self._path_settings.rapidocr_font_path
        self._runtime_state_dir = self._path_settings.runtime_state_dir

    def _initialize_openvino_runtime(self) -> None:
        self._ensure_openvino_runtime()

    def _ensure_openvino_runtime(self) -> ov.Core:
        core = getattr(self, "core", None)
        if core is not None:
            return core

        core = ov.Core()
        self.core = core
        self._configure_openvino_cache()
        self._clip_remote_context_device_name = None
        self._clip_remote_context = self._init_clip_remote_context()
        return core

    def _initialize_model_load_locks(self) -> None:
        self._model_lock = threading.Lock()
        self._clip_vision_load_lock = threading.Lock()
        self._rapidocr_load_lock = threading.Lock()
        self._face_load_lock = threading.Lock()
        self._single_process_lock = _InterProcessFileLock(
            self._runtime_state_dir / "single-process.lock"
        )

    def _acquire_single_process_lock(self) -> None:
        acquired = self._single_process_lock.acquire(timeout=0.0, blocking=False)
        if acquired:
            return
        raise RuntimeError(
            "服务当前固定为单进程运行；检测到已有实例持有运行锁。"
        )

    def _release_single_process_lock(self) -> None:
        lock = getattr(self, "_single_process_lock", None)
        if lock is not None:
            lock.release()

    def _initialize_clip_image_state(self) -> None:
        self._clip_vision_model: Optional[ov.CompiledModel] = None
        self._clip_vision_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._clip_vision_request: Optional[ov.InferRequest] = None
        self._clip_image_batch_size = self._clip_image_settings.batch_size
        self._clip_image_batch_wait_seconds = self._clip_image_settings.batch_wait_seconds
        self._clip_image_dispatch_loop: Optional[asyncio.AbstractEventLoop] = None
        self._clip_image_queue: Optional[asyncio.Queue[Optional[_ClipImageTask]]] = None
        self._clip_image_loop_ready = threading.Event()
        self._clip_image_worker: Optional[threading.Thread] = None

    def _initialize_non_text_model_state(self) -> None:
        self._rapidocr_engine = None
        self._rapidocr_engines = None
        self._rapidocr_engine_pool = None
        self._rapidocr_runtime_cfg: Optional[Dict[str, Any]] = None
        self._face_engine = None
        self._face_dispatch_loop: Optional[asyncio.AbstractEventLoop] = None
        self._face_task_queue = None
        self._face_loop_ready = threading.Event()
        self._face_worker: Optional[threading.Thread] = None
        self._face_preprocess_device: Optional[str] = None

    def _initialize_execution_controls(self) -> None:
        self._execution_settings = load_execution_control_settings()
        self._queue_capacity = self._execution_settings.queue_capacity
        if self._execution_settings.configured_queue_capacity > self._queue_capacity:
            LOG.warning(
                "INFERENCE_QUEUE_MAX_SIZE=%s exceeds MT-Photos safe limit %s; capping to %s.",
                self._execution_settings.configured_queue_capacity,
                self._queue_capacity,
                self._queue_capacity,
            )

        self._queue_timeout_seconds = self._execution_settings.queue_timeout_seconds
        self._execution_timeout_seconds = self._execution_settings.execution_timeout_seconds
        self._ocr_execution_timeout_seconds = (
            self._execution_settings.ocr_execution_timeout_seconds
        )
        self._idle_release_timeout_seconds = (
            self._execution_settings.idle_release_timeout_seconds
        )
        self._ocr_prewarm_enabled = self._execution_settings.ocr_prewarm_enabled
        self._ocr_prewarm_delay_seconds = self._execution_settings.ocr_prewarm_delay_seconds

        self._image_admission = _AdmissionController("image", self._queue_capacity)
        self._ocr_worker_count = self._execution_settings.ocr_worker_count
        self._face_preprocess_worker_count = (
            self._execution_settings.face_preprocess_worker_count
        )
        self._ocr_admission = _AdmissionController(
            "ocr",
            self._execution_settings.ocr_admission_capacity,
        )
        self._face_admission = _AdmissionController(
            "face",
            self._execution_settings.face_batch_size,
        )
        self._face_batch_size = self._face_admission.capacity
        self._face_batch_wait_seconds = self._execution_settings.face_batch_wait_seconds
        self._face_queue_capacity = self._execution_settings.face_queue_capacity

        self._shared_cpu_executor = None
        self._control_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ai-ctl",
        )
        self._ocr_executor = None
        self._face_preprocess_executor = None
        self._ensure_non_text_task_executors_ready()

        self._request_activity_lock = threading.Lock()
        self._last_request_activity_monotonic = time.monotonic()
        self._idle_release_stop = threading.Event()
        self._idle_release_wakeup = threading.Event()
        self._idle_release_thread: Optional[threading.Thread] = None
        self._background_prewarm_cancel = threading.Event()
        self._background_prewarm_thread: Optional[threading.Thread] = None

    def _initialize_non_text_family_state(self) -> None:
        self._non_text_state = _NonTextFamilyStateMachine(
            unload_callback=self._unload_non_text_models,
            is_stopping=lambda: self._stopping,
        )

    @staticmethod
    def _build_shared_cpu_executor() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=max(2, min(8, os.cpu_count() or 4)),
            thread_name_prefix="ai-cpu",
        )

    def _build_ocr_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=self._ocr_worker_count,
            thread_name_prefix="ocr",
        )

    def _build_face_preprocess_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=self._face_preprocess_worker_count,
            thread_name_prefix="face-pre",
        )

    def _ensure_non_text_task_executors_ready(self) -> None:
        with self._executor_lock:
            if self._shared_cpu_executor is None:
                self._shared_cpu_executor = self._build_shared_cpu_executor()
            if self._ocr_executor is None:
                self._ocr_executor = self._build_ocr_executor()
            if self._face_preprocess_executor is None:
                self._face_preprocess_executor = self._build_face_preprocess_executor()

    def _shutdown_non_text_task_executors(self) -> bool:
        recycled = False
        with self._executor_lock:
            executors = (
                ("shared CPU", self._shared_cpu_executor),
                ("OCR", self._ocr_executor),
                ("face preprocess", self._face_preprocess_executor),
            )
            self._shared_cpu_executor = None
            self._ocr_executor = None
            self._face_preprocess_executor = None

        for executor_name, executor in executors:
            if executor is None:
                continue
            recycled = True
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                LOG.warning("%s executor shutdown failed during runtime recycle: %s", executor_name, exc)
        return recycled

    def _start_clip_image_worker(self) -> None:
        with self._clip_image_worker_lock:
            if self._clip_image_worker is not None and self._clip_image_dispatch_loop is not None:
                return
            self._clip_image_loop_ready.clear()
            self._clip_image_worker = threading.Thread(
                target=self._clip_image_worker_thread_main,
                name="ai-clip-queue",
                daemon=True,
            )
            self._clip_image_worker.start()
            ready = self._clip_image_loop_ready.wait(
                timeout=max(2.0, float(self._execution_timeout_seconds))
            )
            if not ready:
                self._clip_image_worker = None
                raise RuntimeError("CLIP image queue worker failed to initialize in time.")

    def _ensure_clip_image_worker_ready(self) -> None:
        if self._stopping:
            raise RuntimeError("模型服务已关闭")
        if self._clip_image_worker is not None and self._clip_image_dispatch_loop is not None:
            return
        self._start_clip_image_worker()

    def _stop_clip_image_worker(self) -> List[_ClipImageTask]:
        pending_tasks: List[_ClipImageTask] = []
        join_timeout_seconds = max(2.0, float(self._execution_timeout_seconds))
        with self._clip_image_worker_lock:
            clip_queue_loop = self._clip_image_dispatch_loop
            if clip_queue_loop is not None:
                shutdown_future = asyncio.run_coroutine_threadsafe(
                    self._shutdown_clip_image_queue_async(),
                    clip_queue_loop,
                )
                try:
                    pending_tasks = shutdown_future.result(timeout=join_timeout_seconds)
                except Exception as exc:
                    LOG.warning("Failed to drain CLIP image queue during runtime recycle: %s", exc)
                try:
                    clip_queue_loop.call_soon_threadsafe(clip_queue_loop.stop)
                except RuntimeError:
                    pass

            clip_worker = self._clip_image_worker
            if clip_worker is not None and clip_worker is not threading.current_thread():
                clip_worker.join(timeout=join_timeout_seconds)
                if clip_worker.is_alive():
                    LOG.warning(
                        "CLIP queue worker did not exit within %.1fs during runtime recycle.",
                        join_timeout_seconds,
                    )
                else:
                    self._clip_image_worker = None
            self._clip_image_loop_ready.clear()
        return pending_tasks

    def _start_background_services(self) -> None:
        self._start_background_prewarm()
        self._start_idle_release_monitor()

    def _log_ready(self) -> None:
        LOG.info(
            "AIModels ready: pid=%s clip_device=%s clip_context=%s cache=%s image_budget=%s clip_queue=%s queue_timeout=%ss exec_timeout=%ss ocr_exec_timeout=%ss clip_batch=%s/%sms ocr_prewarm=%s ocr_idle_release=%ss ocr_admission=%s face_lane=%s face_preprocess_workers=%s face_batch=%s/%sms face_admission=%s",
            self._pid,
            self._clip_inference_device,
            self._clip_remote_context_device_name or "disabled",
            self.ov_cache_dir or "default",
            self._image_admission.capacity,
            self._queue_capacity,
            self._queue_timeout_seconds,
            self._execution_timeout_seconds,
            self._ocr_execution_timeout_seconds,
            self._clip_image_batch_size,
            int(self._clip_image_batch_wait_seconds * 1000.0),
            self._ocr_prewarm_enabled,
            int(self._idle_release_timeout_seconds),
            self._ocr_admission.capacity,
            INSIGHTFACE_SINGLE_LANE,
            self._face_preprocess_worker_count,
            self._face_batch_size,
            int(self._face_batch_wait_seconds * 1000.0),
            self._face_admission.capacity,
        )

    def _configure_openvino_cache(self) -> None:
        core = getattr(self, "core", None)
        if core is None or self.ov_cache_dir is None:
            return
        try:
            core.set_property({"CACHE_DIR": str(self.ov_cache_dir)})
        except Exception as exc:
            LOG.warning("Failed to set global OpenVINO cache dir: %s", exc)

    def _acquire_image_request_slot(self, label: str) -> None:
        if self._image_admission.acquire(timeout=0.0):
            return
        raise RuntimeError(
            f"{label} 图片请求总量已满（上限 {self._image_admission.capacity}），请稍后重试"
        )

    def _release_image_request_slot(self) -> None:
        self._image_admission.release()

    def _start_background_prewarm(self) -> None:
        if not self._ocr_prewarm_enabled:
            return
        self._background_prewarm_cancel.clear()
        self._background_prewarm_thread = threading.Thread(
            target=self._background_prewarm_loop,
            name="ai-ocr-prewarm",
            daemon=True,
        )
        self._background_prewarm_thread.start()

    def _start_idle_release_monitor(self) -> None:
        if self._idle_release_timeout_seconds <= 0:
            return
        self._idle_release_stop.clear()
        self._idle_release_wakeup.clear()
        self._idle_release_thread = threading.Thread(
            target=self._idle_release_loop,
            name="ai-idle-release",
            daemon=True,
        )
        self._idle_release_thread.start()

    def mark_request_activity(self) -> None:
        with self._request_activity_lock:
            self._last_request_activity_monotonic = time.monotonic()
        self._idle_release_wakeup.set()

    def _snapshot_last_request_activity_monotonic(self) -> float:
        with self._request_activity_lock:
            return self._last_request_activity_monotonic

    def _idle_release_loop(self) -> None:
        poll_seconds = min(5.0, max(0.5, self._idle_release_timeout_seconds / 6.0))
        while not self._idle_release_stop.is_set():
            deadline = (
                self._snapshot_last_request_activity_monotonic()
                + self._idle_release_timeout_seconds
            )
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._idle_release_wakeup.wait(timeout=min(poll_seconds, remaining))
                self._idle_release_wakeup.clear()
                continue

            try:
                loaded_families = self.get_loaded_runtime_families()
                if loaded_families:
                    LOG.info(
                        "No business request for %.1fs; releasing runtime model families: %s.",
                        self._idle_release_timeout_seconds,
                        ",".join(loaded_families),
                    )
                    self._release_non_text_models_sync(reason="idle-timeout")
            except Exception as exc:
                LOG.warning("Idle runtime model release failed: %s", exc, exc_info=True)
            finally:
                with self._request_activity_lock:
                    self._last_request_activity_monotonic = time.monotonic()

    def _join_idle_release_thread(self, timeout_seconds: Optional[float]) -> None:
        thread = self._idle_release_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            LOG.warning(
                "Idle-release thread did not exit within %.1fs.",
                float(timeout_seconds or 0.0),
            )
            return
        self._idle_release_thread = None

    def _background_prewarm_loop(self) -> None:
        delay_seconds = self._ocr_prewarm_delay_seconds
        if bool(self._background_prewarm_cancel.wait(timeout=delay_seconds)):
            return
        if self._stopping or self._background_prewarm_cancel.is_set():
            return
        try:
            started_at = time.monotonic()
            self._prewarm_ocr_family()
            if self._stopping or self._background_prewarm_cancel.is_set():
                LOG.info("RapidOCR background prewarm cancelled before OCR stayed resident.")
                return
            self._release_non_text_models_sync(
                reason="ocr-prewarm",
                cancel_background_prewarm=False,
                join_background_prewarm=False,
            )
            LOG.info(
                "RapidOCR background prewarm completed in %.2fs and released OCR family.",
                time.monotonic() - started_at,
            )
        except Exception as exc:
            LOG.warning("RapidOCR background prewarm failed: %s", exc, exc_info=True)

    def _prewarm_ocr_family(self) -> None:
        if self._stopping or self._background_prewarm_cancel.is_set():
            return
        leased = self._acquire_non_text_family_lease(
            "ocr",
            abort_event=self._background_prewarm_cancel,
        )
        if not leased:
            return
        try:
            if self._stopping or self._background_prewarm_cancel.is_set():
                return
            self._ensure_rapidocr_loaded()
            with self._rapidocr_load_lock:
                if self._stopping or self._background_prewarm_cancel.is_set():
                    return
                self._warmup_rapidocr_locked()
        finally:
            self._release_non_text_family_lease("ocr")

    def _init_clip_remote_context(self) -> Optional[Any]:
        core = getattr(self, "core", None)
        if core is None:
            raise RuntimeError("OpenVINO runtime is not initialized.")
        clip_device = self._clip_inference_device.strip().upper()
        force_gpu_remote_context = clip_device == "AUTO"
        wants_gpu_remote_context = force_gpu_remote_context or ("GPU" in clip_device)
        if not wants_gpu_remote_context:
            return None

        available_devices = _normalize_openvino_devices(core.available_devices)
        gpu_devices = _get_openvino_gpu_devices(available_devices)
        runtime_hint = (
            " Ensure the container exposes a real Intel /dev/dri render node and installs "
            "OpenVINO/OpenCL runtime packages (Debian 13 stable baseline: libze1, "
            "ocl-icd-libopencl1, mesa-opencl-icd; optional diagnostics such as clinfo "
            "are not bundled in the runtime image by default)."
        )
        if not gpu_devices:
            LOG.info(
                "OpenVINO available_devices does not include GPU for CLIP request=%s "
                "(available_devices=%s). Continue probing explicit GPU remote context APIs.",
                self._clip_inference_device,
                sorted(available_devices),
            )

        explicit_gpu_devices = _extract_explicit_gpu_devices(clip_device)
        context_candidates: List[str] = []
        for candidate in [*explicit_gpu_devices, "GPU", *gpu_devices]:
            if candidate not in context_candidates:
                context_candidates.append(candidate)

        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None

        for candidate in context_candidates:
            try:
                remote_context = core.get_default_context(candidate)
                resolved_device = str(remote_context.get_device_name()).strip().upper()
                if explicit_gpu_devices and resolved_device not in explicit_gpu_devices:
                    raise RuntimeError(
                        f"resolved device {resolved_device} does not match requested "
                        f"{explicit_gpu_devices}"
                    )
                self._clip_remote_context_device_name = resolved_device
                return remote_context
            except Exception as exc:
                last_exc = exc
                attempt_errors.append(
                    f"get_default_context({candidate}) failed: {_summarize_exception(exc)}"
                )

        try:
            remote_context = core.create_context("GPU", {})
            resolved_device = str(remote_context.get_device_name()).strip().upper()
            if explicit_gpu_devices and resolved_device not in explicit_gpu_devices:
                raise RuntimeError(
                    f"create_context(GPU) resolved to {resolved_device}, "
                    f"expected one of {explicit_gpu_devices}"
                )
            self._clip_remote_context_device_name = resolved_device
            return remote_context
        except Exception as exc:
            last_exc = exc
            attempt_errors.append(
                f"create_context(GPU) failed: {_summarize_exception(exc)}"
            )

        LOG.error(
            "CLIP GPU remote context initialization failed: request=%s available_devices=%s attempts=%s",
            self._clip_inference_device,
            sorted(available_devices),
            " | ".join(attempt_errors),
        )
        if force_gpu_remote_context:
            raise RuntimeError(
                "CLIP_INFERENCE_DEVICE=AUTO requires GPU Remote Context. "
                "OpenVINO GPU context initialization failed. "
                f"available_devices={sorted(available_devices)}.{runtime_hint}"
            ) from last_exc
        raise RuntimeError(
            f"CLIP_INFERENCE_DEVICE={self._clip_inference_device} requests GPU execution, "
            "but GPU Remote Context initialization failed. "
            f"No silent fallback is allowed. available_devices={sorted(available_devices)}."
            f"{runtime_hint}"
        ) from last_exc

    async def _enqueue_clip_image_task_async(self, task: _ClipImageTask) -> None:
        if self._stopping:
            raise RuntimeError("模型服务已关闭")
        queue = self._clip_image_queue
        if queue is None:
            raise RuntimeError("CLIP image queue is not initialized.")
        if queue.full():
            raise RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试")
        queue.put_nowait(task)

    def _submit_clip_image_task(self, payload: Any) -> _ClipImageTask:
        future: Future[Any] = Future()
        task = _ClipImageTask(payload=payload, future=future, created_at=time.time())
        if self._stopping:
            self._set_clip_task_exception(task, RuntimeError("模型服务已关闭"))
            return task
        self._ensure_clip_image_worker_ready()
        loop = self._clip_image_dispatch_loop
        if loop is None:
            self._set_clip_task_exception(
                task,
                RuntimeError("CLIP image queue loop is not initialized."),
            )
            return task
        submit_future = asyncio.run_coroutine_threadsafe(
            self._enqueue_clip_image_task_async(task),
            loop,
        )
        try:
            submit_future.result(timeout=max(1.0, float(self._queue_timeout_seconds)))
        except Exception as exc:
            submit_exc = exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc))
            self._set_clip_task_exception(task, submit_exc)
        return task

    def _cancel_clip_image_task_if_queued(self, task: _ClipImageTask, exc: Exception) -> bool:
        if task.started_event.is_set():
            return False
        task.cancel_requested.set()
        self._set_clip_task_exception(task, exc)
        return True

    def _wait_clip_image_task(self, task: _ClipImageTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = bool(task.started_event.wait(timeout=self._queue_timeout_seconds))
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_clip_image_task_if_queued(task, queue_exc):
                raise queue_exc
            if not bool(task.started_event.wait(timeout=0.05)) and not task.future.done():
                raise queue_exc

        try:
            return task.future.result(timeout=self._execution_timeout_seconds)
        except FutureTimeoutError as exc:
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    async def _await_clip_image_task(self, task: _ClipImageTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = bool(
            await asyncio.to_thread(task.started_event.wait, self._queue_timeout_seconds)
        )
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_clip_image_task_if_queued(task, queue_exc):
                raise queue_exc
            started = bool(await asyncio.to_thread(task.started_event.wait, 0.05))
            if not started and not task.future.done():
                raise queue_exc

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(task.future),
                timeout=self._execution_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    def _safe_set_result(self, future: Future, value: Any) -> None:
        if not future.done():
            future.set_result(value)

    def _safe_set_exception(self, future: Future, exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

    @contextmanager
    def _non_text_request_scope(
        self,
        *,
        family: NonTextFamily,
        label: str,
        admission: _AdmissionController,
        ensure_loaded: Callable[[], None],
    ) -> Iterator[None]:
        lease = _ManagedLease(label)
        self._acquire_image_request_slot(label)
        lease.push(self._release_image_request_slot)
        try:
            self._acquire_non_text_family_lease(family)
            lease.push(lambda: self._release_non_text_family_lease(family))
            ensure_loaded()
            self._acquire_admission(admission, label)
            lease.push(admission.release)
            yield
        finally:
            lease.release()

    @asynccontextmanager
    async def _non_text_request_scope_async(
        self,
        *,
        family: NonTextFamily,
        label: str,
        admission: _AdmissionController,
        ensure_loaded: Callable[[], None],
    ) -> AsyncIterator[None]:
        lease = _ManagedLease(label)
        self._acquire_image_request_slot(label)
        lease.push(self._release_image_request_slot)
        try:
            await self._acquire_non_text_family_lease_async(family)
            lease.push(lambda: self._release_non_text_family_lease(family))
            await self._run_control(ensure_loaded)
            await self._acquire_admission_async(admission, label)
            lease.push(admission.release)
            yield
        finally:
            await lease.release_async()

    def _get_loaded_runtime_families_locked(self) -> List[NonTextFamily]:
        loaded: List[NonTextFamily] = []
        if (
            self._clip_vision_model is not None
            or self._clip_vision_request is not None
            or self._clip_vision_ppp is not None
        ):
            loaded.append("vision")
        if (
            self._rapidocr_engine is not None
            or self._rapidocr_engines is not None
            or self._rapidocr_engine_pool is not None
            or self._rapidocr_runtime_cfg is not None
        ):
            loaded.append("ocr")
        if (
            self._face_engine is not None
        ):
            loaded.append("face")
        return loaded

    def get_loaded_runtime_families(self) -> List[NonTextFamily]:
        with self._clip_vision_load_lock, self._rapidocr_load_lock, self._face_load_lock:
            return list(self._get_loaded_runtime_families_locked())

    def _join_background_prewarm_thread(self, timeout_seconds: Optional[float]) -> None:
        thread = self._background_prewarm_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            LOG.warning(
                "RapidOCR background prewarm thread did not exit within %.1fs.",
                float(timeout_seconds or 0.0),
            )
            return
        self._background_prewarm_thread = None

    def _acquire_non_text_family_lease(
        self,
        family: NonTextFamily,
        abort_event: Optional[threading.Event] = None,
    ) -> bool:
        return self._non_text_state.acquire(family, abort_event=abort_event)

    async def _acquire_non_text_family_lease_async(self, family: NonTextFamily) -> bool:
        return await self._run_control(self._non_text_state.acquire, family)

    def _release_non_text_family_lease(self, family: NonTextFamily) -> None:
        self._non_text_state.release(family)

    def _unload_non_text_models_locked(
        self,
        keep_family: Optional[str] = None,
    ) -> List[str]:
        unloaded: List[str] = []
        if keep_family != "vision" and (
            self._clip_vision_model is not None
            or self._clip_vision_request is not None
            or self._clip_vision_ppp is not None
        ):
            self._unload_clip_vision_model_locked()
            unloaded.append("vision")
        if keep_family != "ocr" and (
            self._rapidocr_engine is not None
            or self._rapidocr_engines is not None
            or self._rapidocr_engine_pool is not None
            or self._rapidocr_runtime_cfg is not None
        ):
            self._unload_rapidocr_model_locked()
            unloaded.append("ocr")
        if keep_family != "face" and (
            self._face_engine is not None
        ):
            self._unload_face_model_locked()
            unloaded.append("face")
        return unloaded

    def _unload_non_text_models(
        self,
        keep_family: Optional[str] = None,
    ) -> List[str]:
        released_openvino_runtime = False
        with self._clip_vision_load_lock, self._rapidocr_load_lock, self._face_load_lock:
            unloaded = self._unload_non_text_models_locked(keep_family=keep_family)
            released_openvino_runtime = self._release_openvino_runtime_if_unused_locked(
                keep_family=keep_family
            )
        if unloaded or released_openvino_runtime:
            gc.collect()
            self._trim_native_memory(reason="non-text-release")
        return unloaded

    def _release_non_text_models_sync(
        self,
        reason: str,
        *,
        cancel_background_prewarm: bool = True,
        join_background_prewarm: bool = True,
    ) -> None:
        join_timeout_seconds = max(2.0, float(self._execution_timeout_seconds))
        self._non_text_state.begin_release()

        try:
            if cancel_background_prewarm:
                self._background_prewarm_cancel.set()
            if join_background_prewarm:
                self._join_background_prewarm_thread(timeout_seconds=join_timeout_seconds)

            self._non_text_state.wait_for_drain()

            unloaded = self._unload_non_text_models()
            recycled_support = self._recycle_non_text_runtime_support_resources()
            if unloaded or recycled_support:
                self._drop_non_text_filesystem_page_cache()
            LOG.info(
                "Runtime model release complete: reason=%s unloaded=%s support_recycled=%s",
                reason,
                ",".join(unloaded) if unloaded else "none",
                recycled_support,
            )
        finally:
            self._non_text_state.finish_release()

    def _load_family_serialized(
        self, family: NonTextFamily, loader: Callable[[], None]
    ) -> None:
        started_at = time.monotonic()
        loader()
        elapsed = time.monotonic() - started_at
        if elapsed >= 0.25:
            LOG.info(
                "Loaded %s model family in %.2fs in pid=%s.",
                family,
                elapsed,
                self._pid,
            )

    def _unload_everything_locked(self) -> None:
        self._unload_clip_vision_model_locked()
        self._unload_rapidocr_model_locked()
        self._unload_face_model_locked()
        self._release_openvino_runtime_if_unused_locked(keep_family=None)
        gc.collect()

    def _release_openvino_runtime_if_unused_locked(
        self,
        keep_family: Optional[str],
    ) -> bool:
        if keep_family in {"vision", "face"}:
            return False
        if (
            self._clip_vision_model is not None
            or self._clip_vision_request is not None
            or self._clip_vision_ppp is not None
            or self._face_engine is not None
        ):
            return False

        released = False
        if getattr(self, "_clip_remote_context", None) is not None:
            self._clip_remote_context = None
            released = True
        if getattr(self, "_clip_remote_context_device_name", None) is not None:
            self._clip_remote_context_device_name = None
            released = True
        if getattr(self, "core", None) is not None:
            self.core = None
            released = True
        return released

    def _recycle_non_text_runtime_support_resources(self) -> bool:
        recycled = False
        pending_clip_tasks = self._stop_clip_image_worker()
        for task in pending_clip_tasks:
            self._set_clip_task_exception(task, RuntimeError("模型资源正在释放"))
        if pending_clip_tasks:
            recycled = True

        pending_face_tasks = self._stop_face_batch_service()
        for task in pending_face_tasks:
            self._set_face_task_exception(task, RuntimeError("模型资源正在释放"))
        if pending_face_tasks:
            recycled = True

        if self._shutdown_non_text_task_executors():
            recycled = True

        if recycled:
            gc.collect()
            self._trim_native_memory(reason="non-text-support-recycle")
        return recycled

    def _drop_non_text_filesystem_page_cache(self) -> None:
        insightface_root = getattr(self, "insightface_root", None)
        evicted_files, evicted_bytes = _drop_filesystem_page_cache(
            [
                getattr(self, "qa_clip_path", None),
                getattr(self, "rapidocr_model_dir_path", None),
                getattr(self, "insightface_model_root", None),
                (insightface_root / "_runtime_models") if insightface_root is not None else None,
                getattr(self, "ov_cache_dir", None),
            ]
        )
        if evicted_files <= 0:
            return
        LOG.info(
            "Dropped Linux filesystem page cache for non-text model assets: files=%s bytes=%s.",
            evicted_files,
            evicted_bytes,
        )

    def _trim_native_memory(self, reason: str) -> None:
        trimmed = _trim_process_memory()
        if trimmed:
            LOG.info("Returned native heap pages to OS after %s.", reason)

    def _build_openvino_preprocess_runner(
        self,
        runner_name: str,
        device_name: str,
        output_height: int,
        output_width: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> _OpenVinoPreprocessRunner:
        core = self._ensure_openvino_runtime()
        parameter = ov.opset13.parameter(
            ov.PartialShape([ov.Dimension.dynamic(), 3, int(output_height), int(output_width)]),
            ov.Type.f32,
            name=f"{runner_name}_input",
        )
        result = ov.opset13.result(parameter)
        result.set_friendly_name(f"{runner_name}_output")
        model_factory: Any = ov.Model
        preprocess_model = model_factory(
            [result],
            [parameter],
            f"{runner_name}_ppp",
        )
        ppp = ov.preprocess.PrePostProcessor(preprocess_model)
        ppp.input().tensor().set_shape([-1, -1, -1, 3]).set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")
        ).set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        ppp.input().preprocess().mean(mean_values)
        ppp.input().preprocess().scale(std_values)
        ppp.input().model().set_layout(ov.Layout("NCHW"))

        compiled = core.compile_model(
            ppp.build(),
            device_name,
            {
                "PERFORMANCE_HINT": "LATENCY",
            },
        )
        runner = _OpenVinoPreprocessRunner(
            compiled_model=compiled,
            input_port=compiled.input(0),
            output_port=compiled.output(0),
            runner_name=runner_name,
            input_height=int(output_height),
            input_width=int(output_width),
        )
        runner.validate()
        return runner

    @staticmethod
    def _run_in_executor(
        executor: ThreadPoolExecutor,
        func: Callable[..., Any],
        *args: Any,
    ) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(executor, func, *args)

    def _run_control(
        self,
        func: Callable[..., Any],
        *args: Any,
    ) -> asyncio.Future[Any]:
        return self._run_in_executor(self._control_executor, func, *args)

    async def _await_with_timeout_and_cooperative_cancel(
        self,
        awaitable: "asyncio.Future[Any] | asyncio.Task[Any]",
        *,
        cancel_event: threading.Event,
        timeout_seconds: float,
        task_name: str,
    ) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.shield(awaitable),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            cancel_event.set()
            try:
                await awaitable
            except _InferenceCancelled:
                pass
            except Exception as cancel_exc:
                LOG.warning("%s cancellation completed with error: %s", task_name, cancel_exc)
            raise RuntimeError(f"推理任务执行超时（>{timeout_seconds}s）") from exc
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                await awaitable
            except _InferenceCancelled:
                pass
            except Exception as cancel_exc:
                LOG.warning("%s cancellation completed with error: %s", task_name, cancel_exc)
            raise

    def _acquire_admission(self, admission: _AdmissionController, label: str) -> None:
        if admission.acquire(timeout=self._queue_timeout_seconds):
            return
        raise RuntimeError(f"{label} 推理任务排队超时（>{self._queue_timeout_seconds}s）")

    async def _acquire_admission_async(self, admission: _AdmissionController, label: str) -> None:
        acquired = await admission.acquire_async(timeout=self._queue_timeout_seconds)
        if acquired:
            return
        raise RuntimeError(f"{label} 推理任务排队超时（>{self._queue_timeout_seconds}s）")

    async def _shutdown_clip_image_queue_async(self) -> List[_ClipImageTask]:
        pending: List[_ClipImageTask] = []
        queue = self._clip_image_queue
        if queue is None:
            return pending
        while True:
            try:
                queued = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            queue.task_done()
            if queued is None:
                continue
            pending.append(queued)
        queue.put_nowait(None)
        return pending

    def release_models_for_restart(self) -> None:
        self._release_non_text_models_sync(reason="restart")

    def release_all_models(self) -> None:
        join_timeout_seconds = max(2.0, float(self._execution_timeout_seconds))
        try:
            self._idle_release_stop.set()
            self._idle_release_wakeup.set()
            self._join_idle_release_thread(timeout_seconds=join_timeout_seconds)
            self._background_prewarm_cancel.set()
            try:
                self._release_non_text_models_sync(
                    reason="shutdown",
                    cancel_background_prewarm=False,
                    join_background_prewarm=True,
                )
            except Exception as exc:
                LOG.warning("Failed to release non-text models during shutdown: %s", exc, exc_info=True)
            self._stopping = True

            clip_worker = self._clip_image_worker
            if clip_worker is not None and clip_worker is not threading.current_thread():
                clip_worker.join(timeout=join_timeout_seconds)
                if clip_worker.is_alive():
                    LOG.warning(
                        "CLIP queue worker did not exit within %.1fs.",
                        join_timeout_seconds,
                    )
                else:
                    self._clip_image_worker = None
            self._clip_image_loop_ready.clear()
            self._join_background_prewarm_thread(timeout_seconds=join_timeout_seconds)

            self._shutdown_non_text_task_executors()

            for executor in (self._control_executor,):
                if executor is not None:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception as exc:
                        LOG.warning("Executor shutdown failed: %s", exc, exc_info=True)
            self._control_executor = None
            self._face_preprocess_executor = None
            self._shared_cpu_executor = None
            self._ocr_executor = None

            try:
                with self._model_lock:
                    self._unload_everything_locked()
            except Exception as exc:
                LOG.warning("Final runtime unload failed during shutdown: %s", exc, exc_info=True)
            self._trim_native_memory(reason="shutdown-final")
            LOG.info("All models released.")
        finally:
            self._release_single_process_lock()
