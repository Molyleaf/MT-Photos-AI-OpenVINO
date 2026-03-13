import asyncio
import gc
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import openvino as ov

from .clip_image import ClipImageMixin
from .clip_text import ClipTextMixin
from .common import (
    _AdmissionController,
    _InferenceTask,
    _InferenceCancelled,
    _InterProcessFileLock,
    _OpenVinoPreprocessRunner,
    TaskType,
    _as_bool,
    _as_float,
    _as_int,
    _extract_explicit_gpu_devices,
    _get_openvino_gpu_devices,
    _normalize_openvino_devices,
    _summarize_exception,
)
from .constants import (
    APP_DIR,
    CLIP_INFERENCE_DEVICE,
    EXEC_TIMEOUT_SECONDS,
    LOG,
    MAX_PENDING_IMAGE_REQUESTS,
    PROJECT_ROOT,
    QUEUE_MAX_SIZE,
    QUEUE_TIMEOUT_SECONDS,
)
from .insightface import InsightFaceMixin
from .rapidocr_lib import RapidOCRMixin


class AIModels(ClipTextMixin, ClipImageMixin, RapidOCRMixin, InsightFaceMixin):
    """
    Text-CLIP stays resident as a single-threaded singleton service.
    Image-CLIP uses a dedicated batch queue after standardized preprocessing.
    Non-text families are lazy-loaded and switch synchronously so only one
    vision/OCR/face family stays resident at a time, with idle release.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self.model_base_path = Path(
            os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models"))
        )
        self.insightface_root = self.model_base_path / "insightface"
        self.insightface_model_root = self.insightface_root / "models"
        self.qa_clip_path = self.model_base_path / "qa-clip" / "openvino"
        self._clip_inference_device = CLIP_INFERENCE_DEVICE

        cache_dir_raw = str(os.environ.get("OV_CACHE_DIR", "")).strip()
        if cache_dir_raw:
            self.ov_cache_dir = Path(cache_dir_raw).expanduser().resolve()
        else:
            self.ov_cache_dir = (PROJECT_ROOT / "cache" / "openvino").resolve()
        self.ov_cache_dir.mkdir(parents=True, exist_ok=True)

        self.rapidocr_config_path = Path(
            os.environ.get(
                "RAPIDOCR_OPENVINO_CONFIG_PATH",
                str(APP_DIR / "config" / "cfg_openvino_cpu.yaml"),
            )
        )
        self.rapidocr_model_dir = os.environ.get(
            "RAPIDOCR_MODEL_DIR", str(self.model_base_path / "rapidocr")
        )
        self.rapidocr_model_dir_path = Path(self.rapidocr_model_dir).expanduser().resolve()
        self.rapidocr_font_path = os.environ.get("RAPIDOCR_FONT_PATH", "")

        self.core = ov.Core()
        self._configure_openvino_cache()

        self._clip_remote_context_device_name: Optional[str] = None
        self._clip_remote_context = self._init_clip_remote_context()

        self._model_lock = threading.Lock()
        self._clip_vision_load_lock = threading.Lock()
        self._rapidocr_load_lock = threading.Lock()
        self._face_load_lock = threading.Lock()

        self._clip_text_model: Optional[ov.CompiledModel] = None
        self._clip_text_request: Optional[ov.InferRequest] = None
        self._clip_text_input_names: Optional[tuple[str, str]] = None
        self._clip_text_host_input_cache: Dict[
            int, tuple[ov.Tensor, Any, ov.Tensor, Any]
        ] = {}
        self._clip_text_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_text_lock = threading.Lock()
        self._runtime_state_dir = (PROJECT_ROOT / "cache" / "runtime").resolve()
        self._runtime_state_dir.mkdir(parents=True, exist_ok=True)
        self._text_service_meta_path = self._runtime_state_dir / "text-clip-service.json"
        self._text_service_lock = _InterProcessFileLock(
            self._runtime_state_dir / "text-clip-service.lock"
        )
        self._text_service_owner = False
        self._text_service_server = None
        self._text_service_thread: Optional[threading.Thread] = None
        self._text_service_port: Optional[int] = None
        self._clip_vision_model: Optional[ov.CompiledModel] = None
        self._clip_vision_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._clip_vision_request: Optional[ov.InferRequest] = None
        self._clip_vision_input_name: Optional[str] = None
        self._clip_vision_host_input_cache: Dict[tuple[int, int, int], tuple[ov.Tensor, Any]] = {}
        self._clip_vision_host_tensor_enabled = self._clip_remote_context is not None
        self._clip_image_batch_size = max(
            1,
            _as_int(
                os.environ.get("CLIP_IMAGE_BATCH", os.environ.get("CLIP_IMAGE_BATCH_SIZE")),
                8,
            ),
        )
        self._clip_image_batch_wait_seconds = max(
            0.0,
            _as_float(os.environ.get("CLIP_IMAGE_BATCH_WAIT_MS"), 5.0) / 1000.0,
        )
        self._rapidocr_engine = None
        self._rapidocr_runtime_cfg: Optional[Dict[str, Any]] = None
        self._ocr_opencl_device_name: Optional[str] = None
        self._ocr_opencl_device_vendor: Optional[str] = None
        self._face_engine = None
        self._face_det_ppp: Optional[_OpenVinoPreprocessRunner] = None
        self._face_rec_ppp: Optional[_OpenVinoPreprocessRunner] = None
        configured_queue_capacity = max(1, QUEUE_MAX_SIZE)
        self._queue_capacity = min(MAX_PENDING_IMAGE_REQUESTS, configured_queue_capacity)
        if configured_queue_capacity > self._queue_capacity:
            LOG.warning(
                "INFERENCE_QUEUE_MAX_SIZE=%s exceeds MT-Photos safe limit %s; capping to %s.",
                configured_queue_capacity,
                MAX_PENDING_IMAGE_REQUESTS,
                self._queue_capacity,
            )
        self._queue_timeout_seconds = max(1, QUEUE_TIMEOUT_SECONDS)
        self._execution_timeout_seconds = max(1, EXEC_TIMEOUT_SECONDS)
        self._image_admission = _AdmissionController("image", self._queue_capacity)
        self._shared_cpu_executor = ThreadPoolExecutor(
            max_workers=max(2, min(8, os.cpu_count() or 4)),
            thread_name_prefix="ai-cpu",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ai-ctl",
        )
        self._ocr_stage_worker_count = max(
            1,
            _as_int(os.environ.get("RAPIDOCR_PERFORMANCE_NUM_REQUESTS"), 2),
        )
        self._ocr_det_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-det",
        )
        self._ocr_cls_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-cls",
        )
        self._ocr_rec_executor = ThreadPoolExecutor(
            max_workers=self._ocr_stage_worker_count,
            thread_name_prefix="ocr-rec",
        )
        self._face_worker_count = self._resolve_face_worker_count()
        self._face_executor = ThreadPoolExecutor(
            max_workers=self._face_worker_count,
            thread_name_prefix="face-ov",
        )
        self._family_load_locks = {
            family: _InterProcessFileLock(self._runtime_state_dir / f"{family}.load.lock")
            for family in ("vision", "ocr", "face")
        }

        self._condition = threading.Condition()
        self._normal_queue: Deque[_InferenceTask] = deque()
        self._ocr_execution_timeout_seconds = max(
            1,
            _as_int(
                os.environ.get("OCR_EXEC_TIMEOUT"),
                max(30, self._execution_timeout_seconds),
            ),
        )
        self._load_lock_timeout_seconds = max(30.0, float(self._execution_timeout_seconds))
        self._idle_release_timeout_seconds = max(
            0.0,
            _as_float(os.environ.get("NON_TEXT_IDLE_RELEASE_SECONDS"), 60.0),
        )
        self._request_activity_lock = threading.Lock()
        self._last_request_monotonic = time.monotonic()
        self._idle_release_stop = threading.Event()
        self._idle_release_wakeup = threading.Event()
        self._idle_release_thread: Optional[threading.Thread] = None
        self._ocr_admission = _AdmissionController(
            "ocr",
            self._resolve_ocr_request_capacity(self._ocr_stage_worker_count),
        )
        self._face_admission = _AdmissionController(
            "face",
            self._resolve_face_request_capacity(self._face_worker_count),
        )
        self._non_text_condition = threading.Condition()
        self._non_text_active_family: Optional[str] = None
        self._non_text_inflight: Dict[str, int] = {
            "vision": 0,
            "ocr": 0,
            "face": 0,
        }
        self._non_text_transition_in_progress = False
        self._stopping = False
        self._background_prewarm_cancel = threading.Event()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ai-model-worker",
            daemon=True,
        )
        self._worker.start()
        self._ensure_text_service_ready(preload=True)
        self._background_prewarm_thread: Optional[threading.Thread] = None
        self._start_background_prewarm()
        self._start_idle_release_monitor()

        LOG.info(
            "AIModels ready: clip_device=%s clip_context=%s cache=%s image_budget=%s clip_queue=%s queue_timeout=%ss exec_timeout=%ss ocr_exec_timeout=%ss clip_batch=%s/%sms text_service=%s ocr_prewarm=%s ocr_idle_release=%ss ocr_admission=%s face_workers=%s face_admission=%s",
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
            "owner" if self._text_service_owner else "client",
            _as_bool(os.environ.get("OCR_PREWARM_ENABLED"), False),
            int(self._idle_release_timeout_seconds),
            self._ocr_admission.capacity,
            self._face_worker_count,
            self._face_admission.capacity,
        )

    def _configure_openvino_cache(self) -> None:
        try:
            self.core.set_property({"CACHE_DIR": str(self.ov_cache_dir)})
        except Exception as exc:
            LOG.warning("Failed to set global OpenVINO cache dir: %s", exc)

    def _resolve_ocr_request_capacity(self, worker_count: int) -> int:
        default_capacity = max(2, int(worker_count) * 2)
        configured = _as_int(os.environ.get("OCR_MAX_CONCURRENT_REQUESTS"), default_capacity)
        return max(1, min(self._queue_capacity, configured))

    def _resolve_face_worker_count(self) -> int:
        configured = _as_int(os.environ.get("INSIGHTFACE_MAX_WORKERS"), 2)
        return max(1, min(self._queue_capacity, configured))

    def _resolve_face_request_capacity(self, worker_count: int) -> int:
        default_capacity = max(2, int(worker_count) * 2)
        configured = _as_int(
            os.environ.get("INSIGHTFACE_MAX_CONCURRENT_REQUESTS"),
            default_capacity,
        )
        return max(1, min(self._queue_capacity, configured))

    def _acquire_image_request_slot(self, label: str) -> None:
        if self._image_admission.acquire(timeout=0.0):
            return
        raise RuntimeError(
            f"{label} 图片请求总量已满（上限 {self._image_admission.capacity}），请稍后重试"
        )

    def _release_image_request_slot(self) -> None:
        self._image_admission.release()

    def _start_background_prewarm(self) -> None:
        if not _as_bool(os.environ.get("OCR_PREWARM_ENABLED"), False):
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
            self._last_request_monotonic = time.monotonic()
        self._idle_release_wakeup.set()

    def _snapshot_last_request_monotonic(self) -> float:
        with self._request_activity_lock:
            return self._last_request_monotonic

    def _idle_release_loop(self) -> None:
        poll_seconds = min(5.0, max(0.5, self._idle_release_timeout_seconds / 6.0))
        while not self._idle_release_stop.is_set():
            deadline = self._snapshot_last_request_monotonic() + self._idle_release_timeout_seconds
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._idle_release_wakeup.wait(timeout=min(poll_seconds, remaining))
                self._idle_release_wakeup.clear()
                continue

            try:
                if self._has_loaded_non_text_models():
                    LOG.info(
                        "No business request for %.1fs; releasing non-text models.",
                        self._idle_release_timeout_seconds,
                    )
                    self._release_non_text_models_sync(reason="idle-timeout")
            except Exception as exc:
                LOG.warning("Idle non-text model release failed: %s", exc, exc_info=True)
            finally:
                with self._request_activity_lock:
                    self._last_request_monotonic = time.monotonic()

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
        delay_seconds = max(0.0, _as_float(os.environ.get("OCR_PREWARM_DELAY_SECONDS"), 1.0))
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
        clip_device = self._clip_inference_device.strip().upper()
        force_gpu_remote_context = clip_device == "AUTO"
        wants_gpu_remote_context = force_gpu_remote_context or ("GPU" in clip_device)
        if not wants_gpu_remote_context:
            return None

        available_devices = _normalize_openvino_devices(self.core.available_devices)
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
                remote_context = self.core.get_default_context(candidate)
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
            remote_context = self.core.create_context("GPU", {})
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

    def _queue_size_locked(self) -> int:
        return len(self._normal_queue)

    def _submit_task(self, kind: TaskType, payload: Any) -> _InferenceTask:
        future: Future[Any] = Future()
        task = _InferenceTask(kind=kind, payload=payload, future=future, created_at=time.time())
        with self._condition:
            if self._stopping:
                future.set_exception(RuntimeError("模型服务已关闭"))
                return task
            if self._queue_size_locked() >= self._queue_capacity:
                future.set_exception(
                    RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试")
                )
                return task
            self._normal_queue.append(task)
            self._condition.notify()
        return task

    def _cancel_task_if_queued(self, task: _InferenceTask, exc: Exception) -> bool:
        with self._condition:
            if task.started_event.is_set():
                return False
            try:
                self._normal_queue.remove(task)
            except ValueError:
                return False
        self._safe_set_exception(task.future, exc)
        return True

    def _wait_task(self, task: _InferenceTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = bool(task.started_event.wait(timeout=self._queue_timeout_seconds))
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_task_if_queued(task, queue_exc):
                raise queue_exc
            if not bool(task.started_event.wait(timeout=0.05)) and not task.future.done():
                raise queue_exc

        try:
            return task.future.result(timeout=self._execution_timeout_seconds)
        except FutureTimeoutError as exc:
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    async def _await_task(self, task: _InferenceTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = bool(
            await asyncio.to_thread(task.started_event.wait, self._queue_timeout_seconds)
        )
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_task_if_queued(task, queue_exc):
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

    @staticmethod
    def _log_detached_async_task_failure(task: "asyncio.Task[Any]", task_name: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            LOG.error("%s failed after caller detached: %s", task_name, exc, exc_info=exc)

    def _bind_non_text_lease_to_future(
        self,
        family: str,
        future: Future[Any] | asyncio.Future[Any],
    ) -> None:
        future.add_done_callback(lambda _future: self._release_non_text_family_lease(family))

    def _bind_image_request_slot_to_future(
        self,
        future: Future[Any] | asyncio.Future[Any],
    ) -> None:
        future.add_done_callback(lambda _future: self._release_image_request_slot())

    def _has_loaded_non_text_models_locked(self) -> bool:
        return bool(
            self._clip_vision_model is not None
            or self._clip_vision_request is not None
            or self._clip_vision_ppp is not None
            or self._rapidocr_engine is not None
            or self._face_engine is not None
            or self._face_det_ppp is not None
            or self._face_rec_ppp is not None
        )

    def _has_loaded_non_text_models(self) -> bool:
        with self._clip_vision_load_lock, self._rapidocr_load_lock, self._face_load_lock:
            return self._has_loaded_non_text_models_locked()

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
        family: str,
        abort_event: Optional[threading.Event] = None,
    ) -> bool | None:
        while True:
            previous_family: Optional[str]
            with self._non_text_condition:
                while self._non_text_transition_in_progress:
                    if abort_event is not None and abort_event.is_set():
                        return False
                    if self._stopping:
                        raise RuntimeError("模型服务已关闭")
                    self._non_text_condition.wait(
                        timeout=0.1 if abort_event is not None else None
                    )
                if abort_event is not None and abort_event.is_set():
                    return False
                if self._stopping:
                    raise RuntimeError("模型服务已关闭")

                previous_family = self._non_text_active_family
                if previous_family in (None, family):
                    self._non_text_active_family = family
                    self._non_text_inflight[family] += 1
                    return True

                if self._non_text_inflight[previous_family] > 0:
                    if abort_event is not None and abort_event.is_set():
                        return False
                    self._non_text_condition.wait(
                        timeout=0.1 if abort_event is not None else None
                    )
                    continue

                self._non_text_transition_in_progress = True

            switch_exc: Optional[Exception] = None
            unloaded_families: List[str] = []
            try:
                unloaded_families = self._unload_non_text_models(keep_family=family)
            except Exception as exc:
                switch_exc = exc

            aborted = abort_event is not None and abort_event.is_set()
            with self._non_text_condition:
                if switch_exc is None and not aborted:
                    self._non_text_active_family = family
                    self._non_text_inflight[family] += 1
                self._non_text_transition_in_progress = False
                self._non_text_condition.notify_all()

            if switch_exc is not None:
                raise switch_exc
            if aborted:
                return False
            if unloaded_families:
                LOG.info(
                    "Switched non-text family from %s to %s; released=%s",
                    previous_family,
                    family,
                    ",".join(unloaded_families),
                )
            return True

    async def _acquire_non_text_family_lease_async(self, family: str) -> bool:
        while True:
            previous_family: Optional[str]
            need_wait = False
            with self._non_text_condition:
                if self._stopping:
                    raise RuntimeError("模型服务已关闭")
                if self._non_text_transition_in_progress:
                    need_wait = True
                else:
                    previous_family = self._non_text_active_family
                    if previous_family in (None, family):
                        self._non_text_active_family = family
                        self._non_text_inflight[family] += 1
                        return True
                    elif self._non_text_inflight[previous_family] > 0:
                        need_wait = True
                    else:
                        self._non_text_transition_in_progress = True
                        need_wait = False

            if need_wait:
                await asyncio.sleep(0.05)
                continue

            switch_exc: Optional[Exception] = None
            unloaded_families: List[str] = []
            try:
                unloaded_families = await self._run_control(
                    partial(self._unload_non_text_models, keep_family=family)
                )
            except Exception as exc:
                switch_exc = exc

            with self._non_text_condition:
                if switch_exc is None:
                    self._non_text_active_family = family
                    self._non_text_inflight[family] += 1
                self._non_text_transition_in_progress = False
                self._non_text_condition.notify_all()

            if switch_exc is not None:
                raise switch_exc
            if unloaded_families:
                LOG.info(
                    "Switched non-text family from %s to %s; released=%s",
                    previous_family,
                    family,
                    ",".join(unloaded_families),
                )
            return True

    def _release_non_text_family_lease(self, family: str) -> None:
        with self._non_text_condition:
            inflight = self._non_text_inflight[family]
            if inflight <= 0:
                return
            self._non_text_inflight[family] = inflight - 1
            if self._non_text_inflight[family] == 0:
                self._non_text_condition.notify_all()

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
        if keep_family != "ocr" and self._rapidocr_engine is not None:
            self._unload_rapidocr_model_locked()
            unloaded.append("ocr")
        if keep_family != "face" and (
            self._face_engine is not None
            or self._face_det_ppp is not None
            or self._face_rec_ppp is not None
        ):
            self._unload_face_model_locked()
            unloaded.append("face")
        return unloaded

    def _unload_non_text_models(
        self,
        keep_family: Optional[str] = None,
    ) -> List[str]:
        with self._clip_vision_load_lock, self._rapidocr_load_lock, self._face_load_lock:
            unloaded = self._unload_non_text_models_locked(keep_family=keep_family)
        if unloaded:
            gc.collect()
        return unloaded

    def _release_non_text_models_sync(
        self,
        reason: str,
        *,
        cancel_background_prewarm: bool = True,
        join_background_prewarm: bool = True,
    ) -> None:
        join_timeout_seconds = max(2.0, float(self._execution_timeout_seconds))
        with self._non_text_condition:
            while self._non_text_transition_in_progress:
                if self._stopping:
                    return
                self._non_text_condition.wait()
            self._non_text_transition_in_progress = True

        try:
            if cancel_background_prewarm:
                self._background_prewarm_cancel.set()
            if join_background_prewarm:
                self._join_background_prewarm_thread(timeout_seconds=join_timeout_seconds)

            with self._non_text_condition:
                while any(self._non_text_inflight.values()):
                    self._non_text_condition.wait()

            unloaded = self._unload_non_text_models()
            LOG.info(
                "Non-text model release complete: reason=%s unloaded=%s",
                reason,
                ",".join(unloaded) if unloaded else "none",
            )
        finally:
            with self._non_text_condition:
                self._non_text_active_family = None
                self._non_text_transition_in_progress = False
                self._non_text_condition.notify_all()

    def _load_family_with_process_lock(
        self, family: str, loader: Callable[[], None]
    ) -> None:
        lock = self._family_load_locks[family]
        started_at = time.monotonic()
        acquired = lock.acquire(timeout=self._load_lock_timeout_seconds, blocking=True)
        if not acquired:
            raise RuntimeError(
                f"{family} 模型加载等待跨进程锁超时（>{self._load_lock_timeout_seconds:.0f}s）"
            )
        try:
            waited = time.monotonic() - started_at
            if waited >= 0.25:
                LOG.info(
                    "Waited %.2fs for %s model load lock in pid=%s.",
                    waited,
                    family,
                    self._pid,
                )
            loader()
        finally:
            lock.release()

    def _unload_text_model_locked(self) -> None:
        self._clip_text_request = None
        self._clip_text_model = None
        self._clip_text_input_names = None
        self._clip_text_host_input_cache.clear()
        self._text_service_port = None
        gc.collect()

    def _unload_everything_locked(self) -> None:
        self._unload_text_model_locked()
        self._unload_clip_vision_model_locked()
        self._unload_rapidocr_model_locked()
        self._unload_face_model_locked()
        gc.collect()

    def _build_openvino_preprocess_runner(
        self,
        runner_name: str,
        device_name: str,
        output_height: int,
        output_width: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> _OpenVinoPreprocessRunner:
        parameter = ov.opset13.parameter(
            ov.PartialShape([ov.Dimension.dynamic(), 3, int(output_height), int(output_width)]),
            ov.Type.f32,
            name=f"{runner_name}_input",
        )
        result = ov.opset13.result(parameter)
        result.set_friendly_name(f"{runner_name}_output")
        preprocess_model = ov.Model(
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

        compiled = self.core.compile_model(
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

    def release_models(self) -> None:
        self._release_non_text_models_sync(reason="manual")

    def release_models_for_restart(self, text_restore_delay_seconds: Optional[float] = None) -> None:
        _ = text_restore_delay_seconds
        self._release_non_text_models_sync(reason="restart")

    def release_all_models(self) -> None:
        self._idle_release_stop.set()
        self._idle_release_wakeup.set()
        self._join_idle_release_thread(
            timeout_seconds=max(2.0, float(self._execution_timeout_seconds))
        )
        self._background_prewarm_cancel.set()
        self._release_non_text_models_sync(
            reason="shutdown",
            cancel_background_prewarm=False,
            join_background_prewarm=True,
        )

        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            pending_tasks = list(self._normal_queue)
            self._normal_queue.clear()
            self._condition.notify_all()

        for task in pending_tasks:
            self._safe_set_exception(task.future, RuntimeError("模型服务正在关闭"))

        self._worker.join()
        self._join_background_prewarm_thread(
            timeout_seconds=max(2.0, float(self._execution_timeout_seconds))
        )

        for executor in (
            self._control_executor,
            self._shared_cpu_executor,
            self._ocr_det_executor,
            self._ocr_cls_executor,
            self._ocr_rec_executor,
            self._face_executor,
        ):
            executor.shutdown(wait=True, cancel_futures=True)

        self._shutdown_text_service()
        with self._model_lock:
            self._unload_everything_locked()
        LOG.info("All models released.")
