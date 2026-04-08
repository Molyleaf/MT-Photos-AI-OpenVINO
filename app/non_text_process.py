import asyncio
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import numpy as np

from models.common import _get_process_memory_snapshot
from models.runtime_settings import load_execution_control_settings
from models.schemas import OCRResult, RepresentResult

LOG = logging.getLogger("mt_photos_ai.models")
_NonTextFamily = str


def _serialize_worker_result(operation: str, result: Any) -> Any:
    if operation == "ocr":
        return result.model_dump()
    if operation == "represent":
        return [item.model_dump() for item in result]
    return list(result)


def _format_memory_snapshot(snapshot: Dict[str, int]) -> str:
    ordered_keys = [
        "VmRSS",
        "RssAnon",
        "RssFile",
        "RssShmem",
        "VmSize",
        "cgroup_memory_current",
        "cgroup_anon",
        "cgroup_file",
        "cgroup_shmem",
        "cgroup_inactive_file",
        "cgroup_active_file",
        "WorkingSetSize",
        "PrivateUsage",
    ]
    return ", ".join(f"{key}={snapshot[key]}" for key in ordered_keys if key in snapshot)


def _log_current_process_memory(reason: str) -> None:
    snapshot = _get_process_memory_snapshot()
    if not snapshot:
        return
    formatted = _format_memory_snapshot(snapshot)
    if formatted:
        LOG.info("Process memory snapshot after %s: %s", reason, formatted)


def _non_text_worker_entry(request_queue: Any, response_queue: Any) -> None:
    # The parent process owns non-text lifecycle. The child only serves
    # requests and exits on every explicit release path so its anonymous memory
    # is reclaimed by process teardown.
    os.environ["NON_TEXT_IDLE_RELEASE_SECONDS"] = "0"
    os.environ["OCR_PREWARM_ENABLED"] = "false"

    from models.runtime import AIModels

    runtime: Optional[AIModels] = None
    executor: Optional[ThreadPoolExecutor] = None
    try:
        runtime = AIModels()
        executor = ThreadPoolExecutor(
            max_workers=max(2, int(getattr(runtime, "_queue_capacity", 2))),
            thread_name_prefix="non-text-rpc",
        )
        response_queue.put({"kind": "ready", "pid": os.getpid()})

        def handle_infer(message: Dict[str, Any]) -> None:
            operation = str(message["operation"])
            request_id = str(message["request_id"])
            payload = message["payload"]
            try:
                if operation == "clip_img":
                    result = runtime.get_image_embedding(payload)
                elif operation == "ocr":
                    result = runtime.get_ocr_results(payload)
                elif operation == "represent":
                    result = runtime.get_face_representation(payload)
                else:
                    raise RuntimeError(f"Unsupported non-text operation: {operation}")
                response_queue.put(
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "operation": operation,
                        "result": _serialize_worker_result(operation, result),
                    }
                )
            except Exception as exc:
                response_queue.put(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error": str(exc),
                    }
                )
            finally:
                request_queue.task_done()

        while True:
            message = request_queue.get()
            kind = str(message.get("kind", ""))
            if kind == "shutdown":
                request_queue.task_done()
                break
            if kind != "infer":
                request_queue.task_done()
                response_queue.put(
                    {
                        "kind": "error",
                        "request_id": str(message.get("request_id", "")),
                        "error": f"Unsupported worker message kind: {kind}",
                    }
                )
                continue
            executor.submit(handle_infer, message)
    except Exception as exc:
        response_queue.put({"kind": "startup-error", "error": str(exc)})
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if runtime is not None:
            try:
                runtime.release_all_models()
            except Exception as release_exc:
                LOG.warning(
                    "Non-text worker failed to release models before exit: %s",
                    release_exc,
                    exc_info=True,
                )
        response_queue.put({"kind": "stopped", "pid": os.getpid()})


class NonTextProcessManager:
    def __init__(
        self,
        *,
        mp_context: Any = None,
        worker_target: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        settings = load_execution_control_settings()
        self._queue_capacity = settings.queue_capacity
        self._queue_timeout_seconds = settings.queue_timeout_seconds
        self._execution_timeout_seconds = settings.execution_timeout_seconds
        self._ocr_execution_timeout_seconds = settings.ocr_execution_timeout_seconds
        self._idle_release_timeout_seconds = settings.idle_release_timeout_seconds

        self._mp_context = mp_context or mp.get_context("spawn")
        self._worker_target = worker_target or _non_text_worker_entry

        self._process_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._family_condition = threading.Condition()
        self._request_activity_lock = threading.Lock()

        self._worker_process: Any = None
        self._request_queue: Any = None
        self._response_queue: Any = None
        self._response_thread: Optional[threading.Thread] = None
        self._response_thread_stop: Optional[threading.Event] = None
        self._worker_pid: Optional[int] = None
        self._worker_family: Optional[_NonTextFamily] = None

        self._pending: Dict[str, Future[Any]] = {}
        self._active_family: Optional[_NonTextFamily] = None
        self._switching = False
        self._inflight = {
            "vision": 0,
            "ocr": 0,
            "face": 0,
        }

        self._stopping = False
        self._last_request_activity_monotonic = time.monotonic()
        self._idle_release_stop = threading.Event()
        self._idle_release_wakeup = threading.Event()
        self._idle_release_thread: Optional[threading.Thread] = None

        self._start_idle_release_monitor()
        LOG.info(
            "Non-text runtime manager ready: queue=%s queue_timeout=%ss exec_timeout=%ss ocr_exec_timeout=%ss idle_release=%ss start_method=%s",
            self._queue_capacity,
            self._queue_timeout_seconds,
            self._execution_timeout_seconds,
            self._ocr_execution_timeout_seconds,
            int(self._idle_release_timeout_seconds),
            self._mp_context.get_start_method(),
        )

    def mark_request_activity(self) -> None:
        with self._request_activity_lock:
            self._last_request_activity_monotonic = time.monotonic()
        self._idle_release_wakeup.set()

    def _snapshot_last_request_activity_monotonic(self) -> float:
        with self._request_activity_lock:
            return self._last_request_activity_monotonic

    def _start_idle_release_monitor(self) -> None:
        if self._idle_release_timeout_seconds <= 0:
            return
        self._idle_release_stop.clear()
        self._idle_release_wakeup.clear()
        self._idle_release_thread = threading.Thread(
            target=self._idle_release_loop,
            name="non-text-idle-release",
            daemon=True,
        )
        self._idle_release_thread.start()

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
                family = self.get_loaded_runtime_family()
                if family is not None:
                    LOG.info(
                        "No business request for %.1fs; releasing non-text worker family: %s.",
                        self._idle_release_timeout_seconds,
                        family,
                    )
                    self._release_worker_when_drained(reason="idle-timeout")
            except Exception as exc:
                LOG.warning("Idle non-text worker release failed: %s", exc, exc_info=True)
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

    def _set_future_result(self, request_id: str, value: Any) -> None:
        future = None
        with self._pending_lock:
            future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(value)

    def _set_future_exception(self, request_id: str, exc: Exception) -> None:
        future = None
        with self._pending_lock:
            future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_exception(exc)

    def _fail_all_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    @staticmethod
    def _deserialize_result(operation: str, payload: Any) -> Any:
        if operation == "ocr":
            return OCRResult.model_validate(payload)
        if operation == "represent":
            return [RepresentResult.model_validate(item) for item in payload]
        return list(payload)

    def _response_loop(
        self,
        *,
        process: Any,
        response_queue: Any,
        stop_event: threading.Event,
    ) -> None:
        while True:
            try:
                message = response_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set() and not process.is_alive():
                    break
                if not process.is_alive():
                    exc = RuntimeError(
                        "Non-text worker exited unexpectedly before responding."
                    )
                    self._fail_all_pending(exc)
                    with self._family_condition:
                        self._active_family = None
                        self._family_condition.notify_all()
                    with self._process_lock:
                        if self._worker_process is process:
                            self._stop_worker_locked("unexpected-exit")
                    break
                continue

            kind = str(message.get("kind", ""))
            if kind == "result":
                request_id = str(message["request_id"])
                operation = str(message["operation"])
                value = self._deserialize_result(operation, message["result"])
                self._set_future_result(request_id, value)
                continue
            if kind == "error":
                request_id = str(message.get("request_id", ""))
                error_text = str(message.get("error", "Unknown worker error"))
                self._set_future_exception(request_id, RuntimeError(error_text))
                continue
            if kind == "stopped":
                break

    @staticmethod
    def _close_queue_handle(handle: Any) -> None:
        if handle is None:
            return
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        join_thread = getattr(handle, "join_thread", None)
        if callable(join_thread):
            try:
                join_thread()
            except Exception:
                pass

    def _start_worker_locked(self, family: _NonTextFamily) -> None:
        if self._worker_process is not None and bool(self._worker_process.is_alive()):
            return
        if self._worker_process is not None:
            self._stop_worker_locked("stale-worker-cleanup")

        request_queue = self._mp_context.JoinableQueue(maxsize=self._queue_capacity)
        response_queue = self._mp_context.Queue()
        process = self._mp_context.Process(
            target=self._worker_target,
            args=(request_queue, response_queue),
            daemon=True,
        )
        process.start()

        startup_timeout = max(5.0, float(self._execution_timeout_seconds))
        try:
            startup_message = response_queue.get(timeout=startup_timeout)
        except queue.Empty as exc:
            process.terminate()
            process.join(timeout=2.0)
            self._close_queue_handle(request_queue)
            self._close_queue_handle(response_queue)
            raise RuntimeError(
                f"Non-text worker did not become ready within {startup_timeout:.1f}s."
            ) from exc

        if str(startup_message.get("kind", "")) != "ready":
            process.join(timeout=2.0)
            self._close_queue_handle(request_queue)
            self._close_queue_handle(response_queue)
            error_text = str(startup_message.get("error", "Non-text worker startup failed."))
            raise RuntimeError(error_text)

        stop_event = threading.Event()
        response_thread = threading.Thread(
            target=self._response_loop,
            kwargs={
                "process": process,
                "response_queue": response_queue,
                "stop_event": stop_event,
            },
            name=f"non-text-response-{process.pid}",
            daemon=True,
        )
        response_thread.start()

        self._worker_process = process
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._response_thread = response_thread
        self._response_thread_stop = stop_event
        self._worker_pid = int(startup_message.get("pid") or process.pid)
        self._worker_family = family
        LOG.info(
            "Started non-text worker process: pid=%s family=%s queue=%s",
            self._worker_pid,
            family,
            self._queue_capacity,
        )

    def _stop_worker_locked(self, reason: str) -> bool:
        process = self._worker_process
        if process is None:
            return False

        request_queue = self._request_queue
        response_queue = self._response_queue
        response_thread = self._response_thread
        response_stop = self._response_thread_stop
        worker_pid = self._worker_pid
        worker_family = self._worker_family

        LOG.info(
            "Stopping non-text worker process: reason=%s pid=%s family=%s",
            reason,
            worker_pid or "unknown",
            worker_family or "none",
        )

        if process.is_alive() and request_queue is not None:
            try:
                request_queue.put({"kind": "shutdown"}, timeout=1.0)
                request_queue.join()
            except Exception as exc:
                LOG.warning("Graceful non-text worker shutdown request failed: %s", exc)
            process.join(timeout=max(2.0, float(self._execution_timeout_seconds)))

        if process.is_alive():
            process.terminate()
            process.join(timeout=max(2.0, float(self._execution_timeout_seconds)))
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
        if process.is_alive():
            LOG.warning(
                "Non-text worker process did not exit cleanly: pid=%s reason=%s",
                worker_pid or "unknown",
                reason,
            )

        if response_stop is not None:
            response_stop.set()
        if response_thread is not None and response_thread is not threading.current_thread():
            response_thread.join(timeout=2.0)

        self._worker_process = None
        self._request_queue = None
        self._response_queue = None
        self._response_thread = None
        self._response_thread_stop = None
        self._worker_pid = None
        self._worker_family = None

        self._close_queue_handle(request_queue)
        self._close_queue_handle(response_queue)
        _log_current_process_memory(f"{reason}-worker-stop")
        return True

    def _acquire_family(self, family: _NonTextFamily) -> None:
        while True:
            previous_family: Optional[_NonTextFamily]
            with self._family_condition:
                while self._switching:
                    if self._stopping:
                        raise RuntimeError("模型服务已关闭")
                    self._family_condition.wait(timeout=0.1)

                if self._stopping:
                    raise RuntimeError("模型服务已关闭")

                previous_family = self._active_family
                if previous_family is None:
                    if any(self._inflight.values()):
                        self._family_condition.wait(timeout=0.1)
                        continue
                    self._active_family = family
                    self._inflight[family] += 1
                    return

                if previous_family == family:
                    self._inflight[family] += 1
                    return

                if self._inflight[previous_family] > 0:
                    self._family_condition.wait(timeout=0.1)
                    continue

                self._switching = True

            self._release_worker_process(reason=f"family-switch:{previous_family}->{family}")

            with self._family_condition:
                self._active_family = family
                self._inflight[family] += 1
                self._switching = False
                self._family_condition.notify_all()
                return

    def _release_family(self, family: _NonTextFamily) -> None:
        with self._family_condition:
            current = int(self._inflight.get(family, 0))
            if current <= 0:
                return
            self._inflight[family] = current - 1
            if self._inflight[family] == 0:
                self._family_condition.notify_all()

    def _clear_active_family_after_forced_release(self, family: _NonTextFamily) -> None:
        with self._family_condition:
            if self._active_family != family:
                return
            self._active_family = None
            self._switching = False
            self._family_condition.notify_all()

    def _ensure_worker_started(self, family: _NonTextFamily) -> None:
        with self._process_lock:
            self._start_worker_locked(family)

    def _release_worker_process(self, reason: str) -> bool:
        with self._process_lock:
            return self._stop_worker_locked(reason)

    def _release_worker_when_drained(self, reason: str) -> None:
        with self._family_condition:
            while self._switching:
                self._family_condition.wait(timeout=0.1)
            while any(self._inflight.values()):
                self._family_condition.wait(timeout=0.1)
            previous_family = self._active_family
            self._switching = True

        try:
            self._release_worker_process(reason=reason)
        finally:
            with self._family_condition:
                self._active_family = None
                self._switching = False
                self._family_condition.notify_all()
            if previous_family:
                LOG.info(
                    "Released non-text worker family=%s reason=%s.",
                    previous_family,
                    reason,
                )

    def get_loaded_runtime_family(self) -> Optional[str]:
        with self._family_condition:
            return self._active_family

    def _submit_request(self, operation: str, payload: np.ndarray) -> Future[Any]:
        request_queue = self._request_queue
        if request_queue is None:
            raise RuntimeError("Non-text worker request queue is not initialized.")

        request_id = uuid.uuid4().hex
        future: Future[Any] = Future()
        with self._pending_lock:
            self._pending[request_id] = future

        try:
            request_queue.put(
                {
                    "kind": "infer",
                    "request_id": request_id,
                    "operation": operation,
                    "payload": np.ascontiguousarray(payload),
                },
                timeout=max(1.0, float(self._queue_timeout_seconds)),
            )
        except queue.Full as exc:
            self._set_future_exception(
                request_id,
                RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试"),
            )
            raise RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试") from exc
        except Exception as exc:
            self._set_future_exception(
                request_id,
                RuntimeError("提交非文本任务到子进程失败。"),
            )
            raise RuntimeError("提交非文本任务到子进程失败。") from exc
        return future

    async def _invoke_async(
        self,
        *,
        family: _NonTextFamily,
        operation: str,
        image: np.ndarray,
        timeout_seconds: int,
    ) -> Any:
        self._acquire_family(family)
        try:
            await asyncio.to_thread(self._ensure_worker_started, family)
            future = self._submit_request(operation, image)
            wrapped = asyncio.wrap_future(future)
            try:
                return await asyncio.wait_for(
                    asyncio.shield(wrapped),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                timeout_message = RuntimeError(f"推理任务执行超时（>{timeout_seconds}s）")
                self._fail_all_pending(timeout_message)
                await asyncio.to_thread(
                    self._release_worker_process,
                    f"{family}-timeout",
                )
                self._clear_active_family_after_forced_release(family)
                await asyncio.gather(wrapped, return_exceptions=True)
                raise timeout_message from exc
        finally:
            self._release_family(family)

    async def get_image_embedding_async(self, image: np.ndarray) -> list[float]:
        return list(
            await self._invoke_async(
                family="vision",
                operation="clip_img",
                image=np.ascontiguousarray(image),
                timeout_seconds=self._execution_timeout_seconds,
            )
        )

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        return await self._invoke_async(
            family="ocr",
            operation="ocr",
            image=np.ascontiguousarray(image),
            timeout_seconds=self._ocr_execution_timeout_seconds,
        )

    async def get_face_representation_async(self, image: np.ndarray) -> list[RepresentResult]:
        return list(
            await self._invoke_async(
                family="face",
                operation="represent",
                image=np.ascontiguousarray(image),
                timeout_seconds=self._execution_timeout_seconds,
            )
        )

    def release_models_for_restart(self) -> None:
        self._release_worker_when_drained(reason="restart")

    def release_all_models(self) -> None:
        self._stopping = True
        self._idle_release_stop.set()
        self._idle_release_wakeup.set()
        self._join_idle_release_thread(timeout_seconds=max(2.0, float(self._execution_timeout_seconds)))
        try:
            self._release_worker_when_drained(reason="shutdown")
        except Exception as exc:
            LOG.warning("Failed to release non-text worker during shutdown: %s", exc, exc_info=True)
            self._fail_all_pending(RuntimeError("模型服务已关闭"))
            self._release_worker_process(reason="shutdown-force")
