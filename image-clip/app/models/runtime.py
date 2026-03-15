import asyncio
import gc
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
import torch
from transformers import AutoModel

from .common import (
    _AdmissionController,
    _ClipImageTask,
    _InterProcessFileLock,
    _ManagedLease,
    _as_contiguous_bgr_uint8,
)
from .constants import (
    CLIP_EMBEDDING_DIMS,
    CLIP_IMAGE_RESOLUTION,
    EXEC_TIMEOUT_SECONDS,
    HF_CACHE_DIR,
    HF_LOCAL_FILES_ONLY,
    IMAGE_CLIP_DEVICE,
    IMAGE_CLIP_USE_FP16,
    LOG,
    MAX_PENDING_IMAGE_REQUESTS,
    MODEL_ID,
    MODEL_PATH,
    PROJECT_ROOT,
    QUEUE_MAX_SIZE,
    QUEUE_TIMEOUT_SECONDS,
    _CLIP_IMAGE_MEAN,
    _CLIP_IMAGE_STD,
)


class _VisionModelWrapper(torch.nn.Module):
    def __init__(self, loaded_model: Any) -> None:
        super().__init__()
        self.vision_model = loaded_model.vision_model
        self.visual_projection = loaded_model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]
        return self.visual_projection(pooled_output)


class ImageClipRuntime:
    def __init__(self) -> None:
        self.model_base_path = Path(MODEL_PATH)
        self.hf_cache_dir = Path(HF_CACHE_DIR)
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_state_dir = (PROJECT_ROOT / "cache" / "runtime").resolve()
        self._runtime_state_dir.mkdir(parents=True, exist_ok=True)

        configured_queue_capacity = max(1, int(QUEUE_MAX_SIZE))
        self._queue_capacity = min(MAX_PENDING_IMAGE_REQUESTS, configured_queue_capacity)
        if configured_queue_capacity > self._queue_capacity:
            LOG.warning(
                "INFERENCE_QUEUE_MAX_SIZE=%s exceeds safe limit %s; capping to %s.",
                configured_queue_capacity,
                MAX_PENDING_IMAGE_REQUESTS,
                self._queue_capacity,
            )

        self._queue_timeout_seconds = max(1, int(QUEUE_TIMEOUT_SECONDS))
        self._execution_timeout_seconds = max(1, int(EXEC_TIMEOUT_SECONDS))
        self._clip_image_batch_size = max(1, int(os.environ.get("CLIP_IMAGE_BATCH", "8")))
        self._clip_image_batch_wait_seconds = max(
            0.0,
            float(os.environ.get("CLIP_IMAGE_BATCH_WAIT_MS", "5")) / 1000.0,
        )
        self._image_admission = _AdmissionController(self._queue_capacity)
        self._shared_cpu_executor = ThreadPoolExecutor(
            max_workers=max(2, min(8, os.cpu_count() or 4)),
            thread_name_prefix="image-clip-cpu",
        )

        self._single_process_lock = _InterProcessFileLock(
            self._runtime_state_dir / "image-clip-single-process.lock"
        )
        self._load_lock = threading.RLock()
        self._stopping = False
        self._loaded = False
        self._device: Optional[torch.device] = None
        self._resolved_device_name = "cuda"
        self._input_dtype = torch.float16 if IMAGE_CLIP_USE_FP16 else torch.float32
        self._vision_model: Optional[torch.nn.Module] = None
        self._clip_image_dispatch_loop: Optional[asyncio.AbstractEventLoop] = None
        self._clip_image_queue: Optional[asyncio.Queue[Optional[_ClipImageTask]]] = None
        self._clip_image_loop_ready = threading.Event()
        self._clip_image_worker: Optional[threading.Thread] = None
        self._model_source = MODEL_ID

    @property
    def runtime_device_label(self) -> str:
        precision = "fp16" if self._input_dtype == torch.float16 else "fp32"
        return f"{self._resolved_device_name} ({precision})"

    def _prepare_hf_cache_env(self) -> None:
        cache_root = self.hf_cache_dir.resolve()
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    def _acquire_single_process_lock(self) -> None:
        acquired = self._single_process_lock.acquire(timeout=0.0, blocking=False)
        if acquired:
            return
        raise RuntimeError("服务当前固定为单进程运行；检测到已有 Image-CLIP 实例持有运行锁。")

    def _release_single_process_lock(self) -> None:
        self._single_process_lock.release()

    def _resolve_cuda_device(self) -> torch.device:
        normalized = str(IMAGE_CLIP_DEVICE).strip() or "cuda"
        if not normalized.lower().startswith("cuda"):
            raise RuntimeError(
                f"IMAGE_CLIP_DEVICE={normalized} does not request CUDA. This service requires CUDA."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and retry.")

        device = torch.device(normalized)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        self._resolved_device_name = f"{device} - {torch.cuda.get_device_name(device_index)}"
        return device

    def _resolve_model_source(self) -> str:
        local_model_dir = self.model_base_path / "qa-clip" / "huggingface"
        if (local_model_dir / "config.json").exists():
            return str(local_model_dir)
        return MODEL_ID

    def _start_clip_image_worker(self) -> None:
        if self._clip_image_worker is not None:
            return
        self._clip_image_worker = threading.Thread(
            target=self._clip_image_worker_thread_main,
            name="image-clip-queue",
            daemon=True,
        )
        self._clip_image_worker.start()
        ready = self._clip_image_loop_ready.wait(timeout=max(2.0, float(self._execution_timeout_seconds)))
        if not ready:
            raise RuntimeError("Image-CLIP queue worker failed to initialize in time.")

    def _clip_image_worker_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._clip_image_dispatch_loop = loop
        self._clip_image_queue = asyncio.Queue(maxsize=self._queue_capacity)
        self._clip_image_loop_ready.set()
        worker_task = loop.create_task(self._worker_loop())
        try:
            loop.run_forever()
        finally:
            worker_task.cancel()
            loop.run_until_complete(asyncio.gather(worker_task, return_exceptions=True))
            self._clip_image_queue = None
            self._clip_image_dispatch_loop = None
            loop.close()

    def _require_clip_image_queue(self) -> asyncio.Queue[Optional[_ClipImageTask]]:
        queue = self._clip_image_queue
        if queue is None:
            raise RuntimeError("Image-CLIP queue is not initialized.")
        return queue

    async def _worker_loop(self) -> None:
        queue = self._require_clip_image_queue()
        stop_after_batch = False
        while True:
            task = await queue.get()
            if task is None:
                queue.task_done()
                break
            if task.cancel_requested.is_set():
                queue.task_done()
                continue
            task.started_at = time.monotonic()
            task.started_event.set()
            batch, stop_after_batch = await self._collect_clip_image_batch(task)
            try:
                self._handle_clip_image_tasks(batch)
            finally:
                for _ in batch:
                    queue.task_done()
            if stop_after_batch:
                break

    async def _collect_clip_image_batch(
        self,
        first_task: _ClipImageTask,
    ) -> tuple[List[_ClipImageTask], bool]:
        batch = [first_task]
        if self._clip_image_batch_size <= 1:
            return batch, False

        queue = self._require_clip_image_queue()
        deadline = asyncio.get_running_loop().time() + self._clip_image_batch_wait_seconds
        stop_after_batch = False

        while len(batch) < self._clip_image_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0:
                break
            try:
                popped = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if popped is None:
                queue.task_done()
                stop_after_batch = True
                break
            if popped.cancel_requested.is_set():
                queue.task_done()
                continue
            popped.started_at = time.monotonic()
            popped.started_event.set()
            batch.append(popped)
        return batch, stop_after_batch

    def _handle_clip_image_tasks(self, tasks: List[_ClipImageTask]) -> None:
        if not tasks:
            return
        try:
            results = self._infer_clip_image_tensor_batch([task.payload for task in tasks])
            for task, result in zip(tasks, results):
                self._safe_set_result(task.future, result)
        except Exception as exc:
            LOG.error("Image-CLIP batch inference failed: %s", exc, exc_info=True)
            for task in tasks:
                self._safe_set_exception(task.future, exc)

    def _validate_model_output_dims(self) -> None:
        if self._vision_model is None or self._device is None:
            raise RuntimeError("Image-CLIP model is not loaded.")

        sample = torch.zeros(
            (1, 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION),
            device=self._device,
            dtype=self._input_dtype,
        )
        with torch.inference_mode():
            embedding = self._vision_model(sample).detach()
        if tuple(embedding.shape) != (1, CLIP_EMBEDDING_DIMS):
            raise RuntimeError(
                "Image-CLIP output dims mismatch: "
                f"expected={(1, CLIP_EMBEDDING_DIMS)}, got={tuple(embedding.shape)}"
            )

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                return

            self._acquire_single_process_lock()
            try:
                self._prepare_hf_cache_env()
                self._device = self._resolve_cuda_device()
                self._model_source = self._resolve_model_source()
                loaded_model = AutoModel.from_pretrained(
                    self._model_source,
                    cache_dir=str(self.hf_cache_dir),
                    local_files_only=HF_LOCAL_FILES_ONLY,
                )
                vision_model = _VisionModelWrapper(loaded_model)
                del loaded_model
                gc.collect()

                if self._input_dtype == torch.float16:
                    vision_model = vision_model.to(device=self._device, dtype=torch.float16)
                else:
                    vision_model = vision_model.to(device=self._device)
                vision_model.eval()

                self._vision_model = vision_model
                self._validate_model_output_dims()
                self._start_clip_image_worker()
                self._loaded = True
                LOG.info(
                    "Image-CLIP model loaded: model=%s device=%s batch=%s/%sms cache=%s",
                    self._model_source,
                    self.runtime_device_label,
                    self._clip_image_batch_size,
                    int(self._clip_image_batch_wait_seconds * 1000.0),
                    self.hf_cache_dir,
                )
            except Exception:
                self._vision_model = None
                self._device = None
                self._loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._release_single_process_lock()
                raise

    @staticmethod
    def _resize_and_center_crop_clip_image(image: np.ndarray) -> np.ndarray:
        image_bgr = _as_contiguous_bgr_uint8(image, context="Image-CLIP")
        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Image-CLIP expects non-empty image, got shape={image_bgr.shape}")

        scale = float(CLIP_IMAGE_RESOLUTION) / float(min(height, width))
        resized_width = max(CLIP_IMAGE_RESOLUTION, int(round(width * scale)))
        resized_height = max(CLIP_IMAGE_RESOLUTION, int(round(height * scale)))
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        top = max(0, (resized_height - CLIP_IMAGE_RESOLUTION) // 2)
        left = max(0, (resized_width - CLIP_IMAGE_RESOLUTION) // 2)
        cropped = resized[
            top : top + CLIP_IMAGE_RESOLUTION,
            left : left + CLIP_IMAGE_RESOLUTION,
        ]
        if cropped.shape[:2] != (CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION):
            cropped = cv2.resize(
                cropped,
                (CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION),
                interpolation=cv2.INTER_CUBIC,
            )
        return np.ascontiguousarray(cropped, dtype=np.uint8)

    def _preprocess_clip_image_tensor(self, image: np.ndarray) -> np.ndarray:
        cropped = self._resize_and_center_crop_clip_image(image)
        rgb = cropped[:, :, ::-1].astype(np.float32) / 255.0
        normalized = (rgb - _CLIP_IMAGE_MEAN) / _CLIP_IMAGE_STD
        chw = np.transpose(normalized, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _infer_clip_image_tensor_batch(self, tensors: List[np.ndarray]) -> List[List[float]]:
        if self._vision_model is None or self._device is None:
            raise RuntimeError("Image-CLIP model is not loaded.")
        if not tensors:
            return []

        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        with torch.inference_mode():
            pixel_values = torch.from_numpy(batch).to(
                device=self._device,
                dtype=self._input_dtype,
                non_blocking=False,
            )
            embeddings = self._vision_model(pixel_values).detach().float().cpu().numpy()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape != (len(tensors), CLIP_EMBEDDING_DIMS):
            raise RuntimeError(
                "Invalid image embedding dims for Image-CLIP batch: "
                f"expected={(len(tensors), CLIP_EMBEDDING_DIMS)}, got={tuple(embeddings.shape)}"
            )
        return [
            embeddings[index].astype(np.float32, copy=False).tolist()
            for index in range(len(tensors))
        ]

    async def _enqueue_clip_image_task_async(self, task: _ClipImageTask) -> None:
        if self._stopping:
            raise RuntimeError("模型服务已关闭")
        queue = self._clip_image_queue
        if queue is None:
            raise RuntimeError("Image-CLIP queue is not initialized.")
        if queue.full():
            raise RuntimeError(f"推理队列已满（上限 {self._queue_capacity}），请稍后重试")
        queue.put_nowait(task)

    def _submit_clip_image_task(self, payload: Any) -> _ClipImageTask:
        future: Future[Any] = Future()
        task = _ClipImageTask(payload=payload, future=future, created_at=time.time())
        loop = self._clip_image_dispatch_loop
        if self._stopping:
            future.set_exception(RuntimeError("模型服务已关闭"))
            return task
        if loop is None:
            future.set_exception(RuntimeError("Image-CLIP queue loop is not initialized."))
            return task
        submit_future = asyncio.run_coroutine_threadsafe(
            self._enqueue_clip_image_task_async(task),
            loop,
        )
        try:
            submit_future.result(timeout=max(1.0, float(self._queue_timeout_seconds)))
        except Exception as exc:
            submit_exc = exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc))
            self._safe_set_exception(future, submit_exc)
        return task

    def _cancel_clip_image_task_if_queued(self, task: _ClipImageTask, exc: Exception) -> bool:
        if task.started_event.is_set():
            return False
        task.cancel_requested.set()
        self._safe_set_exception(task.future, exc)
        return True

    async def _await_clip_image_task(self, task: _ClipImageTask) -> Any:
        if task.future.done():
            return task.future.result()

        started = bool(await asyncio.to_thread(task.started_event.wait, self._queue_timeout_seconds))
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

    async def get_image_embedding_async(self, image: np.ndarray) -> List[float]:
        if not self._loaded:
            await asyncio.to_thread(self.load)

        lease = _ManagedLease("Image-CLIP")
        acquired = await self._image_admission.acquire_async(timeout=0.0)
        if not acquired:
            raise RuntimeError(
                f"CLIP 图片请求总量已满（上限 {self._image_admission.capacity}），请稍后重试"
            )
        lease.push(self._image_admission.release)

        try:
            payload = await asyncio.get_running_loop().run_in_executor(
                self._shared_cpu_executor,
                self._preprocess_clip_image_tensor,
                image,
            )
            task = self._submit_clip_image_task(payload=payload)
            lease.bind_future(task.future)
            return await self._await_clip_image_task(task)
        except Exception:
            await lease.release_async()
            raise

    def _safe_set_result(self, future: Future[Any], value: Any) -> None:
        if not future.done():
            future.set_result(value)

    def _safe_set_exception(self, future: Future[Any], exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

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

    def release(self) -> None:
        with self._load_lock:
            self._stopping = True
            pending_tasks: List[_ClipImageTask] = []

            clip_queue_loop = self._clip_image_dispatch_loop
            if clip_queue_loop is not None:
                shutdown_future = asyncio.run_coroutine_threadsafe(
                    self._shutdown_clip_image_queue_async(),
                    clip_queue_loop,
                )
                try:
                    pending_tasks = shutdown_future.result(
                        timeout=max(2.0, float(self._execution_timeout_seconds))
                    )
                except Exception as exc:
                    LOG.warning("Failed to drain Image-CLIP queue during shutdown: %s", exc)
                clip_queue_loop.call_soon_threadsafe(clip_queue_loop.stop)

            for task in pending_tasks:
                self._safe_set_exception(task.future, RuntimeError("模型服务正在关闭"))

            if self._clip_image_worker is not None:
                self._clip_image_worker.join(timeout=max(2.0, float(self._execution_timeout_seconds)))
                self._clip_image_worker = None

            self._shared_cpu_executor.shutdown(wait=True, cancel_futures=True)
            self._vision_model = None
            self._device = None
            self._loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._release_single_process_lock()
