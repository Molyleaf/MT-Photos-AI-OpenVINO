import asyncio
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import cv2
import numpy as np
import openvino as ov

from .common import _ClipImageTask, _as_contiguous_bgr_uint8
from .constants import (
    CLIP_EMBEDDING_DIMS,
    CLIP_IMAGE_RESOLUTION,
    LOG,
    _CLIP_IMAGE_MEAN,
    _CLIP_IMAGE_STD,
)


class ClipImageMixin:
    core: ov.Core
    qa_clip_path: Path
    _clip_inference_device: str
    _clip_remote_context: Any
    _clip_image_condition: Any
    _stopping: bool
    _clip_image_queue: Any
    _shared_cpu_executor: ThreadPoolExecutor
    _clip_image_batch_size: int
    _clip_image_batch_wait_seconds: float
    _clip_vision_load_lock: Any
    _clip_vision_model: Optional[ov.CompiledModel]
    _clip_vision_ppp: Any
    _clip_vision_request: Optional[ov.InferRequest]

    if TYPE_CHECKING:
        def _acquire_image_request_slot(self, label: str) -> None: ...
        def _release_image_request_slot(self) -> None: ...
        def _load_family_with_process_lock(self, family: str, loader: Any) -> None: ...
        def _build_openvino_preprocess_runner(self, **kwargs: Any) -> Any: ...
        def _safe_set_result(self, future: Future[Any], value: Any) -> None: ...
        def _safe_set_exception(self, future: Future[Any], exc: Exception) -> None: ...
        def _submit_clip_image_task(self, payload: Any) -> _ClipImageTask: ...
        def _wait_clip_image_task(self, task: _ClipImageTask) -> Any: ...
        async def _await_clip_image_task(self, task: _ClipImageTask) -> Any: ...
        def _bind_non_text_lease_to_future(
            self,
            family: str,
            future: Future[Any] | asyncio.Future[Any],
        ) -> None: ...
        def _bind_image_request_slot_to_future(
            self,
            future: Future[Any] | asyncio.Future[Any],
        ) -> None: ...
        def _acquire_non_text_family_lease(self, family: str) -> bool: ...
        async def _acquire_non_text_family_lease_async(self, family: str) -> bool: ...
        def _release_non_text_family_lease(self, family: str) -> None: ...
        def _run_control(
            self,
            func: Callable[..., Any],
            *args: Any,
        ) -> asyncio.Future[Any]: ...
        @staticmethod
        def _run_in_executor(
            executor: ThreadPoolExecutor,
            func: Any,
            *args: Any,
        ) -> asyncio.Future[Any]: ...

    def _worker_loop(self) -> None:
        while True:
            clip_image_task: Optional[_ClipImageTask] = None

            with self._clip_image_condition:
                while True:
                    if self._stopping or self._clip_image_queue:
                        break
                    self._clip_image_condition.wait()

                if self._stopping:
                    break

                if self._clip_image_queue:
                    clip_image_task = self._clip_image_queue.popleft()
                    clip_image_task.started_at = time.monotonic()
                    clip_image_task.started_event.set()

            if clip_image_task is not None:
                self._handle_clip_image_tasks(
                    self._collect_clip_image_batch(clip_image_task)
                )

    def _collect_clip_image_batch(self, first_task: _ClipImageTask) -> List[_ClipImageTask]:
        batch = [first_task]
        if self._clip_image_batch_size <= 1:
            return batch

        deadline = time.monotonic() + self._clip_image_batch_wait_seconds

        while len(batch) < self._clip_image_batch_size:
            with self._clip_image_condition:
                if not self._clip_image_queue:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return batch
                    self._clip_image_condition.wait(timeout=remaining)
                    continue

                popped = self._clip_image_queue.popleft()
                popped.started_at = time.monotonic()
                popped.started_event.set()
                batch.append(popped)
        return batch

    def _handle_clip_image_tasks(self, tasks: List[_ClipImageTask]) -> None:
        if not tasks:
            return
        try:
            self._ensure_clip_vision_loaded()
            payloads = [np.asarray(task.payload, dtype=np.float32) for task in tasks]
            results = self._infer_clip_image_tensor_batch(payloads)
            for task, result in zip(tasks, results):
                self._safe_set_result(task.future, result)
        except Exception as exc:
            LOG.error("Task clip_img failed: %s", exc, exc_info=True)
            for task in tasks:
                self._safe_set_exception(task.future, exc)

    def _unload_clip_vision_model_locked(self) -> None:
        self._clip_vision_request = None
        self._clip_vision_model = None
        self._clip_vision_ppp = None

    def _unload_clip_vision_model(self) -> None:
        with self._clip_vision_load_lock:
            self._unload_clip_vision_model_locked()

    def _ensure_clip_vision_loaded(self) -> None:
        with self._clip_vision_load_lock:
            if self._clip_vision_model is not None and self._clip_vision_request is not None:
                return
            self._load_family_with_process_lock("vision", self._load_clip_vision_locked)

    def _compile_clip_model(
        self,
        model_or_path: Any,
        performance_hint: str,
    ) -> ov.CompiledModel:
        config = {
            "PERFORMANCE_HINT": performance_hint,
        }

        if self._clip_remote_context is not None:
            if isinstance(model_or_path, Path):
                model = self.core.read_model(str(model_or_path))
            else:
                model = model_or_path
            return self.core.compile_model(model, self._clip_remote_context, config)

        if isinstance(model_or_path, Path):
            return self.core.compile_model(str(model_or_path), self._clip_inference_device, config)
        return self.core.compile_model(model_or_path, self._clip_inference_device, config)

    def _load_clip_vision_locked(self) -> None:
        vision_model_path = self.qa_clip_path / "openvino_image_fp16.xml"
        if not vision_model_path.exists():
            raise FileNotFoundError(f"Missing vision model: {vision_model_path}")

        model = self.core.read_model(str(vision_model_path))
        model.reshape(
            {
                model.input(0): ov.PartialShape(
                    [ov.Dimension.dynamic(), 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION]
                )
            }
        )
        compiled_model = self._compile_clip_model(
            model_or_path=model,
            performance_hint="THROUGHPUT",
        )
        output_dim = compiled_model.outputs[0].get_partial_shape()[1].get_length()
        if output_dim != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Vision embedding dims mismatch: expected={CLIP_EMBEDDING_DIMS}, got={output_dim}"
            )

        self._clip_vision_model = compiled_model
        self._clip_vision_request = compiled_model.create_infer_request()
        self._clip_vision_ppp = self._build_openvino_preprocess_runner(
            runner_name="clip_vision",
            device_name=self._clip_inference_device,
            output_height=CLIP_IMAGE_RESOLUTION,
            output_width=CLIP_IMAGE_RESOLUTION,
            mean_values=_CLIP_IMAGE_MEAN.tolist(),
            std_values=_CLIP_IMAGE_STD.tolist(),
        )
        LOG.info(
            "CLIP Vision model loaded on %s with post-preprocess batching (batch=%s).",
            self._clip_inference_device,
            self._clip_image_batch_size,
        )

    @staticmethod
    def _resize_and_center_crop_clip_image(image: np.ndarray) -> np.ndarray:
        image_bgr = _as_contiguous_bgr_uint8(image, context="CLIP vision")
        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"CLIP vision expects non-empty image, got shape={image_bgr.shape}")

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
        runner = self._clip_vision_ppp
        if runner is None:
            raise RuntimeError("CLIP vision preprocess runner is not loaded.")
        cropped = self._resize_and_center_crop_clip_image(image)
        normalized = runner.run(cropped[np.newaxis, ...])
        if normalized.shape != (1, 3, CLIP_IMAGE_RESOLUTION, CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP preprocess output shape mismatch: "
                f"got={tuple(normalized.shape)}"
            )
        return np.ascontiguousarray(normalized[0], dtype=np.float32)

    def _infer_clip_image_tensor_batch(self, tensors: List[np.ndarray]) -> List[List[float]]:
        if not self._clip_vision_model or not self._clip_vision_request:
            raise RuntimeError("CLIP vision model is not loaded.")
        if not tensors:
            return []

        batch = np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)
        self._clip_vision_request.set_input_tensor(0, ov.Tensor(batch, shared_memory=True))
        self._clip_vision_request.infer()
        embeddings = np.asarray(self._clip_vision_request.get_output_tensor(0).data)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape != (len(tensors), CLIP_EMBEDDING_DIMS):
            raise RuntimeError(
                "Invalid image embedding dims for CLIP vision batch: "
                f"expected={(len(tensors), CLIP_EMBEDDING_DIMS)}, got={tuple(embeddings.shape)}"
            )
        return [
            embeddings[index].astype(np.float32, copy=False).tolist()
            for index in range(len(tensors))
        ]

    def get_image_embedding(self, image: np.ndarray) -> List[float]:
        lease_bound = False
        image_slot_bound = False
        self._acquire_image_request_slot("CLIP")
        try:
            self._acquire_non_text_family_lease("vision")
            try:
                self._ensure_clip_vision_loaded()
                payload = self._preprocess_clip_image_tensor(image)
                task = self._submit_clip_image_task(payload=payload)
                self._bind_non_text_lease_to_future("vision", task.future)
                self._bind_image_request_slot_to_future(task.future)
                lease_bound = True
                image_slot_bound = True
                return self._wait_clip_image_task(task)
            finally:
                if not lease_bound:
                    self._release_non_text_family_lease("vision")
        finally:
            if not image_slot_bound:
                self._release_image_request_slot()

    async def get_image_embedding_async(self, image: np.ndarray) -> List[float]:
        lease_bound = False
        image_slot_bound = False
        self._acquire_image_request_slot("CLIP")
        try:
            await self._acquire_non_text_family_lease_async("vision")
            try:
                await self._run_control(self._ensure_clip_vision_loaded)
                payload = await self._run_in_executor(
                    self._shared_cpu_executor,
                    self._preprocess_clip_image_tensor,
                    image,
                )
                task = self._submit_clip_image_task(payload=payload)
                self._bind_non_text_lease_to_future("vision", task.future)
                self._bind_image_request_slot_to_future(task.future)
                lease_bound = True
                image_slot_bound = True
                return await self._await_clip_image_task(task)
            finally:
                if not lease_bound:
                    self._release_non_text_family_lease("vision")
        finally:
            if not image_slot_bound:
                self._release_image_request_slot()
