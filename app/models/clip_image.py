import asyncio
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov

from .common import TaskType, _InferenceTask, _as_contiguous_bgr_uint8
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
    _condition: Any
    _stopping: bool
    _normal_queue: Any
    _shared_cpu_executor: ThreadPoolExecutor
    _clip_image_batch_size: int
    _clip_image_batch_wait_seconds: float
    _clip_vision_load_lock: Any
    _clip_vision_model: Optional[ov.CompiledModel]
    _clip_vision_ppp: Any
    _clip_vision_request: Optional[ov.InferRequest]
    _clip_vision_input_name: Optional[str]
    _clip_vision_host_input_cache: dict[tuple[int, int, int], tuple[ov.Tensor, Any]]
    _clip_vision_host_tensor_enabled: bool

    if TYPE_CHECKING:
        def _load_family_with_process_lock(self, family: str, loader: Any) -> None: ...
        def _build_openvino_preprocess_runner(self, **kwargs: Any) -> Any: ...
        def _safe_set_result(self, future: Future[Any], value: Any) -> None: ...
        def _safe_set_exception(self, future: Future[Any], exc: Exception) -> None: ...
        def _submit_task(self, kind: TaskType, payload: Any) -> _InferenceTask: ...
        def _wait_task(self, task: _InferenceTask) -> Any: ...
        async def _await_task(self, task: _InferenceTask) -> Any: ...
        def _bind_non_text_lease_to_future(
            self,
            family: str,
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
            normal_task: Optional[_InferenceTask] = None

            with self._condition:
                while True:
                    if self._stopping or self._normal_queue:
                        break
                    self._condition.wait()

                if self._stopping:
                    break

                if self._normal_queue:
                    normal_task = self._normal_queue.popleft()
                    normal_task.started_at = time.monotonic()
                    normal_task.started_event.set()

            if normal_task is not None:
                self._handle_clip_image_tasks(self._collect_clip_image_batch(normal_task))

    def _collect_clip_image_batch(self, first_task: _InferenceTask) -> List[_InferenceTask]:
        batch = [first_task]
        if self._clip_image_batch_size <= 1:
            return batch

        deadline = time.monotonic() + self._clip_image_batch_wait_seconds

        while len(batch) < self._clip_image_batch_size:
            with self._condition:
                if not self._normal_queue:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return batch
                    self._condition.wait(timeout=remaining)
                    continue

                next_task = self._normal_queue[0]
                if next_task.kind != "clip_img":
                    return batch
                popped = self._normal_queue.popleft()
                popped.started_at = time.monotonic()
                popped.started_event.set()
                batch.append(popped)
        return batch

    def _handle_clip_image_tasks(self, tasks: List[_InferenceTask]) -> None:
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
        self._clip_vision_input_name = None
        self._clip_vision_host_input_cache.clear()

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

    def _build_clip_vision_ppp_model(self, model_path: Path) -> ov.Model:
        model = self.core.read_model(str(model_path))
        input_shape = model.inputs[0].get_partial_shape()
        if not input_shape.rank.is_static or input_shape.rank.get_length() != 4:
            raise RuntimeError(
                "CLIP vision input rank must be 4, "
                f"got rank={input_shape.rank}"
            )

        model_height = input_shape[2]
        model_width = input_shape[3]
        if model_height.is_static and int(model_height.get_length()) != int(CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP image height mismatch: "
                f"expected={CLIP_IMAGE_RESOLUTION}, got={int(model_height.get_length())}"
            )
        if model_width.is_static and int(model_width.get_length()) != int(CLIP_IMAGE_RESOLUTION):
            raise RuntimeError(
                "CLIP image width mismatch: "
                f"expected={CLIP_IMAGE_RESOLUTION}, got={int(model_width.get_length())}"
            )

        ppp = ov.preprocess.PrePostProcessor(model)
        ppp.input().tensor().set_shape([-1, -1, -1, 3]).set_element_type(ov.Type.u8).set_layout(
            ov.Layout("NHWC")
        ).set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().preprocess().resize(
            ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR,
            int(CLIP_IMAGE_RESOLUTION),
            int(CLIP_IMAGE_RESOLUTION),
        )
        ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(ov.Type.f32)
        ppp.input().preprocess().scale([255.0, 255.0, 255.0])
        ppp.input().preprocess().mean(_CLIP_IMAGE_MEAN.tolist())
        ppp.input().preprocess().scale(_CLIP_IMAGE_STD.tolist())
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        return ppp.build()

    def _get_clip_vision_host_input(
        self, batch_size: int, image_height: int, image_width: int
    ) -> Optional[Tuple[ov.Tensor, np.ndarray]]:
        if (
            not self._clip_vision_host_tensor_enabled
            or self._clip_remote_context is None
            or self._clip_vision_model is None
        ):
            return None

        cache_key = (int(batch_size), int(image_height), int(image_width))
        cached = self._clip_vision_host_input_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            input_port = self._clip_vision_model.inputs[0]
            input_shape = ov.Shape([int(batch_size), int(image_height), int(image_width), 3])
            input_tensor = self._clip_remote_context.create_host_tensor(
                input_port.get_element_type(), input_shape
            )
            input_view = np.asarray(input_tensor.data)
            cache_entry = (input_tensor, input_view)
            self._clip_vision_host_input_cache[cache_key] = cache_entry
            return cache_entry
        except Exception as exc:
            LOG.warning(
                "CLIP vision host tensor allocation failed, fallback to shared numpy inputs: %s",
                exc,
            )
            self._clip_vision_host_tensor_enabled = False
            self._clip_vision_host_input_cache.clear()
            return None

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
        self._clip_vision_input_name = self._clip_vision_model.inputs[0].any_name
        self._clip_vision_ppp = self._build_openvino_preprocess_runner(
            runner_name="clip_vision",
            device_name=self._clip_inference_device,
            output_height=CLIP_IMAGE_RESOLUTION,
            output_width=CLIP_IMAGE_RESOLUTION,
            mean_values=_CLIP_IMAGE_MEAN.tolist(),
            std_values=_CLIP_IMAGE_STD.tolist(),
        )
        self._clip_vision_host_input_cache.clear()
        self._clip_vision_host_tensor_enabled = False
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

    def get_image_embedding(self, image: np.ndarray, filename: str = "unknown") -> List[float]:
        _ = filename
        lease_bound = False
        self._acquire_non_text_family_lease("vision")
        try:
            self._ensure_clip_vision_loaded()
            payload = self._preprocess_clip_image_tensor(image)
            task = self._submit_task(kind="clip_img", payload=payload)
            self._bind_non_text_lease_to_future("vision", task.future)
            lease_bound = True
            return self._wait_task(task)
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("vision")

    async def get_image_embedding_async(
        self, image: np.ndarray, filename: str = "unknown"
    ) -> List[float]:
        _ = filename
        lease_bound = False
        await self._acquire_non_text_family_lease_async("vision")
        try:
            await self._run_control(self._ensure_clip_vision_loaded)
            payload = await self._run_in_executor(
                self._shared_cpu_executor,
                self._preprocess_clip_image_tensor,
                image,
            )
            task = self._submit_task(kind="clip_img", payload=payload)
            self._bind_non_text_lease_to_future("vision", task.future)
            lease_bound = True
            return await self._await_task(task)
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("vision")
