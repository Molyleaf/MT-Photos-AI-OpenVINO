import asyncio
import inspect
import json
import os
import queue
import shutil
import sys
import threading
import time
from abc import abstractmethod, ABC
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import onnx
from insightface.app import FaceAnalysis

from .common import (
    _AdmissionController,
    _FaceInferenceTask,
    _InferenceCancelled,
    NonTextFamily,
    _OpenVinoPreprocessRunner,
    _as_bool,
    _as_contiguous_bgr_uint8,
    _as_int,
    _ensure_intel_opencl_device,
    _get_compiled_model_execution_devices,
    _normalize_non_text_openvino_device,
    _resolve_non_text_openvino_runtime_device,
    _to_opencv_umat,
    _to_channel_triplet,
)
from .constants import INFERENCE_DEVICE, LOG, MODEL_NAME
from .schemas import FacialArea, RepresentResult

_INSIGHTFACE_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
_INSIGHTFACE_BATCH_DIM_PARAM = "batch"


@dataclass(frozen=True)
class _InsightFaceInitAttempt:
    name: str
    runtime_root: Path
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class _LoadedInsightFaceRuntime:
    face_app: FaceAnalysis
    runtime_root: Path
    provider_runtime: Dict[str, Dict[str, Any]]
    ppp_execution_devices: Dict[str, List[str]]
    rec_session_shapes: Dict[str, List[str]]


@dataclass(slots=True)
class _PreparedInsightFaceImage:
    detector_input: np.ndarray
    det_scale: float
    image_shape: Tuple[int, int]
    source_umat: cv2.UMat


@dataclass(slots=True)
class _PendingInsightFaceRecognition:
    task_index: int
    face_meta: List[Tuple[np.ndarray, float]]
    aligned_faces: List[np.ndarray]


def _estimate_similarity_transform_matrix(src: Any, dst: Any) -> np.ndarray:
    src_points = np.asarray(src, dtype=np.float64)
    dst_points = np.asarray(dst, dtype=np.float64)
    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError(
            "InsightFace alignment expects matching 2D landmarks, "
            f"got src={src_points.shape}, dst={dst_points.shape}"
        )

    point_count = src_points.shape[0]
    if point_count < 2:
        raise ValueError(
            "InsightFace alignment requires at least 2 points, "
            f"got {point_count}"
        )

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_demean = src_points - src_mean
    dst_demean = dst_points - dst_mean

    covariance = dst_demean.T @ src_demean / float(point_count)
    diagonal = np.ones((2,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        diagonal[1] = -1.0

    u_mat, singular_values, vh_mat = np.linalg.svd(covariance)
    rank = int(np.linalg.matrix_rank(covariance))
    if rank == 0:
        raise RuntimeError("InsightFace alignment landmarks are ill-conditioned.")

    if rank == 1:
        if np.linalg.det(u_mat) * np.linalg.det(vh_mat) > 0:
            rotation = u_mat @ vh_mat
        else:
            last_value = diagonal[1]
            diagonal[1] = -1.0
            rotation = u_mat @ np.diag(diagonal) @ vh_mat
            diagonal[1] = last_value
    else:
        rotation = u_mat @ np.diag(diagonal) @ vh_mat

    src_variance = float(src_demean.var(axis=0).sum())
    if src_variance <= 0.0:
        raise RuntimeError("InsightFace alignment source landmarks have zero variance.")
    scale = float(singular_values @ diagonal) / src_variance

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = dst_mean - scale * (rotation @ src_mean.T)
    transform[:2, :2] *= scale

    if np.isnan(transform).any():
        raise RuntimeError("InsightFace alignment similarity transform contains NaN.")
    return transform[:2, :].astype(np.float32, copy=False)


def _estimate_insightface_norm_matrix(
    landmark: Any,
    image_size: int,
) -> np.ndarray:
    landmark_array = np.asarray(landmark, dtype=np.float32)
    if landmark_array.shape != (5, 2):
        raise ValueError(
            "InsightFace alignment expects 5 facial landmarks, "
            f"got {landmark_array.shape}"
        )

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(
            "InsightFace alignment image_size must be divisible by 112 or 128, "
            f"got {image_size}"
        )

    destination = _INSIGHTFACE_ARCFACE_TEMPLATE.copy() * ratio
    destination[:, 0] += diff_x
    return _estimate_similarity_transform_matrix(landmark_array, destination)


class InsightFaceMixin(ABC):
    core: Any
    ov_cache_dir: Optional[Path]
    insightface_root: Path
    insightface_model_root: Path
    _face_engine: Optional[FaceAnalysis]
    _face_det_ppp: Optional[_OpenVinoPreprocessRunner]
    _face_rec_ppp: Optional[_OpenVinoPreprocessRunner]
    _face_load_lock: Any
    _face_admission: _AdmissionController
    _face_worker_count: int
    _face_batch_size: int
    _face_batch_wait_seconds: float
    _face_queue_capacity: int
    _face_task_queue: Optional[queue.Queue[Optional[_FaceInferenceTask]]]
    _face_batch_dispatcher: Optional[threading.Thread]
    _face_batch_workers: List[threading.Thread]
    _face_worker_queues: List[queue.Queue[Optional[List[_FaceInferenceTask]]]]
    _face_available_workers: Optional[queue.Queue[int]]
    _face_batch_ready: threading.Event
    _execution_timeout_seconds: int

    @abstractmethod
    def _build_openvino_preprocess_runner(
        self,
        runner_name: str,
        device_name: str,
        output_height: int,
        output_width: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> _OpenVinoPreprocessRunner:
        raise NotImplementedError

    @abstractmethod
    def _load_family_with_process_lock(self, family: NonTextFamily, loader: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _non_text_request_scope(
        self,
        *,
        family: NonTextFamily,
        admission: _AdmissionController,
        label: str,
        ensure_loaded: Any,
    ) -> AbstractContextManager[Any]:
        raise NotImplementedError

    @abstractmethod
    def _non_text_request_scope_async(
        self,
        *,
        family: NonTextFamily,
        admission: _AdmissionController,
        label: str,
        ensure_loaded: Any,
    ) -> AbstractAsyncContextManager[Any]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _run_in_executor(
        executor: ThreadPoolExecutor,
        func: Any,
        *args: Any,
    ) -> asyncio.Future[Any]:
        raise NotImplementedError

    @abstractmethod
    async def _await_with_timeout_and_cooperative_cancel(
        self,
        awaitable: asyncio.Future[Any] | asyncio.Task[Any],
        *,
        cancel_event: threading.Event,
        timeout_seconds: float,
        task_name: str,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def _to_int_pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
        try:
            first, second = value
            return int(first), int(second)
        except Exception:
            return default

    @staticmethod
    def _enable_insightface_opencl_pipeline() -> None:
        name, vendor = _ensure_intel_opencl_device("InsightFace preprocessing/alignment")
        LOG.info(
            "OpenCV OpenCL enabled for InsightFace resize/alignment on Intel device: %s (%s).",
            name,
            vendor,
        )

    @staticmethod
    def _align_face_opencl(
        img: Any,
        landmark: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        matrix = _estimate_insightface_norm_matrix(landmark=landmark, image_size=int(image_size))
        if matrix.shape != (2, 3):
            raise RuntimeError(
                "InsightFace estimate_norm returned invalid affine matrix shape: "
                f"{matrix.shape}"
            )
        if not cv2.ocl.useOpenCL():
            raise RuntimeError(
                "OpenCV OpenCL was disabled during InsightFace alignment. "
                "No silent fallback is allowed."
            )
        if isinstance(img, cv2.UMat):
            source = img
        else:
            source = _to_opencv_umat(
                _as_contiguous_bgr_uint8(img, context="InsightFace alignment")
            )
        try:
            warped_umat = cv2.warpAffine(
                source,
                cast(Any, matrix),
                (int(image_size), int(image_size)),
                borderValue=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenCV OpenCL warpAffine failed in InsightFace alignment. "
                "No silent fallback is allowed."
            ) from exc

        if isinstance(warped_umat, cv2.UMat):
            return _as_contiguous_bgr_uint8(warped_umat.get(), context="InsightFace alignment")
        return _as_contiguous_bgr_uint8(
            np.asarray(warped_umat),
            context="InsightFace alignment",
        )

    def _prepare_insightface_detection_input(
        self,
        img: np.ndarray,
        det_model: Any,
    ) -> _PreparedInsightFaceImage:
        image = _as_contiguous_bgr_uint8(img, context="InsightFace detector")
        input_size = tuple(det_model.input_size)
        im_ratio = float(image.shape[0]) / float(image.shape[1])
        model_ratio = float(input_size[1]) / float(input_size[0])
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / float(image.shape[0])

        if not cv2.ocl.useOpenCL():
            raise RuntimeError(
                "OpenCV OpenCL was disabled during InsightFace detector preprocessing. "
                "No silent fallback is allowed."
            )

        source_umat = _to_opencv_umat(image)
        try:
            resized_umat = cv2.resize(
                source_umat,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            det_umat = cv2.copyMakeBorder(
                resized_umat,
                0,
                int(input_size[1] - new_height),
                0,
                int(input_size[0] - new_width),
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenCV OpenCL detector resize/letterbox failed. "
                "No silent fallback is allowed."
            ) from exc

        return _PreparedInsightFaceImage(
            detector_input=_as_contiguous_bgr_uint8(
                det_umat.get(),
                context="InsightFace detector",
            ),
            det_scale=det_scale,
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            source_umat=source_umat,
        )

    @staticmethod
    def _resolve_insightface_detector_helpers(
        det_model: Any,
    ) -> Tuple[Any, Optional[Any]]:
        module = sys.modules.get(det_model.__class__.__module__)
        if module is None:
            raise RuntimeError(
                f"Cannot resolve module for detector class: {det_model.__class__.__module__}"
            )
        distance2bbox = getattr(module, "distance2bbox", None)
        distance2kps = getattr(module, "distance2kps", None)
        if distance2bbox is None:
            raise RuntimeError("InsightFace detector module missing distance2bbox helper.")
        if getattr(det_model, "use_kps", False) and distance2kps is None:
            raise RuntimeError("InsightFace detector module missing distance2kps helper.")
        return distance2bbox, distance2kps

    @staticmethod
    def _normalize_insightface_detector_state(
        det_model: Any,
        *,
        det_size: Tuple[int, int],
        det_thresh: float,
        nms_thresh: float = 0.4,
    ) -> None:
        init_vars = getattr(det_model, "_init_vars", None)
        required_attrs = (
            "input_name",
            "output_names",
            "fmc",
            "_feat_stride_fpn",
            "_num_anchors",
            "use_kps",
            "input_mean",
            "input_std",
        )
        if callable(init_vars) and any(not hasattr(det_model, attr) for attr in required_attrs):
            init_vars()

        if not hasattr(det_model, "center_cache") or getattr(det_model, "center_cache") is None:
            det_model.center_cache = {}
        if getattr(det_model, "det_thresh", None) is None:
            det_model.det_thresh = float(det_thresh)
        if getattr(det_model, "nms_thresh", None) is None:
            det_model.nms_thresh = float(nms_thresh)
        if getattr(det_model, "input_size", None) is None:
            det_model.input_size = tuple(int(value) for value in det_size)

    @classmethod
    def _prepare_insightface_detector(
        cls,
        face_app: FaceAnalysis,
        *,
        det_size: Tuple[int, int],
        det_thresh: float,
    ) -> None:
        det_model = getattr(face_app, "det_model", None)
        if det_model is None:
            raise RuntimeError("InsightFace detection model is missing.")

        prepare = getattr(det_model, "prepare", None)
        if callable(prepare):
            prepare(
                0,
                input_size=tuple(int(value) for value in det_size),
                det_thresh=float(det_thresh),
            )
        face_app.det_thresh = float(det_thresh)
        face_app.det_size = tuple(int(value) for value in det_size)
        cls._normalize_insightface_detector_state(
            det_model,
            det_size=det_size,
            det_thresh=float(det_thresh),
        )
        input_width, input_height = tuple(int(value) for value in det_model.input_size)
        num_anchors = int(getattr(det_model, "_num_anchors"))
        for stride in list(getattr(det_model, "_feat_stride_fpn", [])):
            cls._get_insightface_anchor_centers(
                det_model,
                height=input_height // int(stride),
                width=input_width // int(stride),
                stride=int(stride),
                num_anchors=num_anchors,
            )

    @staticmethod
    def _reshape_insightface_detector_output(
        output: Any,
        *,
        batch_size: int,
        items_per_image: int,
        channels: int,
        label: str,
    ) -> np.ndarray:
        array = np.asarray(output)
        expected_flat = items_per_image * batch_size
        if array.ndim == 3 and array.shape[0] == batch_size:
            return array.reshape(batch_size, items_per_image, channels)
        if array.ndim == 2 and array.shape[0] == expected_flat:
            return array.reshape(batch_size, items_per_image, channels)
        if array.ndim == 2 and batch_size == 1 and array.shape[0] == items_per_image:
            return array.reshape(1, items_per_image, channels)
        raise RuntimeError(
            "InsightFace detector output shape mismatch for "
            f"{label}: batch_size={batch_size} items_per_image={items_per_image} "
            f"channels={channels} got_shape={tuple(array.shape)}"
        )

    @staticmethod
    def _get_insightface_anchor_centers(
        det_model: Any,
        *,
        height: int,
        width: int,
        stride: int,
        num_anchors: int,
    ) -> np.ndarray:
        key = (height, width, stride)
        anchor_centers = det_model.center_cache.get(key)
        if anchor_centers is not None:
            return anchor_centers

        anchor_centers = np.stack(
            cast(Any, np.mgrid[:height, :width][::-1]),
            axis=-1,
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1)
            anchor_centers = anchor_centers.reshape((-1, 2))
        if len(det_model.center_cache) < 100:
            det_model.center_cache[key] = anchor_centers
        return anchor_centers

    def _run_insightface_detector_forward_batch(
        self,
        det_model: Any,
        det_images: List[np.ndarray],
    ) -> List[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        preprocess_runner = self._face_det_ppp
        if preprocess_runner is None:
            raise RuntimeError("InsightFace detection PPP runner is not initialized.")
        if not det_images:
            return []

        self._normalize_insightface_detector_state(
            det_model,
            det_size=self._to_int_pair(getattr(det_model, "input_size", (640, 640)), (640, 640)),
            det_thresh=float(getattr(det_model, "det_thresh", 0.5) or 0.5),
            nms_thresh=float(getattr(det_model, "nms_thresh", 0.4) or 0.4),
        )
        distance2bbox, distance2kps = self._resolve_insightface_detector_helpers(det_model)
        batch = np.ascontiguousarray(np.stack(det_images, axis=0), dtype=np.uint8)
        blob = preprocess_runner.run(batch)
        net_outs = det_model.session.run(det_model.output_names, {det_model.input_name: blob})

        batch_size = int(blob.shape[0])
        input_height = int(blob.shape[2])
        input_width = int(blob.shape[3])
        fmc = det_model.fmc
        feat_stride_fpn = list(getattr(det_model, "_feat_stride_fpn"))
        num_anchors = int(getattr(det_model, "_num_anchors"))
        per_image_scores: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
        per_image_bboxes: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
        per_image_kpss: List[List[np.ndarray]] = [[] for _ in range(batch_size)]

        for idx, stride in enumerate(feat_stride_fpn):
            height = input_height // stride
            width = input_width // stride
            item_count = height * width * num_anchors
            anchor_centers = self._get_insightface_anchor_centers(
                det_model,
                height=height,
                width=width,
                stride=stride,
                num_anchors=num_anchors,
            )
            score_batch = self._reshape_insightface_detector_output(
                net_outs[idx],
                batch_size=batch_size,
                items_per_image=item_count,
                channels=1,
                label=f"score@{stride}",
            )
            bbox_batch = self._reshape_insightface_detector_output(
                net_outs[idx + fmc],
                batch_size=batch_size,
                items_per_image=item_count,
                channels=4,
                label=f"bbox@{stride}",
            )

            if det_model.use_kps:
                if distance2kps is None:
                    raise RuntimeError(
                        "InsightFace detector module missing distance2kps for a "
                        "keypoint-enabled model."
                    )
                kps_batch = self._reshape_insightface_detector_output(
                    net_outs[idx + fmc * 2],
                    batch_size=batch_size,
                    items_per_image=item_count,
                    channels=10,
                    label=f"kps@{stride}",
                )
            else:
                kps_batch = None

            for batch_index in range(batch_size):
                scores = score_batch[batch_index].reshape(-1)
                bbox_preds = bbox_batch[batch_index] * stride
                pos_inds = np.where(scores >= det_model.det_thresh)[0]
                if pos_inds.size == 0:
                    continue
                bboxes = distance2bbox(anchor_centers, bbox_preds)
                per_image_scores[batch_index].append(scores[pos_inds])
                per_image_bboxes[batch_index].append(bboxes[pos_inds])

                if kps_batch is not None:
                    kps_preds = kps_batch[batch_index] * stride
                    kpss = distance2kps(anchor_centers, kps_preds)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    per_image_kpss[batch_index].append(kpss[pos_inds])

        return [
            (per_image_scores[index], per_image_bboxes[index], per_image_kpss[index])
            for index in range(batch_size)
        ]

    def _postprocess_insightface_detections(
        self,
        det_model: Any,
        *,
        scores_list: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        kpss_list: List[np.ndarray],
        det_scale: float,
        image_shape: Tuple[int, int],
        max_num: int,
        metric: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not scores_list:
            return np.empty((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        if scores_ravel.size == 0:
            return np.empty((0, 5), dtype=np.float32), None

        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if det_model.use_kps and kpss_list:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = det_model.nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image_shape[0] // 2, image_shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == "max" else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def _detect_faces_batch(
        self,
        det_model: Any,
        prepared_images: List[_PreparedInsightFaceImage],
        max_num: int = 0,
        metric: str = "default",
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        self._normalize_insightface_detector_state(
            det_model,
            det_size=self._to_int_pair(getattr(det_model, "input_size", (640, 640)), (640, 640)),
            det_thresh=float(getattr(det_model, "det_thresh", 0.5) or 0.5),
            nms_thresh=float(getattr(det_model, "nms_thresh", 0.4) or 0.4),
        )
        if not prepared_images:
            return []

        forward_outputs = self._run_insightface_detector_forward_batch(
            det_model,
            [item.detector_input for item in prepared_images],
        )
        results: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        for prepared, (scores_list, bboxes_list, kpss_list) in zip(prepared_images, forward_outputs):
            results.append(
                self._postprocess_insightface_detections(
                    det_model,
                    scores_list=scores_list,
                    bboxes_list=bboxes_list,
                    kpss_list=kpss_list,
                    det_scale=prepared.det_scale,
                    image_shape=prepared.image_shape,
                    max_num=max_num,
                    metric=metric,
                )
            )
        return results

    def _detect_faces(
        self,
        det_model: Any,
        img: np.ndarray,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        prepared = self._prepare_insightface_detection_input(img, det_model)
        return self._detect_faces_batch(
            det_model,
            [prepared],
            max_num=max_num,
            metric=metric,
        )[0]

    def _get_face_embeddings(self, rec_model: Any, aligned_faces: List[np.ndarray]) -> np.ndarray:
        preprocess_runner = self._face_rec_ppp
        if preprocess_runner is None:
            raise RuntimeError("InsightFace recognition PPP runner is not initialized.")
        if not aligned_faces:
            return np.empty((0, 0), dtype=np.float32)

        prepared = [
            _as_contiguous_bgr_uint8(np.asarray(item), context="InsightFace recognition")
            for item in aligned_faces
        ]
        batch = np.stack(prepared, axis=0)
        blob = preprocess_runner.run(batch)
        features = np.asarray(
            rec_model.session.run(rec_model.output_names, {rec_model.input_name: blob})[0]
        )
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] != len(prepared):
            raise RuntimeError(
                "InsightFace recognition output mismatch after batched run: "
                f"expected_batch={len(prepared)} got_shape={tuple(features.shape)}"
            )
        return features

    def _attach_insightface_preprocess(self, face_app: FaceAnalysis, device_name: str) -> None:
        det_model = getattr(face_app, "det_model", None)
        rec_model = getattr(face_app, "models", {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("InsightFace detection/recognition model not found.")

        det_width, det_height = map(int, det_model.input_size)
        rec_width, rec_height = map(int, rec_model.input_size)

        self._face_det_ppp = self._build_openvino_preprocess_runner(
            runner_name="insightface_det",
            device_name=device_name,
            output_height=det_height,
            output_width=det_width,
            mean_values=_to_channel_triplet(getattr(det_model, "input_mean", 127.5)),
            std_values=_to_channel_triplet(getattr(det_model, "input_std", 127.5)),
        )
        self._face_rec_ppp = self._build_openvino_preprocess_runner(
            runner_name="insightface_rec",
            device_name=device_name,
            output_height=rec_height,
            output_width=rec_width,
            mean_values=_to_channel_triplet(getattr(rec_model, "input_mean", 127.5)),
            std_values=_to_channel_triplet(getattr(rec_model, "input_std", 127.5)),
        )

        self._enable_insightface_opencl_pipeline()
        LOG.info(
            "InsightFace preprocessing configured: OpenCV(OpenCL resize/align) + "
            "OpenVINO PPP."
        )

    def _resolve_insightface_root(self) -> Path:
        candidate_roots = (self.insightface_model_root, self.insightface_root)
        for root in candidate_roots:
            if (root / MODEL_NAME).is_dir():
                return root
        raise FileNotFoundError(
            "InsightFace model directory missing for antelopev2. Checked paths: "
            f"{self.insightface_model_root / MODEL_NAME}, {self.insightface_root / MODEL_NAME}"
        )

    @staticmethod
    def _normalize_ort_shape(shape: Any) -> Tuple[Any, ...]:
        if isinstance(shape, (list, tuple)):
            return tuple(shape)
        return tuple()

    @classmethod
    def _shape_has_static_batch_one(cls, shape: Any) -> bool:
        normalized_shape = cls._normalize_ort_shape(shape)
        if not normalized_shape:
            return False
        first_dim = normalized_shape[0]
        try:
            return int(first_dim) == 1
        except (TypeError, ValueError):
            return False

    @classmethod
    def _format_ort_shape(cls, shape: Any) -> List[str]:
        return [
            "None" if item is None else str(item)
            for item in cls._normalize_ort_shape(shape)
        ]

    @classmethod
    def _normalize_insightface_recognition_state(cls, rec_model: Any) -> Dict[str, List[str]]:
        if rec_model is None:
            raise RuntimeError("InsightFace recognition model is missing.")

        session = getattr(rec_model, "session", None)
        if session is None or not hasattr(session, "get_inputs") or not hasattr(session, "get_outputs"):
            raise RuntimeError("InsightFace recognition session is missing ORT shape metadata.")

        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if not inputs or not outputs:
            raise RuntimeError("InsightFace recognition session returned empty inputs/outputs.")

        input_shape = cls._normalize_ort_shape(getattr(inputs[0], "shape", None))
        output_shape = cls._normalize_ort_shape(getattr(outputs[0], "shape", None))
        output_names = [str(item.name) for item in outputs if getattr(item, "name", None)]

        rec_model.input_name = str(getattr(rec_model, "input_name", None) or inputs[0].name)
        rec_model.output_names = output_names or list(getattr(rec_model, "output_names", []))
        rec_model.input_shape = list(input_shape)
        rec_model.output_shape = list(output_shape)

        if cls._shape_has_static_batch_one(output_shape):
            raise RuntimeError(
                "InsightFace recognition output metadata is pinned to batch=1. "
                "High-throughput batched recognition requires a dynamic output batch dimension."
            )

        return {
            "input_shape": cls._format_ort_shape(input_shape),
            "output_shape": cls._format_ort_shape(output_shape),
        }

    @staticmethod
    def _ensure_runtime_model_link(src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"InsightFace required model missing: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                raise RuntimeError(f"InsightFace runtime model path is a directory: {dst}")
            try:
                if os.path.samefile(src, dst):
                    return
            except Exception:
                pass
            src_stat = src.stat()
            dst_stat = dst.stat()
            if dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime_ns >= src_stat.st_mtime_ns:
                return
            dst.unlink()

        try:
            os.symlink(src, dst)
        except Exception:
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    @classmethod
    def _prepare_batched_insightface_detection_model(cls, src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"InsightFace required model missing: {src}")
        if onnx is None:
            raise RuntimeError(
                "InsightFace batched detection metadata patch requires the 'onnx' package."
            )

        if dst.is_file() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            try:
                existing_model = onnx.load(str(dst))
                existing_inputs = existing_model.graph.input
                if existing_inputs:
                    existing_dims = existing_inputs[0].type.tensor_type.shape.dim
                    existing_first_dim = existing_dims[0] if existing_dims else None
                    has_static_dim = bool(
                        getattr(existing_first_dim, "HasField", lambda *_: False)("dim_value")
                    )
                    if existing_first_dim is not None and not has_static_dim:
                        return
            except Exception:
                pass

        model = onnx.load(str(src))
        if not model.graph.input:
            raise RuntimeError(f"InsightFace detection model has no inputs: {src}")

        input_dims = model.graph.input[0].type.tensor_type.shape.dim
        if not input_dims:
            raise RuntimeError(
                "InsightFace detection model input shape metadata is missing. "
                f"model={src}"
            )
        first_input_dim = input_dims[0]
        if getattr(first_input_dim, "HasField", lambda *_: False)("dim_value"):
            first_input_dim.ClearField("dim_value")
        first_input_dim.dim_param = _INSIGHTFACE_BATCH_DIM_PARAM

        for output_index, output_value in enumerate(model.graph.output):
            output_dims = output_value.type.tensor_type.shape.dim
            if not output_dims:
                continue
            first_dim = output_dims[0]
            if getattr(first_dim, "HasField", lambda *_: False)("dim_value"):
                first_dim.ClearField("dim_value")
            first_dim.dim_param = f"anchors_{output_index}"

        dst.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dst.with_suffix(f"{dst.suffix}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        try:
            onnx.save(model, str(temp_path))
            os.replace(temp_path, dst)
            shutil.copystat(src, dst)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def _prepare_batched_insightface_recognition_model(cls, src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"InsightFace required model missing: {src}")
        if onnx is None:
            raise RuntimeError(
                "InsightFace batched recognition metadata patch requires the 'onnx' package."
            )

        if dst.is_file() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            try:
                existing_model = onnx.load(str(dst))
                existing_dims = existing_model.graph.output[0].type.tensor_type.shape.dim
                existing_first_dim = existing_dims[0] if existing_dims else None
                existing_has_static_dim = bool(
                    getattr(existing_first_dim, "HasField", lambda *_: False)("dim_value")
                )
                if existing_first_dim is not None and (
                    not existing_has_static_dim or int(existing_first_dim.dim_value) != 1
                ):
                    return
            except Exception:
                pass

        model = onnx.load(str(src))
        if not model.graph.output:
            raise RuntimeError(f"InsightFace recognition model has no outputs: {src}")
        output_dims = model.graph.output[0].type.tensor_type.shape.dim
        if not output_dims:
            raise RuntimeError(
                "InsightFace recognition model output shape metadata is missing. "
                f"model={src}"
            )

        first_dim = output_dims[0]
        has_static_dim = bool(getattr(first_dim, "HasField", lambda *_: False)("dim_value"))
        if not has_static_dim or int(first_dim.dim_value) != 1:
            cls._ensure_runtime_model_link(src, dst)
            return

        first_dim.ClearField("dim_value")
        first_dim.dim_param = _INSIGHTFACE_BATCH_DIM_PARAM

        dst.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dst.with_suffix(f"{dst.suffix}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        try:
            onnx.save(model, str(temp_path))
            os.replace(temp_path, dst)
            shutil.copystat(src, dst)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _prepare_insightface_runtime_root(self, source_root: Path) -> Path:
        source_model_dir = source_root / MODEL_NAME
        runtime_root = self.insightface_root / "_runtime_models"
        primary_model_dir = runtime_root / MODEL_NAME
        compat_model_dir = runtime_root / "models" / MODEL_NAME
        primary_model_dir.mkdir(parents=True, exist_ok=True)
        compat_model_dir.mkdir(parents=True, exist_ok=True)

        det_src = source_model_dir / "scrfd_10g_bnkps.onnx"
        rec_src = source_model_dir / "glintr100.onnx"
        det_primary = primary_model_dir / det_src.name
        rec_primary = primary_model_dir / rec_src.name

        self._prepare_batched_insightface_detection_model(det_src, det_primary)
        self._prepare_batched_insightface_recognition_model(rec_src, rec_primary)
        self._ensure_runtime_model_link(det_primary, compat_model_dir / det_src.name)
        self._ensure_runtime_model_link(rec_primary, compat_model_dir / rec_src.name)
        return runtime_root

    def _build_insightface_provider_options(self, provider_device: str) -> Dict[str, str]:
        provider_options: Dict[str, str] = {
            "device_type": provider_device,
            "enable_opencl_throttling": str(
                _as_bool(os.environ.get("INSIGHTFACE_OV_ENABLE_OPENCL_THROTTLING"), False)
            ).lower(),
        }
        if self.ov_cache_dir is not None:
            provider_options["cache_dir"] = str(self.ov_cache_dir)
        num_threads = _as_int(os.environ.get("INSIGHTFACE_OV_NUM_THREADS"), -1)
        if num_threads > 0:
            provider_options["num_of_threads"] = str(num_threads)
        return provider_options

    @staticmethod
    def _enforce_insightface_openvino_provider(
        face_app: FaceAnalysis,
        provider_options: Dict[str, str],
    ) -> None:
        models_map = getattr(face_app, "models", {})
        configured_sessions = 0
        for task_name, model in models_map.items():
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "set_providers"):
                continue
            try:
                session.set_providers(
                    ["OpenVINOExecutionProvider"],
                    [provider_options],
                )
            except TypeError:
                session.set_providers(
                    [("OpenVINOExecutionProvider", provider_options)]
                )
            except Exception as exc:
                raise RuntimeError(
                    "InsightFace failed to set OpenVINOExecutionProvider for task="
                    f"{task_name}. No silent fallback is allowed."
                ) from exc
            configured_sessions += 1

        if configured_sessions == 0:
            raise RuntimeError(
                "InsightFace session initialization missing; cannot enforce OpenVINOExecutionProvider."
            )

    @staticmethod
    def _is_insightface_init_kwargs_error(exc: Exception) -> bool:
        if not isinstance(exc, TypeError):
            return False
        message = str(exc)
        return any(
            keyword in message
            for keyword in ("providers", "provider_options", "allowed_modules")
        )

    @staticmethod
    def _strip_insightface_unsupported_init_kwargs(
        kwargs: Dict[str, Any],
        exc: Exception,
    ) -> Tuple[Dict[str, Any], List[str]]:
        message = str(exc)
        stripped_kwargs = dict(kwargs)
        stripped_keys: List[str] = []
        for key in ("provider_options", "providers", "allowed_modules"):
            if key in stripped_kwargs and key in message:
                stripped_kwargs.pop(key, None)
                stripped_keys.append(key)
        if stripped_keys:
            return stripped_kwargs, stripped_keys

        if any(keyword in message for keyword in ("providers", "provider_options", "allowed_modules")):
            for key in ("provider_options", "providers", "allowed_modules"):
                if key in stripped_kwargs:
                    stripped_kwargs.pop(key, None)
                    stripped_keys.append(key)
        return stripped_kwargs, stripped_keys

    @staticmethod
    def _build_insightface_init_attempt(
        runtime_root: Path,
        provider_names: List[str],
        provider_options: Dict[str, str],
        *,
        attempt_name: str,
    ) -> _InsightFaceInitAttempt:
        init_signature = inspect.signature(FaceAnalysis.__init__)
        init_parameters = init_signature.parameters
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in init_parameters.values()
        )
        supports_allowed_modules = "allowed_modules" in init_parameters
        supports_provider_kwargs = "providers" in init_parameters or supports_var_kwargs
        supports_provider_options = "provider_options" in init_parameters or supports_var_kwargs

        kwargs: Dict[str, Any] = {
            "name": MODEL_NAME,
            "root": str(runtime_root),
        }
        if supports_allowed_modules:
            kwargs["allowed_modules"] = ["detection", "recognition"]
        if supports_provider_kwargs:
            kwargs["providers"] = list(provider_names)
            if supports_provider_options:
                kwargs["provider_options"] = [dict(provider_options)]
        return _InsightFaceInitAttempt(attempt_name, runtime_root, kwargs)

    def _instantiate_insightface_face_analysis(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> Tuple[FaceAnalysis, Path]:
        source_model_root = self._resolve_insightface_root()
        source_init_root = (
            source_model_root.parent
            if source_model_root.name.lower() == "models"
            else source_model_root
        )
        runtime_root = self._prepare_insightface_runtime_root(source_model_root)
        pending_attempts: List[_InsightFaceInitAttempt] = [
            self._build_insightface_init_attempt(
                runtime_root,
                provider_names,
                provider_options,
                attempt_name="runtime-preferred",
            )
        ]
        source_fallback_added = False
        if source_init_root != runtime_root:
            source_fallback_added = True
            pending_attempts.append(
                self._build_insightface_init_attempt(
                    source_init_root,
                    provider_names,
                    provider_options,
                    attempt_name="source-preferred",
                )
            )

        seen_attempts: set[Tuple[str, str]] = set()
        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None
        while pending_attempts:
            attempt = pending_attempts.pop(0)
            attempt_key = (
                str(attempt.runtime_root),
                json.dumps(attempt.kwargs, sort_keys=True, ensure_ascii=True),
            )
            if attempt_key in seen_attempts:
                continue
            seen_attempts.add(attempt_key)
            try:
                face_app = FaceAnalysis(**attempt.kwargs)
                self._normalize_insightface_recognition_state(
                    getattr(face_app, "models", {}).get("recognition")
                )
                if attempt.name != "runtime-preferred":
                    LOG.warning(
                        "InsightFace initialized via compatibility path %s (runtime_root=%s)",
                        attempt.name,
                        attempt.runtime_root,
                    )
                return face_app, attempt.runtime_root
            except Exception as exc:
                last_exc = exc
                attempt_errors.append(f"{attempt.name}: {exc}")
                if self._is_insightface_init_kwargs_error(exc):
                    stripped_kwargs, stripped_keys = self._strip_insightface_unsupported_init_kwargs(
                        attempt.kwargs,
                        exc,
                    )
                    if stripped_keys and stripped_kwargs != attempt.kwargs:
                        LOG.warning(
                            "InsightFace init retry after unsupported kwargs on %s: removed=%s error=%s",
                            attempt.name,
                            ",".join(stripped_keys),
                            exc,
                        )
                        pending_attempts.insert(
                            0,
                            _InsightFaceInitAttempt(
                                f"{attempt.name}-compat",
                                attempt.runtime_root,
                                stripped_kwargs,
                            ),
                        )
                        continue

                LOG.warning(
                    "InsightFace init attempt %s failed (runtime_root=%s): %s",
                    attempt.name,
                    attempt.runtime_root,
                    exc,
                )
                if not source_fallback_added and attempt.runtime_root != source_init_root:
                    source_fallback_added = True
                    fallback_kwargs = dict(attempt.kwargs)
                    fallback_kwargs["root"] = str(source_init_root)
                    pending_attempts.append(
                        _InsightFaceInitAttempt(
                            "source-fallback",
                            source_init_root,
                            fallback_kwargs,
                        )
                    )

        summary = "; ".join(attempt_errors) or "no attempts executed"
        raise RuntimeError(
            "InsightFace initialization failed after compatibility retries: "
            f"{summary}"
        ) from last_exc

    def _initialize_loaded_insightface_runtime(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
        provider_device: str,
    ) -> _LoadedInsightFaceRuntime:
        face_app, runtime_root = self._instantiate_insightface_face_analysis(
            provider_names,
            provider_options,
        )
        self._prepare_insightface_detector(
            face_app,
            det_size=(640, 640),
            det_thresh=0.5,
        )
        self._enforce_insightface_openvino_provider(face_app, provider_options)
        rec_session_shapes = self._normalize_insightface_recognition_state(
            getattr(face_app, "models", {}).get("recognition")
        )
        provider_runtime = self._validate_insightface_openvino_provider(
            face_app,
            expected_device_type=provider_device,
        )
        self._attach_insightface_preprocess(face_app, device_name=provider_device)
        ppp_execution_devices = {
            "det": _get_compiled_model_execution_devices(
                getattr(self._face_det_ppp, "compiled_model", None)
            ),
            "rec": _get_compiled_model_execution_devices(
                getattr(self._face_rec_ppp, "compiled_model", None)
            ),
        }
        return _LoadedInsightFaceRuntime(
            face_app=face_app,
            runtime_root=runtime_root,
            provider_runtime=provider_runtime,
            ppp_execution_devices=ppp_execution_devices,
            rec_session_shapes=rec_session_shapes,
        )

    @staticmethod
    def _validate_insightface_openvino_provider(
        face_app: FaceAnalysis,
        expected_device_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        models_map = getattr(face_app, "models", {})
        runtime_state: Dict[str, Dict[str, Any]] = {}
        normalized_expected = str(expected_device_type or "").strip().upper()
        for task_name, model in models_map.items():
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                continue
            providers = [str(item) for item in session.get_providers()]
            if not providers or providers[0] != "OpenVINOExecutionProvider":
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}, providers={providers}. "
                    "OpenVINOExecutionProvider must be the active primary provider. "
                    "No silent fallback is allowed."
                )
            task_state: Dict[str, Any] = {"providers": providers}
            get_provider_options = getattr(session, "get_provider_options", None)
            if callable(get_provider_options):
                try:
                    provider_options = get_provider_options()
                except Exception:
                    provider_options = None
                if isinstance(provider_options, dict):
                    openvino_options = provider_options.get("OpenVINOExecutionProvider")
                    if isinstance(openvino_options, dict):
                        normalized_options = {
                            str(key): str(value) for key, value in openvino_options.items()
                        }
                        task_state["openvino_options"] = normalized_options
                        actual_device = str(
                            normalized_options.get("device_type", "")
                        ).strip().upper()
                        if normalized_expected and actual_device and actual_device != normalized_expected:
                            raise RuntimeError(
                                "InsightFace provider validation failed for task="
                                f"{task_name}, expected device_type={normalized_expected} "
                                f"but got {actual_device}. No silent fallback is allowed."
                            )
            runtime_state[task_name] = task_state
        return runtime_state

    def _unload_face_model_locked(self) -> None:
        self._face_engine = None
        self._face_det_ppp = None
        self._face_rec_ppp = None

    def _unload_face_model(self) -> None:
        with self._face_load_lock:
            self._unload_face_model_locked()

    def _ensure_face_loaded(self) -> None:
        with self._face_load_lock:
            if (
                self._face_engine is not None
                and self._face_det_ppp is not None
                and self._face_rec_ppp is not None
            ):
                return
            self._load_family_with_process_lock("face", self._load_face_locked)

    def _load_face_locked(self) -> None:
        configured_provider_device = _normalize_non_text_openvino_device(
            os.environ.get("INSIGHTFACE_OV_DEVICE", INFERENCE_DEVICE)
        )
        provider_device = _resolve_non_text_openvino_runtime_device(
            configured_provider_device,
            self.core.available_devices,
            consumer="ort_ep",
        )
        provider_options = self._build_insightface_provider_options(provider_device)
        provider_names = ["OpenVINOExecutionProvider"]
        try:
            runtime = self._initialize_loaded_insightface_runtime(
                provider_names,
                provider_options,
                provider_device,
            )
            self._face_engine = runtime.face_app
            LOG.info(
                "InsightFace loaded with providers=%s configured_device=%s runtime_device=%s provider_options=%s provider_runtime=%s ppp_execution_devices=%s rec_session_shapes=%s face_workers=%s face_batch=%s/%sms face_admission=%s (runtime_root=%s)",
                provider_names,
                configured_provider_device,
                provider_device,
                provider_options,
                runtime.provider_runtime,
                runtime.ppp_execution_devices,
                runtime.rec_session_shapes,
                self._face_worker_count,
                self._face_batch_size,
                int(self._face_batch_wait_seconds * 1000.0),
                self._face_admission.capacity,
                runtime.runtime_root,
            )
        except Exception as exc:
            self._face_engine = None
            self._face_det_ppp = None
            self._face_rec_ppp = None
            raise RuntimeError(
                "InsightFace must run with OpenVINOExecutionProvider + OpenCV OpenCL resize/alignment. "
                "No silent fallback is allowed."
            ) from exc

    @staticmethod
    def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _InferenceCancelled("Face task cancelled")

    @staticmethod
    def _set_face_task_result(task: _FaceInferenceTask, value: List[RepresentResult]) -> None:
        if not task.future.done():
            task.future.set_result(value)

    @staticmethod
    def _set_face_task_exception(task: _FaceInferenceTask, exc: Exception) -> None:
        if not task.future.done():
            task.future.set_exception(exc)

    def _submit_face_task(self, image: np.ndarray) -> _FaceInferenceTask:
        future: Future[List[RepresentResult]] = Future()
        task = _FaceInferenceTask(payload=image, future=future, created_at=time.time())
        if self._stopping:
            future.set_exception(RuntimeError("模型服务已关闭"))
            return task

        task_queue = self._face_task_queue
        if task_queue is None:
            future.set_exception(RuntimeError("InsightFace batch queue is not initialized."))
            return task

        try:
            task_queue.put(task, timeout=max(1.0, float(self._queue_timeout_seconds)))
        except queue.Full:
            self._set_face_task_exception(
                task,
                RuntimeError(f"推理队列已满（上限 {self._face_queue_capacity}），请稍后重试"),
            )
        except Exception as exc:
            submit_exc = exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc))
            self._set_face_task_exception(task, submit_exc)
        return task

    def _cancel_face_task_if_queued(self, task: _FaceInferenceTask, exc: Exception) -> bool:
        if task.started_event.is_set():
            return False
        task.cancel_requested.set()
        self._set_face_task_exception(task, exc)
        return True

    def _wait_face_task(self, task: _FaceInferenceTask) -> List[RepresentResult]:
        if task.future.done():
            return task.future.result()

        started = bool(task.started_event.wait(timeout=self._queue_timeout_seconds))
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_face_task_if_queued(task, queue_exc):
                raise queue_exc
            if not bool(task.started_event.wait(timeout=0.05)) and not task.future.done():
                raise queue_exc

        try:
            return task.future.result(timeout=self._execution_timeout_seconds)
        except FutureTimeoutError as exc:
            task.cancel_requested.set()
            try:
                task.future.result(timeout=max(1.0, float(self._execution_timeout_seconds)))
            except _InferenceCancelled:
                pass
            except Exception as cancel_exc:
                LOG.warning("Face task cancellation completed with error: %s", cancel_exc)
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    async def _await_face_task(self, task: _FaceInferenceTask) -> List[RepresentResult]:
        if task.future.done():
            return task.future.result()

        started = bool(await asyncio.to_thread(task.started_event.wait, self._queue_timeout_seconds))
        if not started:
            if task.future.done():
                return task.future.result()
            queue_exc = RuntimeError(f"推理任务排队超时（>{self._queue_timeout_seconds}s）")
            if self._cancel_face_task_if_queued(task, queue_exc):
                raise queue_exc
            started = bool(await asyncio.to_thread(task.started_event.wait, 0.05))
            if not started and not task.future.done():
                raise queue_exc

        wrapped = asyncio.wrap_future(task.future)
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped),
                timeout=self._execution_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            task.cancel_requested.set()
            try:
                await wrapped
            except _InferenceCancelled:
                pass
            except Exception as cancel_exc:
                LOG.warning("Face task cancellation completed with error: %s", cancel_exc)
            raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc

    def _collect_face_task_batch(
        self,
        first_task: _FaceInferenceTask,
    ) -> Tuple[List[_FaceInferenceTask], bool]:
        batch = [first_task]
        if self._face_batch_size <= 1:
            return batch, False

        task_queue = self._face_task_queue
        if task_queue is None:
            return batch, False

        deadline = time.monotonic() + self._face_batch_wait_seconds
        stop_after_batch = False
        while len(batch) < self._face_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                queued = task_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if queued is None:
                stop_after_batch = True
                break
            batch.append(queued)
        return batch, stop_after_batch

    def _face_batch_dispatcher_loop(self) -> None:
        task_queue = self._face_task_queue
        available_workers = self._face_available_workers
        if task_queue is None or available_workers is None:
            return

        while True:
            queued = task_queue.get()
            if queued is None:
                break
            batch, stop_after_batch = self._collect_face_task_batch(queued)
            active_batch: List[_FaceInferenceTask] = []
            for task in batch:
                if task.cancel_requested.is_set():
                    if not task.future.done():
                        self._set_face_task_exception(task, _InferenceCancelled("Face task cancelled"))
                    continue
                active_batch.append(task)
            if active_batch:
                worker_index = available_workers.get()
                self._face_worker_queues[worker_index].put(active_batch)
            if stop_after_batch:
                break

    def _face_batch_worker_loop(
        self,
        worker_index: int,
        worker_queue: "queue.Queue[Optional[List[_FaceInferenceTask]]]",
        available_workers: "queue.Queue[int]",
    ) -> None:
        available_workers.put(worker_index)
        while True:
            batch = worker_queue.get()
            if batch is None:
                break
            try:
                self._handle_face_task_batch(batch)
            finally:
                available_workers.put(worker_index)

    def _start_face_batch_service(self) -> None:
        if self._face_task_queue is not None:
            return

        self._face_task_queue = queue.Queue(maxsize=self._face_queue_capacity)
        self._face_available_workers = queue.Queue(maxsize=self._face_worker_count)
        self._face_worker_queues = []
        self._face_batch_workers = []
        for worker_index in range(self._face_worker_count):
            worker_queue: "queue.Queue[Optional[List[_FaceInferenceTask]]]" = queue.Queue(maxsize=1)
            worker_thread = threading.Thread(
                target=self._face_batch_worker_loop,
                args=(worker_index, worker_queue, self._face_available_workers),
                name=f"face-batch-{worker_index}",
                daemon=True,
            )
            self._face_worker_queues.append(worker_queue)
            self._face_batch_workers.append(worker_thread)
            worker_thread.start()

        self._face_batch_dispatcher = threading.Thread(
            target=self._face_batch_dispatcher_loop,
            name="face-batch-dispatch",
            daemon=True,
        )
        self._face_batch_dispatcher.start()
        self._face_batch_ready.set()

    def _stop_face_batch_service(self) -> List[_FaceInferenceTask]:
        pending: List[_FaceInferenceTask] = []
        task_queue = self._face_task_queue
        if task_queue is not None:
            while True:
                try:
                    queued = task_queue.get_nowait()
                except queue.Empty:
                    break
                if queued is None:
                    continue
                pending.append(queued)
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass

        dispatcher = self._face_batch_dispatcher
        if dispatcher is not None:
            dispatcher.join(timeout=max(2.0, float(self._execution_timeout_seconds)))
            self._face_batch_dispatcher = None

        for worker_queue in self._face_worker_queues:
            try:
                worker_queue.put_nowait(None)
            except Exception:
                pass
        for worker_thread in self._face_batch_workers:
            worker_thread.join(timeout=max(2.0, float(self._execution_timeout_seconds)))

        self._face_task_queue = None
        self._face_available_workers = None
        self._face_worker_queues = []
        self._face_batch_workers = []
        self._face_batch_ready.clear()
        return pending

    def _infer_face_batch(
        self,
        tasks: List[_FaceInferenceTask],
    ) -> List[Tuple[Optional[List[RepresentResult]], Optional[Exception]]]:
        if self._face_engine is None:
            raise RuntimeError("Face model is not loaded.")

        det_model = getattr(self._face_engine, "det_model", None)
        rec_model = getattr(self._face_engine, "models", {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("InsightFace detection/recognition model not found.")

        outcomes: List[Tuple[Optional[List[RepresentResult]], Optional[Exception]]] = [
            (None, None) for _ in tasks
        ]
        prepared_images: List[_PreparedInsightFaceImage] = []
        prepared_task_indices: List[int] = []
        for task_index, task in enumerate(tasks):
            if task.cancel_requested.is_set():
                outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                continue
            prepared_images.append(
                self._prepare_insightface_detection_input(task.payload, det_model)
            )
            prepared_task_indices.append(task_index)

        if not prepared_images:
            return outcomes

        detections_batch = self._detect_faces_batch(
            det_model,
            prepared_images,
            max_num=0,
            metric="default",
        )
        rec_image_size = int(rec_model.input_size[0])
        recognition_items: List[_PendingInsightFaceRecognition] = []
        for prepared_index, (detections, kpss) in enumerate(detections_batch):
            task_index = prepared_task_indices[prepared_index]
            task = tasks[task_index]
            if task.cancel_requested.is_set():
                outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                continue
            if detections.shape[0] == 0:
                outcomes[task_index] = ([], None)
                continue
            if kpss is None:
                outcomes[task_index] = (
                    None,
                    RuntimeError(
                        "InsightFace detection returned no landmarks for recognition. "
                        "No silent fallback is allowed."
                    ),
                )
                continue

            aligned_faces: List[np.ndarray] = []
            face_meta: List[Tuple[np.ndarray, float]] = []
            for face_index in range(detections.shape[0]):
                if task.cancel_requested.is_set():
                    outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                    break
                bbox = np.asarray(detections[face_index, 0:4], dtype=np.float32)
                det_score = float(detections[face_index, 4])
                aligned_faces.append(
                    self._align_face_opencl(
                        prepared_images[prepared_index].source_umat,
                        np.asarray(kpss[face_index]),
                        rec_image_size,
                    )
                )
                face_meta.append((bbox, det_score))
            if outcomes[task_index][1] is not None:
                continue
            recognition_items.append(
                _PendingInsightFaceRecognition(
                    task_index=task_index,
                    face_meta=face_meta,
                    aligned_faces=aligned_faces,
                )
            )

        all_aligned_faces: List[np.ndarray] = []
        recognition_spans: List[Tuple[int, List[Tuple[np.ndarray, float]], int, int]] = []
        for item in recognition_items:
            task = tasks[item.task_index]
            if task.cancel_requested.is_set():
                outcomes[item.task_index] = (None, _InferenceCancelled("Face task cancelled"))
                continue
            start = len(all_aligned_faces)
            all_aligned_faces.extend(item.aligned_faces)
            recognition_spans.append(
                (item.task_index, item.face_meta, start, len(item.aligned_faces))
            )

        if all_aligned_faces:
            embeddings = self._get_face_embeddings(rec_model, all_aligned_faces)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        for task_index, face_meta, start, count in recognition_spans:
            task = tasks[task_index]
            if task.cancel_requested.is_set():
                outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                continue
            task_embeddings = embeddings[start : start + count]
            if task_embeddings.shape[0] != len(face_meta):
                outcomes[task_index] = (
                    None,
                    RuntimeError(
                        "InsightFace recognition output mismatch: "
                        f"expected={len(face_meta)} got={task_embeddings.shape[0]}"
                    ),
                )
                continue
            results: List[RepresentResult] = []
            for index, (bbox, det_score) in enumerate(face_meta):
                if task.cancel_requested.is_set():
                    outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                    break
                embedding = np.asarray(task_embeddings[index], dtype=np.float32).reshape(-1)
                embedding_norm = float(np.linalg.norm(embedding))
                if embedding_norm <= 0.0:
                    outcomes[task_index] = (
                        None,
                        RuntimeError("InsightFace recognition produced zero-norm embedding."),
                    )
                    break
                normed_embedding = embedding / embedding_norm
                bbox_int = np.array(bbox).astype(int)
                x1, y1, x2, y2 = bbox_int
                results.append(
                    RepresentResult(
                        embedding=[float(value) for value in normed_embedding],
                        facial_area=FacialArea(
                            x=int(x1),
                            y=int(y1),
                            w=int(x2 - x1),
                            h=int(y2 - y1),
                        ),
                        face_confidence=float(det_score),
                    )
                )
            if outcomes[task_index][1] is None:
                outcomes[task_index] = (results, None)
        return outcomes

    def _handle_face_task_batch(self, tasks: List[_FaceInferenceTask]) -> None:
        if not tasks:
            return

        active_tasks: List[_FaceInferenceTask] = []
        for task in tasks:
            if task.cancel_requested.is_set():
                if not task.future.done():
                    self._set_face_task_exception(task, _InferenceCancelled("Face task cancelled"))
                continue
            task.started_at = time.monotonic()
            task.started_event.set()
            active_tasks.append(task)

        if not active_tasks:
            return

        try:
            outcomes = self._infer_face_batch(active_tasks)
            for task, (result, exc) in zip(active_tasks, outcomes):
                if exc is not None:
                    self._set_face_task_exception(task, exc)
                else:
                    self._set_face_task_result(task, result or [])
        except Exception as exc:
            LOG.error("Task represent failed: %s", exc, exc_info=True)
            for task in active_tasks:
                if task.cancel_requested.is_set():
                    self._set_face_task_exception(task, _InferenceCancelled("Face task cancelled"))
                else:
                    self._set_face_task_exception(task, exc)

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        with self._non_text_request_scope(
            family="face",
            label="InsightFace",
            admission=self._face_admission,
            ensure_loaded=self._ensure_face_loaded,
        ):
            task = self._submit_face_task(_as_contiguous_bgr_uint8(image, context="InsightFace"))
            return self._wait_face_task(task)

    async def get_face_representation_async(self, image: np.ndarray) -> List[RepresentResult]:
        async with self._non_text_request_scope_async(
            family="face",
            label="InsightFace",
            admission=self._face_admission,
            ensure_loaded=self._ensure_face_loaded,
        ):
            task = await asyncio.to_thread(
                self._submit_face_task,
                _as_contiguous_bgr_uint8(image, context="InsightFace"),
            )
            return await self._await_face_task(task)
