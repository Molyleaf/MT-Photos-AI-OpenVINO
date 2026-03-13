import asyncio
import inspect
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnx
from insightface.app import FaceAnalysis

if __package__ == "models":
    from schemas import FacialArea, RepresentResult
else:
    from ..schemas import FacialArea, RepresentResult

from .common import (
    _AdmissionController,
    _InferenceCancelled,
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


class InsightFaceMixin:
    core: Any
    ov_cache_dir: Optional[Path]
    insightface_root: Path
    insightface_model_root: Path
    _face_engine: Optional[FaceAnalysis]
    _face_det_ppp: Optional[_OpenVinoPreprocessRunner]
    _face_rec_ppp: Optional[_OpenVinoPreprocessRunner]
    _face_load_lock: Any
    _face_executor: ThreadPoolExecutor
    _face_admission: _AdmissionController
    _execution_timeout_seconds: int

    @staticmethod
    def _enable_insightface_opencl_alignment() -> None:
        name, vendor = _ensure_intel_opencl_device("InsightFace alignment")
        LOG.info(
            "OpenCV OpenCL enabled for InsightFace alignment on Intel device: %s (%s).",
            name,
            vendor,
        )

    @staticmethod
    def _align_face_opencl(
        img: np.ndarray,
        landmark: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        matrix = _estimate_insightface_norm_matrix(landmark=landmark, image_size=int(image_size))
        if matrix.shape != (2, 3):
            raise RuntimeError(
                "InsightFace estimate_norm returned invalid affine matrix shape: "
                f"{matrix.shape}"
            )
        source = _as_contiguous_bgr_uint8(img, context="InsightFace alignment")

        if not cv2.ocl.useOpenCL():
            raise RuntimeError(
                "OpenCV OpenCL was disabled during InsightFace alignment. "
                "No silent fallback is allowed."
            )
        try:
            warped_umat = cv2.warpAffine(
                _to_opencv_umat(source),
                matrix,
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

    def _run_insightface_detector_forward(self, det_model: Any, img: np.ndarray) -> Any:
        preprocess_runner = self._face_det_ppp
        if preprocess_runner is None:
            raise RuntimeError("InsightFace detection PPP runner is not initialized.")

        self._normalize_insightface_detector_state(
            det_model,
            det_size=tuple(int(value) for value in getattr(det_model, "input_size", (640, 640))),
            det_thresh=float(getattr(det_model, "det_thresh", 0.5) or 0.5),
            nms_thresh=float(getattr(det_model, "nms_thresh", 0.4) or 0.4),
        )
        distance2bbox, distance2kps = self._resolve_insightface_detector_helpers(det_model)
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        image_bgr = _as_contiguous_bgr_uint8(np.asarray(img), context="InsightFace detector")
        blob = preprocess_runner.run(image_bgr[np.newaxis, ...])
        net_outs = det_model.session.run(det_model.output_names, {det_model.input_name: blob})

        input_height = int(blob.shape[2])
        input_width = int(blob.shape[3])
        fmc = det_model.fmc
        batched_output = bool(getattr(det_model, "batched", False))

        for idx, stride in enumerate(det_model._feat_stride_fpn):
            kps_preds: Optional[np.ndarray] = None
            if batched_output:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                if det_model.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                if det_model.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in det_model.center_cache:
                anchor_centers = det_model.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if det_model._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * det_model._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(det_model.center_cache) < 100:
                    det_model.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= det_model.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if det_model.use_kps:
                if kps_preds is None or distance2kps is None:
                    raise RuntimeError(
                        "InsightFace detector returned no keypoint predictions "
                        "for a keypoint-enabled model."
                    )
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def _detect_faces(
        self,
        det_model: Any,
        img: np.ndarray,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._normalize_insightface_detector_state(
            det_model,
            det_size=tuple(int(value) for value in getattr(det_model, "input_size", (640, 640))),
            det_thresh=float(getattr(det_model, "det_thresh", 0.5) or 0.5),
            nms_thresh=float(getattr(det_model, "nms_thresh", 0.4) or 0.4),
        )
        assert det_model.input_size is not None
        input_size = tuple(det_model.input_size)
        image = _as_contiguous_bgr_uint8(img, context="InsightFace detector")
        im_ratio = float(image.shape[0]) / float(image.shape[1])
        model_ratio = float(input_size[1]) / float(input_size[0])
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / float(image.shape[0])
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self._run_insightface_detector_forward(det_model, det_img)
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
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
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

        self._enable_insightface_opencl_alignment()
        LOG.info("InsightFace preprocessing configured: OpenCV(OpenCL align) + OpenVINO PPP.")

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
    def _prepare_batched_insightface_recognition_model(cls, src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"InsightFace required model missing: {src}")
        if onnx is None:
            raise RuntimeError(
                "InsightFace batched recognition metadata patch requires the 'onnx' package."
            )

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

        self._ensure_runtime_model_link(det_src, det_primary)
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
    def _build_insightface_init_attempts(
        source_root: Path,
        runtime_root: Path,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> List[_InsightFaceInitAttempt]:
        init_signature = inspect.signature(FaceAnalysis.__init__)
        init_parameters = init_signature.parameters
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in init_parameters.values()
        )
        supports_allowed_modules = "allowed_modules" in init_parameters
        supports_provider_kwargs = "providers" in init_parameters or supports_var_kwargs
        supports_provider_options = "provider_options" in init_parameters or supports_var_kwargs

        def _build_kwargs(
            root: Path,
            *,
            include_provider_kwargs: bool,
            include_allowed_modules: bool,
        ) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "name": MODEL_NAME,
                "root": str(root),
            }
            if include_allowed_modules:
                kwargs["allowed_modules"] = ["detection", "recognition"]
            if include_provider_kwargs:
                kwargs["providers"] = list(provider_names)
                if supports_provider_options:
                    kwargs["provider_options"] = [dict(provider_options)]
            return kwargs

        return [
            _InsightFaceInitAttempt(
                "runtime-with-provider-kwargs",
                runtime_root,
                _build_kwargs(
                    runtime_root,
                    include_provider_kwargs=supports_provider_kwargs,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            _InsightFaceInitAttempt(
                "runtime-without-provider-kwargs",
                runtime_root,
                _build_kwargs(
                    runtime_root,
                    include_provider_kwargs=False,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            _InsightFaceInitAttempt(
                "source-with-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=supports_provider_kwargs,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            _InsightFaceInitAttempt(
                "source-without-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=False,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            _InsightFaceInitAttempt(
                "legacy-with-provider-kwargs",
                runtime_root,
                _build_kwargs(
                    runtime_root,
                    include_provider_kwargs=supports_provider_kwargs,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            _InsightFaceInitAttempt(
                "legacy-minimal-kwargs",
                runtime_root,
                _build_kwargs(
                    runtime_root,
                    include_provider_kwargs=False,
                    include_allowed_modules=False,
                ),
            ),
        ]

    def _instantiate_insightface_face_analysis(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> Tuple[FaceAnalysis, Path]:
        source_root = self._resolve_insightface_root()
        runtime_root = self._prepare_insightface_runtime_root(source_root)
        attempts = self._build_insightface_init_attempts(
            source_root,
            runtime_root,
            provider_names,
            provider_options,
        )

        seen_attempts: set[Tuple[str, str]] = set()
        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None
        for attempt in attempts:
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
                if attempt.name not in {
                    "runtime-with-provider-kwargs",
                    "source-with-provider-kwargs",
                }:
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
                    LOG.warning(
                        "InsightFace init retry after unsupported kwargs on %s: %s",
                        attempt.name,
                        exc,
                    )
                else:
                    LOG.warning(
                        "InsightFace init attempt %s failed (runtime_root=%s): %s",
                        attempt.name,
                        attempt.runtime_root,
                        exc,
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
        self._enforce_insightface_openvino_provider(face_app, provider_options)
        face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        self._enforce_insightface_openvino_provider(face_app, provider_options)
        self._normalize_insightface_detector_state(
            getattr(face_app, "det_model"),
            det_size=(640, 640),
            det_thresh=float(getattr(face_app, "det_thresh", 0.5) or 0.5),
        )
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
            if self._face_engine is not None:
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
                "InsightFace loaded with providers=%s configured_device=%s runtime_device=%s provider_options=%s provider_runtime=%s ppp_execution_devices=%s rec_session_shapes=%s face_workers=%s face_admission=%s (runtime_root=%s)",
                provider_names,
                configured_provider_device,
                provider_device,
                provider_options,
                runtime.provider_runtime,
                runtime.ppp_execution_devices,
                runtime.rec_session_shapes,
                getattr(self._face_executor, "_max_workers", "unknown"),
                self._face_admission.capacity,
                runtime.runtime_root,
            )
        except Exception as exc:
            self._face_engine = None
            self._face_det_ppp = None
            self._face_rec_ppp = None
            raise RuntimeError(
                "InsightFace must run with OpenVINOExecutionProvider + OpenCV OpenCL alignment. "
                "No silent fallback is allowed."
            ) from exc

    @staticmethod
    def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _InferenceCancelled("Face task cancelled")

    def _infer_face(
        self,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[RepresentResult]:
        if self._face_engine is None:
            raise RuntimeError("Face model is not loaded.")

        det_model = getattr(self._face_engine, "det_model", None)
        rec_model = getattr(self._face_engine, "models", {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("InsightFace detection/recognition model not found.")

        face_input = _as_contiguous_bgr_uint8(image, context="InsightFace")
        self._raise_if_cancelled(cancel_event)
        detections, kpss = self._detect_faces(det_model, face_input, max_num=0, metric="default")
        if detections.shape[0] == 0:
            return []

        rec_image_size = int(rec_model.input_size[0])
        aligned_faces: List[np.ndarray] = []
        face_meta: List[Tuple[np.ndarray, float]] = []
        for index in range(detections.shape[0]):
            self._raise_if_cancelled(cancel_event)
            bbox = np.asarray(detections[index, 0:4], dtype=np.float32)
            det_score = float(detections[index, 4])
            if kpss is None:
                raise RuntimeError(
                    "InsightFace detection returned no landmarks for recognition. "
                    "No silent fallback is allowed."
                )
            aligned_faces.append(
                self._align_face_opencl(face_input, np.asarray(kpss[index]), rec_image_size)
            )
            face_meta.append((bbox, det_score))

        self._raise_if_cancelled(cancel_event)
        embeddings = self._get_face_embeddings(rec_model, aligned_faces)
        if embeddings.shape[0] != len(face_meta):
            raise RuntimeError(
                "InsightFace recognition output mismatch: "
                f"expected={len(face_meta)} got={embeddings.shape[0]}"
            )

        results: List[RepresentResult] = []
        for index, (bbox, det_score) in enumerate(face_meta):
            self._raise_if_cancelled(cancel_event)
            embedding = np.asarray(embeddings[index], dtype=np.float32).reshape(-1)
            embedding_norm = float(np.linalg.norm(embedding))
            if embedding_norm <= 0.0:
                raise RuntimeError("InsightFace recognition produced zero-norm embedding.")
            normed_embedding = embedding / embedding_norm
            bbox = np.array(bbox).astype(int)
            x1, y1, x2, y2 = bbox
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
        return results

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        with self._non_text_request_scope(
            family="face",
            label="InsightFace",
            admission=self._face_admission,
            ensure_loaded=self._ensure_face_loaded,
        ):
            return self._infer_face(image)

    async def get_face_representation_async(self, image: np.ndarray) -> List[RepresentResult]:
        cancel_event = threading.Event()
        async with self._non_text_request_scope_async(
            family="face",
            label="InsightFace",
            admission=self._face_admission,
            ensure_loaded=self._ensure_face_loaded,
        ):
            future = self._run_in_executor(
                self._face_executor,
                self._infer_face,
                image,
                cancel_event,
            )
            return await self._await_with_timeout_and_cooperative_cancel(
                future,
                cancel_event=cancel_event,
                timeout_seconds=self._execution_timeout_seconds,
                task_name="Face task",
            )
