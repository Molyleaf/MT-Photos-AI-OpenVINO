import asyncio
import os
import shutil
import threading
import time
from abc import abstractmethod, ABC
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import onnx
from insightface.app import FaceAnalysis

from .common import (
    _AdmissionController,
    _FaceInferenceTask,
    _InferenceCancelled,
    NonTextFamily,
    _as_bool,
    _as_contiguous_bgr_uint8,
    _as_int,
    _normalize_non_text_openvino_device,
    _resolve_non_text_openvino_runtime_device,
)
from .constants import INFERENCE_DEVICE, INSIGHTFACE_SINGLE_LANE, LOG, MODEL_NAME
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
class _LoadedInsightFaceRuntime:
    face_app: FaceAnalysis
    runtime_root: Path
    provider_runtime: Dict[str, Dict[str, Any]]
    det_session_shapes: Dict[str, Any]
    rec_session_shapes: Dict[str, Any]


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
    _face_load_lock: Any
    _face_admission: _AdmissionController
    _face_preprocess_worker_count: int
    _face_batch_size: int
    _face_batch_wait_seconds: float
    _face_queue_capacity: int
    _face_dispatch_loop: Optional[asyncio.AbstractEventLoop]
    _face_task_queue: Optional[asyncio.Queue[Optional[_FaceInferenceTask]]]
    _face_loop_ready: Any
    _face_worker: Optional[threading.Thread]
    _face_preprocess_executor: ThreadPoolExecutor
    _face_preprocess_device: Optional[str]
    _execution_timeout_seconds: int

    @abstractmethod
    def _load_family_serialized(self, family: NonTextFamily, loader: Any) -> None:
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

    def _configure_insightface_preprocess_backend(self, device_name: str) -> None:
        normalized_device = str(device_name or "").strip().upper()
        self._face_preprocess_device = "CPU"
        LOG.info(
            "InsightFace preprocessing backend: provider_device=%s detection=insightface-native(CPU) alignment=local-OpenCV(CPU) recognition_blob=insightface-native(CPU)",
            normalized_device or "CPU",
        )

    def _align_face(
        self,
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

        source = _as_contiguous_bgr_uint8(img, context="InsightFace alignment")
        try:
            aligned = cv2.warpAffine(
                source,
                cast(Any, matrix),
                (int(image_size), int(image_size)),
                flags=cv2.INTER_LINEAR,
                borderValue=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenCV CPU warpAffine failed in InsightFace CPU alignment."
            ) from exc

        return _as_contiguous_bgr_uint8(aligned, context="InsightFace alignment")

    @staticmethod
    def _prepare_insightface_models(
        face_app: FaceAnalysis,
        *,
        det_size: Tuple[int, int],
        det_thresh: float,
    ) -> None:
        det_model = getattr(face_app, "det_model", None)
        rec_model = getattr(face_app, "models", {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("InsightFace detection/recognition model not found.")

        prepare_det = getattr(det_model, "prepare", None)
        if callable(prepare_det):
            prepare_det(
                0,
                input_size=tuple(int(value) for value in det_size),
                det_thresh=float(det_thresh),
            )
        prepare_rec = getattr(rec_model, "prepare", None)
        if callable(prepare_rec):
            prepare_rec(0)

        if not hasattr(det_model, "center_cache") or getattr(det_model, "center_cache", None) is None:
            det_model.center_cache = {}
        if getattr(det_model, "input_size", None) is None:
            det_model.input_size = tuple(int(value) for value in det_size)
        if getattr(det_model, "det_thresh", None) is None:
            det_model.det_thresh = float(det_thresh)

        face_app.det_thresh = float(det_thresh)
        face_app.det_size = tuple(int(value) for value in det_size)

    def _detect_faces_native(
        self,
        det_model: Any,
        img: np.ndarray,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image = _as_contiguous_bgr_uint8(img, context="InsightFace detector")
        try:
            detections, kpss = det_model.detect(image, max_num=max_num, metric=metric)
        except Exception as exc:
            raise RuntimeError("InsightFace native detector.detect() failed.") from exc
        return (
            np.asarray(detections, dtype=np.float32),
            None if kpss is None else np.asarray(kpss, dtype=np.float32),
        )

    def _get_face_embeddings_native(self, rec_model: Any, aligned_faces: List[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            return np.empty((0, 0), dtype=np.float32)

        prepared = [
            _as_contiguous_bgr_uint8(np.asarray(item), context="InsightFace recognition")
            for item in aligned_faces
        ]
        try:
            features = np.asarray(rec_model.get_feat(prepared))
        except Exception as exc:
            raise RuntimeError("InsightFace native recognition get_feat() failed.") from exc
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] != len(prepared):
            raise RuntimeError(
                "InsightFace recognition output mismatch after batched run: "
                f"expected_batch={len(prepared)} got_shape={tuple(features.shape)}"
            )
        return np.asarray(features, dtype=np.float32)

    def _resolve_insightface_source_model_dir(self) -> Path:
        model_dir = self.insightface_model_root / MODEL_NAME
        if model_dir.is_dir():
            return model_dir
        raise FileNotFoundError(
            "InsightFace model directory missing for antelopev2. Expected path: "
            f"{model_dir}"
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

    @staticmethod
    def _onnx_dim_has_value(dim: Any) -> bool:
        has_field = getattr(dim, "HasField", None)
        if callable(has_field):
            return bool(has_field("dim_value"))
        try:
            int(getattr(dim, "dim_value"))
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def _onnx_dim_value(cls, dim: Any) -> Optional[int]:
        if not cls._onnx_dim_has_value(dim):
            return None
        return int(dim.dim_value)

    @classmethod
    def _save_patched_insightface_model(cls, model: Any, src: Path, dst: Path) -> None:
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
    def _insightface_recognition_model_has_dynamic_output_batch(cls, model: Any) -> bool:
        if not model.graph.output:
            return False
        output_dims = model.graph.output[0].type.tensor_type.shape.dim
        return bool(output_dims) and cls._onnx_dim_value(output_dims[0]) is None

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

    @classmethod
    def _normalize_insightface_detector_session_state(cls, det_model: Any) -> Dict[str, Any]:
        if det_model is None:
            raise RuntimeError("InsightFace detection model is missing.")

        session = getattr(det_model, "session", None)
        if session is None or not hasattr(session, "get_inputs") or not hasattr(session, "get_outputs"):
            raise RuntimeError("InsightFace detection session is missing ORT shape metadata.")

        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if not inputs or not outputs:
            raise RuntimeError("InsightFace detection session returned empty inputs/outputs.")

        input_shape = cls._normalize_ort_shape(getattr(inputs[0], "shape", None))
        output_shapes = [
            cls._normalize_ort_shape(getattr(output_meta, "shape", None))
            for output_meta in outputs
        ]
        det_model.input_name = str(getattr(det_model, "input_name", None) or inputs[0].name)
        det_model.output_names = [
            str(item.name) for item in outputs if getattr(item, "name", None)
        ] or list(getattr(det_model, "output_names", []))
        det_model.input_shape = list(input_shape)
        det_model.output_shapes = [list(shape) for shape in output_shapes]

        return {
            "input_shape": cls._format_ort_shape(input_shape),
            "output_shapes": [cls._format_ort_shape(shape) for shape in output_shapes],
        }

    @staticmethod
    def _copy_runtime_insightface_model(
        src: Path,
        dst: Path,
    ) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"InsightFace required model missing: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dst.with_suffix(f"{dst.suffix}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        try:
            shutil.copyfile(src, temp_path)
            shutil.copystat(src, temp_path)
            os.replace(temp_path, dst)
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
                if cls._insightface_recognition_model_has_dynamic_output_batch(existing_model):
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
        if cls._onnx_dim_value(first_dim) is None:
            cls._copy_runtime_insightface_model(src, dst)
            return

        first_dim.ClearField("dim_value")
        first_dim.dim_param = _INSIGHTFACE_BATCH_DIM_PARAM

        cls._save_patched_insightface_model(model, src, dst)

    def _prepare_insightface_runtime_root(self, source_model_dir: Path) -> Path:
        runtime_root = self.insightface_root / "_runtime_models"
        runtime_model_dir = runtime_root / "models" / MODEL_NAME
        legacy_model_dir = runtime_root / MODEL_NAME
        if legacy_model_dir.exists():
            if legacy_model_dir.is_dir():
                shutil.rmtree(legacy_model_dir)
            else:
                legacy_model_dir.unlink()
        runtime_model_dir.mkdir(parents=True, exist_ok=True)

        det_src = source_model_dir / "scrfd_10g_bnkps.onnx"
        rec_src = source_model_dir / "glintr100.onnx"
        det_runtime = runtime_model_dir / det_src.name
        rec_runtime = runtime_model_dir / rec_src.name

        self._copy_runtime_insightface_model(det_src, det_runtime)
        self._prepare_batched_insightface_recognition_model(rec_src, rec_runtime)
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
    def _validate_insightface_loaded_modules(face_app: FaceAnalysis) -> None:
        models_map = getattr(face_app, "models", {})
        loaded_tasks = {str(task_name) for task_name in models_map.keys()}
        required_tasks = {"detection", "recognition"}
        missing_tasks = sorted(required_tasks - loaded_tasks)
        unexpected_tasks = sorted(loaded_tasks - required_tasks)
        if missing_tasks or unexpected_tasks:
            raise RuntimeError(
                "InsightFace loaded unexpected module set under strict initialization: "
                f"missing={missing_tasks or ['none']} unexpected={unexpected_tasks or ['none']}"
            )

    def _instantiate_insightface_face_analysis(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> Tuple[FaceAnalysis, Path]:
        source_model_dir = self._resolve_insightface_source_model_dir()
        runtime_root = self._prepare_insightface_runtime_root(source_model_dir)
        init_kwargs: Dict[str, Any] = {
            "name": MODEL_NAME,
            "root": str(runtime_root),
            "allowed_modules": ["detection", "recognition"],
            "providers": list(provider_names),
            "provider_options": [dict(provider_options)],
        }
        try:
            face_app = FaceAnalysis(**init_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Installed insightface does not support strict "
                "FaceAnalysis(name=..., root=..., allowed_modules=..., providers=..., "
                "provider_options=...) initialization. Compatibility retries are disabled."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"InsightFace strict initialization failed for runtime_root={runtime_root}: {exc}"
            ) from exc

        self._validate_insightface_loaded_modules(face_app)
        self._normalize_insightface_recognition_state(
            getattr(face_app, "models", {}).get("recognition")
        )
        return face_app, runtime_root

    def _initialize_loaded_insightface_runtime(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
        provider_device: str,
    ) -> _LoadedInsightFaceRuntime:
        face_app: Optional[FaceAnalysis] = None
        try:
            face_app, runtime_root = self._instantiate_insightface_face_analysis(
                provider_names,
                provider_options,
            )
            det_session_shapes = self._normalize_insightface_detector_session_state(
                getattr(face_app, "det_model", None)
            )
            self._prepare_insightface_models(
                face_app,
                det_size=(640, 640),
                det_thresh=0.5,
            )
            self._disable_insightface_session_fallback(face_app)
            rec_session_shapes = self._normalize_insightface_recognition_state(
                getattr(face_app, "models", {}).get("recognition")
            )
            provider_runtime = self._validate_insightface_openvino_provider(
                face_app,
                expected_device_type=provider_device,
            )
            self._configure_insightface_preprocess_backend(provider_device)
            return _LoadedInsightFaceRuntime(
                face_app=face_app,
                runtime_root=runtime_root,
                provider_runtime=provider_runtime,
                det_session_shapes=det_session_shapes,
                rec_session_shapes=rec_session_shapes,
            )
        except Exception:
            self._dispose_insightface_face_analysis(face_app)
            self._face_preprocess_device = None
            raise

    @staticmethod
    def _disable_insightface_session_fallback(face_app: FaceAnalysis) -> None:
        models_map = getattr(face_app, "models", {})
        for task_name in ("detection", "recognition"):
            model = models_map.get(task_name)
            if model is None:
                raise RuntimeError(
                    f"InsightFace fallback control failed: missing task={task_name}."
                )
            session = getattr(model, "session", None)
            disable_fallback = getattr(session, "disable_fallback", None)
            if session is None or not callable(disable_fallback):
                raise RuntimeError(
                    "InsightFace fallback control failed for task="
                    f"{task_name}: ORT session does not expose disable_fallback()."
                )
            disable_fallback()
            if getattr(session, "_enable_fallback", False):
                raise RuntimeError(
                    "InsightFace fallback control failed for task="
                    f"{task_name}: ORT session fallback is still enabled."
                )

    @staticmethod
    def _validate_insightface_openvino_provider(
        face_app: FaceAnalysis,
        expected_device_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        models_map = getattr(face_app, "models", {})
        runtime_state: Dict[str, Dict[str, Any]] = {}
        required_tasks = {"detection", "recognition"}
        normalized_expected = str(expected_device_type or "").strip().upper()
        for task_name in sorted(required_tasks):
            model = models_map.get(task_name)
            if model is None:
                raise RuntimeError(
                    f"InsightFace provider validation failed: missing required task={task_name}."
                )
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                raise RuntimeError(
                    f"InsightFace provider validation failed: task={task_name} has no ORT session."
                )
            providers = [str(item) for item in session.get_providers()]
            if not providers or providers[0] != "OpenVINOExecutionProvider":
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}, providers={providers}. "
                    "OpenVINOExecutionProvider must be the active primary provider. "
                    "No silent fallback is allowed."
                )
            task_state: Dict[str, Any] = {"providers": providers}
            task_state["session_run_fallback_disabled"] = (
                getattr(session, "_enable_fallback", None) is False
            )
            get_provider_options = getattr(session, "get_provider_options", None)
            if not callable(get_provider_options):
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}: ORT session does not expose provider options."
                )
            try:
                provider_options = get_provider_options()
            except Exception as exc:
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}: unable to read provider options."
                ) from exc
            if not isinstance(provider_options, dict):
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}: provider options are missing."
                )

            openvino_options = provider_options.get("OpenVINOExecutionProvider")
            if not isinstance(openvino_options, dict):
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}: OpenVINOExecutionProvider options are missing."
                )

            normalized_options = {
                str(key): str(value) for key, value in openvino_options.items()
            }
            task_state["openvino_options"] = normalized_options
            actual_device = str(
                normalized_options.get("device_type", "")
            ).strip().upper()
            task_state["reported_device_type"] = actual_device or None
            task_state["expected_device_type"] = normalized_expected or None
            if normalized_expected and actual_device and actual_device != normalized_expected:
                raise RuntimeError(
                    "InsightFace provider validation failed for task="
                    f"{task_name}, expected device_type={normalized_expected} "
                    f"but got {actual_device}. No silent fallback is allowed."
                )
            runtime_state[task_name] = task_state
        return runtime_state

    @staticmethod
    def _dispose_insightface_face_analysis(face_app: Optional[FaceAnalysis]) -> None:
        if face_app is None:
            return

        det_model = getattr(face_app, "det_model", None)
        if det_model is not None and isinstance(getattr(det_model, "center_cache", None), dict):
            det_model.center_cache.clear()

        models_map = getattr(face_app, "models", None)
        if isinstance(models_map, dict):
            for model in list(models_map.values()):
                if isinstance(getattr(model, "center_cache", None), dict):
                    model.center_cache.clear()
                if hasattr(model, "session"):
                    try:
                        model.session = None
                    except Exception:
                        pass
            models_map.clear()

        try:
            face_app.det_model = None
        except Exception:
            pass

    def _unload_face_model_locked(self) -> None:
        self._dispose_insightface_face_analysis(self._face_engine)
        self._face_engine = None
        self._face_preprocess_device = None

    def _unload_face_model(self) -> None:
        with self._face_load_lock:
            self._unload_face_model_locked()

    def _ensure_face_loaded(self) -> None:
        with self._face_load_lock:
            if self._face_engine is not None:
                return
            self._load_family_serialized("face", self._load_face_locked)

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
                "InsightFace loaded with providers=%s configured_device=%s runtime_device=%s preprocess_device=%s provider_options=%s provider_runtime=%s det_session_shapes=%s rec_session_shapes=%s face_lane=%s face_preprocess_workers=%s face_batch=%s/%sms face_admission=%s pipeline=%s (runtime_root=%s)",
                provider_names,
                configured_provider_device,
                provider_device,
                self._face_preprocess_device or "CPU",
                provider_options,
                runtime.provider_runtime,
                runtime.det_session_shapes,
                runtime.rec_session_shapes,
                INSIGHTFACE_SINGLE_LANE,
                self._face_preprocess_worker_count,
                self._face_batch_size,
                int(self._face_batch_wait_seconds * 1000.0),
                self._face_admission.capacity,
                "native-detect/native-recognition+local-align",
                runtime.runtime_root,
            )
        except Exception as exc:
            self._face_engine = None
            self._face_preprocess_device = None
            raise RuntimeError(
                "InsightFace must run with OpenVINOExecutionProvider + insightface native CPU preprocessing + local CPU alignment. "
                "No silent fallback is allowed."
            ) from exc

    @staticmethod
    def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _InferenceCancelled("Face task cancelled")

    @staticmethod
    def _set_face_task_result(task: _FaceInferenceTask, value: List[RepresentResult]) -> None:
        task.payload = None
        if not task.future.done():
            task.future.set_result(value)

    @staticmethod
    def _set_face_task_exception(task: _FaceInferenceTask, exc: Exception) -> None:
        task.payload = None
        if not task.future.done():
            task.future.set_exception(exc)

    def _run_face_preprocess_stage(
        self,
        jobs: List[Callable[[], Any]],
    ) -> List[Any]:
        if not jobs:
            return []
        if self._face_preprocess_worker_count <= 1 or len(jobs) <= 1:
            return [job() for job in jobs]

        futures = {
            self._face_preprocess_executor.submit(job): index
            for index, job in enumerate(jobs)
        }
        results: List[Any] = [None] * len(jobs)
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise
        return results

    def _build_pending_face_recognition(
        self,
        *,
        task_index: int,
        detections: np.ndarray,
        kpss: np.ndarray,
        source_frame: Any,
        rec_image_size: int,
        cancel_event: threading.Event,
    ) -> _PendingInsightFaceRecognition:
        face_meta: List[Tuple[np.ndarray, float]] = []
        aligned_faces: List[np.ndarray] = []
        for face_index in range(detections.shape[0]):
            self._raise_if_cancelled(cancel_event)
            bbox = np.asarray(detections[face_index, 0:4], dtype=np.float32)
            det_score = float(detections[face_index, 4])
            aligned_faces.append(
                self._align_face(
                    source_frame,
                    np.asarray(kpss[face_index]),
                    rec_image_size,
                )
            )
            face_meta.append((bbox, det_score))
        return _PendingInsightFaceRecognition(
            task_index=task_index,
            face_meta=face_meta,
            aligned_faces=aligned_faces,
        )

    def _build_pending_face_recognition_result(
        self,
        *,
        task_index: int,
        detections: np.ndarray,
        kpss: np.ndarray,
        source_frame: Any,
        rec_image_size: int,
        cancel_event: threading.Event,
    ) -> Tuple[int, Optional[_PendingInsightFaceRecognition], Optional[Exception]]:
        try:
            return (
                task_index,
                self._build_pending_face_recognition(
                    task_index=task_index,
                    detections=detections,
                    kpss=kpss,
                    source_frame=source_frame,
                    rec_image_size=rec_image_size,
                    cancel_event=cancel_event,
                ),
                None,
            )
        except _InferenceCancelled as exc:
            return task_index, None, exc

    def _submit_face_task(self, image: np.ndarray) -> _FaceInferenceTask:
        future: Future[List[RepresentResult]] = Future()
        task = _FaceInferenceTask(payload=image, future=future, created_at=time.time())
        loop = self._face_dispatch_loop
        if self._stopping:
            future.set_exception(RuntimeError("模型服务已关闭"))
            return task
        if loop is None:
            future.set_exception(RuntimeError("InsightFace queue loop is not initialized."))
            return task

        try:
            submit_future = asyncio.run_coroutine_threadsafe(
                self._enqueue_face_task_async(task),
                loop,
            )
            submit_future.result(timeout=max(1.0, float(self._queue_timeout_seconds)))
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

    def _require_face_dispatch_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._face_dispatch_loop
        if loop is None:
            raise RuntimeError("InsightFace queue loop is not initialized.")
        return loop

    def _require_face_queue(self) -> asyncio.Queue[Optional[_FaceInferenceTask]]:
        task_queue = self._face_task_queue
        if task_queue is None:
            raise RuntimeError("InsightFace batch queue is not initialized.")
        return task_queue

    def _face_worker_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._face_dispatch_loop = loop
        self._face_task_queue = asyncio.Queue(maxsize=self._face_queue_capacity)
        self._face_loop_ready.set()
        worker_task = loop.create_task(self._face_worker_loop())
        try:
            loop.run_forever()
        finally:
            worker_task.cancel()
            loop.run_until_complete(asyncio.gather(worker_task, return_exceptions=True))
            self._face_task_queue = None
            self._face_dispatch_loop = None
            loop.close()

    async def _face_worker_loop(self) -> None:
        task_queue = self._require_face_queue()
        stop_after_batch = False
        while True:
            queued = await task_queue.get()
            if queued is None:
                task_queue.task_done()
                break
            if queued.cancel_requested.is_set():
                task_queue.task_done()
                continue
            queued.started_at = time.monotonic()
            queued.started_event.set()
            batch, stop_after_batch = await self._collect_face_task_batch(queued)
            try:
                self._handle_face_task_batch(batch)
            finally:
                for _ in batch:
                    task_queue.task_done()
            if stop_after_batch:
                break

    async def _collect_face_task_batch(
        self,
        first_task: _FaceInferenceTask,
    ) -> Tuple[List[_FaceInferenceTask], bool]:
        batch = [first_task]
        if self._face_batch_size <= 1:
            return batch, False

        task_queue = self._require_face_queue()
        deadline = asyncio.get_running_loop().time() + self._face_batch_wait_seconds
        stop_after_batch = False
        while len(batch) < self._face_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0:
                break
            try:
                queued = await asyncio.wait_for(task_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if queued is None:
                task_queue.task_done()
                stop_after_batch = True
                break
            if queued.cancel_requested.is_set():
                task_queue.task_done()
                continue
            queued.started_at = time.monotonic()
            queued.started_event.set()
            batch.append(queued)
        return batch, stop_after_batch

    async def _enqueue_face_task_async(self, task: _FaceInferenceTask) -> None:
        if self._stopping:
            raise RuntimeError("模型服务已关闭")
        task_queue = self._face_task_queue
        if task_queue is None:
            raise RuntimeError("InsightFace batch queue is not initialized.")
        if task_queue.full():
            raise RuntimeError(f"推理队列已满（上限 {self._face_queue_capacity}），请稍后重试")
        task_queue.put_nowait(task)

    def _start_face_batch_service(self) -> None:
        if self._face_worker is not None:
            return

        self._face_worker = threading.Thread(
            target=self._face_worker_thread_main,
            name="ai-face-queue",
            daemon=True,
        )
        self._face_worker.start()
        ready = self._face_loop_ready.wait(
            timeout=max(2.0, float(self._execution_timeout_seconds))
        )
        if not ready:
            raise RuntimeError("InsightFace queue worker failed to initialize in time.")

    def _stop_face_batch_service(self) -> List[_FaceInferenceTask]:
        pending: List[_FaceInferenceTask] = []
        loop = self._face_dispatch_loop
        if loop is not None:
            shutdown_future = asyncio.run_coroutine_threadsafe(
                self._shutdown_face_queue_async(),
                loop,
            )
            try:
                pending = shutdown_future.result(
                    timeout=max(2.0, float(self._execution_timeout_seconds))
                )
            except Exception as exc:
                LOG.warning("Failed to drain InsightFace queue during shutdown: %s", exc)
            loop.call_soon_threadsafe(loop.stop)

        worker = self._face_worker
        if worker is not None:
            worker.join(timeout=max(2.0, float(self._execution_timeout_seconds)))
            self._face_worker = None

        self._face_loop_ready.clear()
        return pending

    async def _shutdown_face_queue_async(self) -> List[_FaceInferenceTask]:
        pending: List[_FaceInferenceTask] = []
        task_queue = self._face_task_queue
        if task_queue is None:
            return pending
        while True:
            try:
                queued = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            task_queue.task_done()
            if queued is None:
                continue
            pending.append(queued)
        task_queue.put_nowait(None)
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
        rec_image_size = int(rec_model.input_size[0])
        recognition_job_task_indices: List[int] = []
        recognition_jobs: List[
            Callable[[], Tuple[int, Optional[_PendingInsightFaceRecognition], Optional[Exception]]]
        ] = []
        for task_index, task in enumerate(tasks):
            if task.cancel_requested.is_set():
                outcomes[task_index] = (None, _InferenceCancelled("Face task cancelled"))
                continue

            detections, kpss = self._detect_faces_native(
                det_model,
                cast(np.ndarray, task.payload),
                max_num=0,
                metric="default",
            )
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

            recognition_job_task_indices.append(task_index)
            recognition_jobs.append(
                lambda task_index=task_index,
                detections=np.asarray(detections, dtype=np.float32),
                kpss=np.asarray(kpss, dtype=np.float32),
                source_frame=task.payload,
                cancel_event=task.cancel_requested: self._build_pending_face_recognition_result(
                    task_index=task_index,
                    detections=detections,
                    kpss=kpss,
                    source_frame=source_frame,
                    rec_image_size=rec_image_size,
                    cancel_event=cancel_event,
                )
            )

        recognition_items_by_task: Dict[int, _PendingInsightFaceRecognition] = {}
        for task_index, item, exc in cast(
            List[Tuple[int, Optional[_PendingInsightFaceRecognition], Optional[Exception]]],
            self._run_face_preprocess_stage(recognition_jobs),
        ):
            if exc is not None:
                outcomes[task_index] = (None, exc)
                continue
            if item is not None:
                recognition_items_by_task[task_index] = item

        all_aligned_faces: List[np.ndarray] = []
        recognition_spans: List[Tuple[int, List[Tuple[np.ndarray, float]], int, int]] = []
        for task_index in recognition_job_task_indices:
            item = recognition_items_by_task.get(task_index)
            if item is None:
                continue
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
            embeddings = self._get_face_embeddings_native(rec_model, all_aligned_faces)
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
            task = self._submit_face_task(
                _as_contiguous_bgr_uint8(image, context="InsightFace")
            )
            return await self._await_face_task(task)
