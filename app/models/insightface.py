import asyncio
import inspect
import json
import os
import shutil
import sys
import types
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

if TYPE_CHECKING:
    from app.schemas import FacialArea, RepresentResult
elif __package__ and "." in __package__:
    from ..schemas import FacialArea, RepresentResult
else:
    _schemas = import_module("schemas")
    FacialArea = _schemas.FacialArea
    RepresentResult = _schemas.RepresentResult

from .common import (
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
    _execution_timeout_seconds: int

    if TYPE_CHECKING:
        def _build_openvino_preprocess_runner(
            self,
            runner_name: str,
            device_name: str,
            output_height: int,
            output_width: int,
            mean_values: List[float],
            std_values: List[float],
        ) -> _OpenVinoPreprocessRunner: ...
        def _load_family_with_process_lock(self, family: str, loader: Any) -> None: ...
        def _acquire_non_text_family_lease(self, family: str) -> bool: ...
        def _release_non_text_family_lease(self, family: str) -> None: ...
        @staticmethod
        def _run_in_executor(
            executor: ThreadPoolExecutor,
            func: Any,
            *args: Any,
        ) -> asyncio.Future[Any]: ...
        def _bind_non_text_lease_to_future(
            self,
            family: str,
            future: Future[Any] | asyncio.Future[Any],
        ) -> None: ...
        @staticmethod
        def _log_detached_async_task_failure(task: "asyncio.Task[Any]", task_name: str) -> None: ...

    @staticmethod
    def _enable_insightface_opencl_alignment() -> None:
        name, vendor = _ensure_intel_opencl_device("InsightFace alignment")
        LOG.info(
            "OpenCV OpenCL enabled for InsightFace alignment on Intel device: %s (%s).",
            name,
            vendor,
        )

    @staticmethod
    def _patch_insightface_norm_crop_opencl() -> None:
        from insightface.utils import face_align

        if getattr(face_align, "_mt_opencl_norm_crop_patch", False):
            return

        def _estimate_norm_no_warning(
            landmark: np.ndarray,
            image_size: int = 112,
            mode: str = "arcface",
        ) -> np.ndarray:
            _ = mode
            return _estimate_insightface_norm_matrix(
                landmark=landmark,
                image_size=int(image_size),
            )

        def _norm_crop_opencl(
            img: np.ndarray,
            landmark: np.ndarray,
            image_size: int = 112,
            mode: str = "arcface",
        ) -> np.ndarray:
            matrix = _estimate_norm_no_warning(landmark, image_size, mode)
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

        face_align.estimate_norm = _estimate_norm_no_warning
        face_align.norm_crop = _norm_crop_opencl
        face_align._mt_opencl_norm_crop_patch = True

    def _patch_face_detector_forward(
        self,
        det_model: Any,
        preprocess_runner: _OpenVinoPreprocessRunner,
    ) -> None:
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

        def _forward_with_ppp(model_self: Any, img: np.ndarray, threshold: float) -> Any:
            scores_list: List[np.ndarray] = []
            bboxes_list: List[np.ndarray] = []
            kpss_list: List[np.ndarray] = []

            image_bgr = _as_contiguous_bgr_uint8(np.asarray(img), context="InsightFace detector")
            blob = preprocess_runner.run(image_bgr[np.newaxis, ...])
            net_outs = model_self.session.run(model_self.output_names, {model_self.input_name: blob})

            input_height = int(blob.shape[2])
            input_width = int(blob.shape[3])
            fmc = model_self.fmc
            batched_output = bool(getattr(model_self, "batched", False))

            for idx, stride in enumerate(model_self._feat_stride_fpn):
                kps_preds: Optional[np.ndarray] = None
                if batched_output:
                    scores = net_outs[idx][0]
                    bbox_preds = net_outs[idx + fmc][0] * stride
                    if model_self.use_kps:
                        kps_preds = net_outs[idx + fmc * 2][0] * stride
                else:
                    scores = net_outs[idx]
                    bbox_preds = net_outs[idx + fmc] * stride
                    if model_self.use_kps:
                        kps_preds = net_outs[idx + fmc * 2] * stride

                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in model_self.center_cache:
                    anchor_centers = model_self.center_cache[key]
                else:
                    anchor_centers = np.stack(
                        np.mgrid[:height, :width][::-1], axis=-1
                    ).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                    if model_self._num_anchors > 1:
                        anchor_centers = np.stack([anchor_centers] * model_self._num_anchors, axis=1)
                        anchor_centers = anchor_centers.reshape((-1, 2))
                    if len(model_self.center_cache) < 100:
                        model_self.center_cache[key] = anchor_centers

                pos_inds = np.where(scores >= threshold)[0]
                bboxes = distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                if model_self.use_kps:
                    if kps_preds is None:
                        raise RuntimeError(
                            "InsightFace detector returned no keypoint predictions "
                            "for a keypoint-enabled model."
                        )
                    kpss = distance2kps(anchor_centers, kps_preds)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)

            return scores_list, bboxes_list, kpss_list

        det_model.forward = types.MethodType(_forward_with_ppp, det_model)

    @staticmethod
    def _patch_face_recognition_get_feat(
        rec_model: Any,
        preprocess_runner: _OpenVinoPreprocessRunner,
    ) -> None:
        def _get_feat_with_ppp(model_self: Any, imgs: Any) -> np.ndarray:
            if not isinstance(imgs, list):
                imgs = [imgs]
            if not imgs:
                return np.empty((0, 0), dtype=np.float32)

            prepared = [
                _as_contiguous_bgr_uint8(np.asarray(item), context="InsightFace recognition")
                for item in imgs
            ]
            try:
                batch = np.stack(prepared, axis=0)
                blob = preprocess_runner.run(batch)
                return model_self.session.run(model_self.output_names, {model_self.input_name: blob})[0]
            except ValueError:
                features: List[np.ndarray] = []
                for item in prepared:
                    blob = preprocess_runner.run(item[np.newaxis, ...])
                    single = model_self.session.run(
                        model_self.output_names,
                        {model_self.input_name: blob},
                    )[0]
                    features.append(np.asarray(single))
                return np.concatenate(features, axis=0)

        rec_model.get_feat = types.MethodType(_get_feat_with_ppp, rec_model)

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

        self._patch_face_detector_forward(det_model, self._face_det_ppp)
        self._patch_face_recognition_get_feat(rec_model, self._face_rec_ppp)
        self._enable_insightface_opencl_alignment()
        self._patch_insightface_norm_crop_opencl()
        LOG.info("InsightFace preprocessing patched: OpenCV(OpenCL align) + OpenVINO PPP.")

    def _resolve_insightface_root(self) -> Path:
        candidate_roots = (self.insightface_model_root, self.insightface_root)
        for root in candidate_roots:
            if (root / MODEL_NAME).is_dir():
                return root
        raise FileNotFoundError(
            "InsightFace model directory missing for antelopev2. Checked paths: "
            f"{self.insightface_model_root / MODEL_NAME}, {self.insightface_root / MODEL_NAME}"
        )

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

    def _instantiate_insightface_face_analysis(
        self,
        provider_names: List[str],
        provider_options: Dict[str, str],
    ) -> Tuple[FaceAnalysis, Path]:
        source_root = self._resolve_insightface_root()
        init_signature = inspect.signature(FaceAnalysis.__init__)
        init_parameters = init_signature.parameters
        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in init_parameters.values()
        )
        supports_allowed_modules = "allowed_modules" in init_parameters
        supports_provider_kwargs = "providers" in init_parameters or supports_var_kwargs
        supports_provider_options = "provider_options" in init_parameters or supports_var_kwargs
        legacy_root: Optional[Path] = None

        def _build_kwargs(
            runtime_root: Path,
            *,
            include_provider_kwargs: bool,
            include_allowed_modules: bool,
        ) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "name": MODEL_NAME,
                "root": str(runtime_root),
            }
            if include_allowed_modules:
                kwargs["allowed_modules"] = ["detection", "recognition"]
            if include_provider_kwargs:
                kwargs["providers"] = list(provider_names)
                if supports_provider_options:
                    kwargs["provider_options"] = [dict(provider_options)]
            return kwargs

        attempts: List[Tuple[str, Path, Dict[str, Any]]] = [
            (
                "source-with-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=supports_provider_kwargs,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
            (
                "source-without-provider-kwargs",
                source_root,
                _build_kwargs(
                    source_root,
                    include_provider_kwargs=False,
                    include_allowed_modules=supports_allowed_modules,
                ),
            ),
        ]

        if legacy_root is None:
            legacy_root = self._prepare_legacy_insightface_runtime_root(source_root)
        attempts.extend(
            (
                (
                    "legacy-with-provider-kwargs",
                    legacy_root,
                    _build_kwargs(
                        legacy_root,
                        include_provider_kwargs=supports_provider_kwargs,
                        include_allowed_modules=supports_allowed_modules,
                    ),
                ),
                (
                    "legacy-minimal-kwargs",
                    legacy_root,
                    _build_kwargs(
                        legacy_root,
                        include_provider_kwargs=False,
                        include_allowed_modules=False,
                    ),
                ),
            )
        )

        seen_attempts: set[Tuple[str, str]] = set()
        attempt_errors: List[str] = []
        last_exc: Optional[Exception] = None
        for attempt_name, runtime_root, kwargs in attempts:
            attempt_key = (str(runtime_root), json.dumps(kwargs, sort_keys=True, ensure_ascii=True))
            if attempt_key in seen_attempts:
                continue
            seen_attempts.add(attempt_key)
            try:
                face_app = FaceAnalysis(**kwargs)
                if attempt_name != "source-with-provider-kwargs":
                    LOG.warning(
                        "InsightFace initialized via compatibility path %s (runtime_root=%s)",
                        attempt_name,
                        runtime_root,
                    )
                return face_app, runtime_root
            except Exception as exc:
                last_exc = exc
                attempt_errors.append(f"{attempt_name}: {exc}")
                if self._is_insightface_init_kwargs_error(exc):
                    LOG.warning(
                        "InsightFace init retry after unsupported kwargs on %s: %s",
                        attempt_name,
                        exc,
                    )
                else:
                    LOG.warning(
                        "InsightFace init attempt %s failed (runtime_root=%s): %s",
                        attempt_name,
                        runtime_root,
                        exc,
                    )

        summary = "; ".join(attempt_errors) or "no attempts executed"
        raise RuntimeError(
            "InsightFace initialization failed after compatibility retries: "
            f"{summary}"
        ) from last_exc

    def _prepare_legacy_insightface_runtime_root(self, source_root: Path) -> Path:
        source_model_dir = source_root / MODEL_NAME
        runtime_root = self.insightface_root / "_runtime_models"
        runtime_model_dir = runtime_root / "models" / MODEL_NAME
        runtime_model_dir.mkdir(parents=True, exist_ok=True)

        required_files = ("scrfd_10g_bnkps.onnx", "glintr100.onnx")
        for filename in required_files:
            src = source_model_dir / filename
            dst = runtime_model_dir / filename
            if not src.is_file():
                raise FileNotFoundError(f"InsightFace required model missing: {src}")
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except Exception:
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
        return runtime_root

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
            face_app, root_for_runtime = self._instantiate_insightface_face_analysis(
                provider_names,
                provider_options,
            )
            self._enforce_insightface_openvino_provider(face_app, provider_options)
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            self._enforce_insightface_openvino_provider(face_app, provider_options)
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
            self._face_engine = face_app
            LOG.info(
                "InsightFace loaded with providers=%s configured_device=%s runtime_device=%s provider_options=%s provider_runtime=%s ppp_execution_devices=%s (runtime_root=%s)",
                provider_names,
                configured_provider_device,
                provider_device,
                provider_options,
                provider_runtime,
                ppp_execution_devices,
                root_for_runtime,
            )
        except Exception as exc:
            self._face_engine = None
            self._face_det_ppp = None
            self._face_rec_ppp = None
            raise RuntimeError(
                "InsightFace must run with OpenVINOExecutionProvider + OpenCV OpenCL alignment. "
                "No silent fallback is allowed."
            ) from exc

    def _infer_face(self, image: np.ndarray) -> List[RepresentResult]:
        if self._face_engine is None:
            raise RuntimeError("Face model is not loaded.")

        face_input = _as_contiguous_bgr_uint8(image, context="InsightFace")
        faces = self._face_engine.get(face_input)
        if not faces:
            return []

        results: List[RepresentResult] = []
        for face in faces:
            bbox = np.array(face.bbox).astype(int)
            x1, y1, x2, y2 = bbox
            results.append(
                RepresentResult(
                    embedding=[float(value) for value in face.normed_embedding],
                    facial_area=FacialArea(
                        x=int(x1),
                        y=int(y1),
                        w=int(x2 - x1),
                        h=int(y2 - y1),
                    ),
                    face_confidence=float(face.det_score),
                )
            )
        return results

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        self._acquire_non_text_family_lease("face")
        try:
            self._ensure_face_loaded()
            return self._infer_face(image)
        finally:
            self._release_non_text_family_lease("face")

    async def get_face_representation_async(self, image: np.ndarray) -> List[RepresentResult]:
        lease_bound = False
        await asyncio.to_thread(self._acquire_non_text_family_lease, "face")
        try:
            await asyncio.to_thread(self._ensure_face_loaded)
            future = self._run_in_executor(self._face_executor, self._infer_face, image)
            self._bind_non_text_lease_to_future("face", future)
            lease_bound = True
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self._execution_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                future.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="Face task")
                )
                raise RuntimeError(f"推理任务执行超时（>{self._execution_timeout_seconds}s）") from exc
            except asyncio.CancelledError:
                future.add_done_callback(
                    partial(self._log_detached_async_task_failure, task_name="Face task")
                )
                raise
        finally:
            if not lease_bound:
                self._release_non_text_family_lease("face")
