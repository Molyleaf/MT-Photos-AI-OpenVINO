import asyncio
import logging
import os
import threading
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from queue import LifoQueue
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, cast

import cv2
import numpy as np
import openvino as ov
import yaml
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_rec.main import LangRec, VisRes, reorder_bidi_for_display
from rapidocr.inference_engine.base import InferSession
from rapidocr.main import RapidOCRError, TextClsOutput, TextDetOutput, TextRecOutput
from rapidocr.utils.log import logger as RAPIDOCR_LOGGER
from rapidocr.utils.process_img import apply_vertical_padding, get_rotate_crop_image
from rapidocr.utils.typings import EngineType

if TYPE_CHECKING:
    from app.schemas import OCRBox, OCRResult
elif __package__ and "." in __package__:
    from ..schemas import OCRBox, OCRResult
else:
    _schemas = import_module("schemas")
    OCRBox = _schemas.OCRBox
    OCRResult = _schemas.OCRResult

from .common import (
    _AdmissionController,
    _InferenceCancelled,
    _as_bool,
    _as_contiguous_bgr_uint8,
    _as_int,
    _default_rapidocr_stage_device,
    _ensure_intel_opencl_device,
    _get_compiled_model_execution_devices,
    _get_openvino_session_execution_devices,
    _has_openvino_gpu_device,
    _normalize_non_text_openvino_device,
    _normalize_openvino_devices,
    _normalize_rapidocr_limit_type,
    _openvino_device_expr_requests_cpu,
    _openvino_device_expr_requests_gpu,
    _openvino_device_expr_requests_npu,
    _resolve_non_text_openvino_runtime_device,
    _split_openvino_device_expr,
    _to_opencv_umat,
)
from .constants import (
    DEFAULT_NON_TEXT_OV_DEVICE,
    INFERENCE_DEVICE,
    LOG,
    RAPIDOCR_CLS_MOBILE_V2_FILE,
    RAPIDOCR_V5_DICT_FILE,
    RAPIDOCR_V5_MOBILE_DET_FILE,
    RAPIDOCR_V5_MOBILE_REC_FILE,
)

_OV_DEVICE_PRIORITIES_KEY = str(
    getattr(ov.properties.device, "priorities", "MULTI_DEVICE_PRIORITIES")
)
_OV_AUTO_STARTUP_FALLBACK_KEY = str(
    getattr(ov.properties.intel_auto, "enable_startup_fallback", "ENABLE_STARTUP_FALLBACK")
)
_OV_AUTO_RUNTIME_FALLBACK_KEY = str(
    getattr(ov.properties.intel_auto, "enable_runtime_fallback", "ENABLE_RUNTIME_FALLBACK")
)

_TextClsResult = Tuple[str, float]
_TextRecWordResult = Tuple[str, float, Optional[List[List[int]]]]


class _SuppressExpectedRapidOCRNoTextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().strip() != "The text detection result is empty"


def _empty_text_rec_output() -> TextRecOutput:
    return TextRecOutput(
        imgs=[],
        txts=cast(Any, tuple()),
        scores=[],
        word_results=cast(Any, tuple()),
        elapse=0.0,
    )


class _RapidOCROpenVINOInferSession(InferSession):
    def __init__(self, cfg: Any):
        from rapidocr.inference_engine.openvino.main import CPUConfig, Core, OpenVINOError

        self._openvino_error_cls = OpenVINOError
        engine_cfg = cfg.get("engine_cfg", {}) or {}
        stage_device = cfg.get("device_name", None)
        configured_device = _normalize_non_text_openvino_device(
            str(
                stage_device
                if stage_device not in (None, "")
                else engine_cfg.get(
                    "device_name",
                    os.environ.get("RAPIDOCR_DEVICE", INFERENCE_DEVICE),
                )
            )
        )
        core = Core()
        selected_device = _resolve_non_text_openvino_runtime_device(
            configured_device,
            core.available_devices,
            consumer="openvino",
        )

        model_path = cfg.get("model_path", None)
        if model_path is None:
            raise RuntimeError(
                "RapidOCR OpenVINO local model_path is required. "
                "Online download fallback is disabled."
            )
        model_path = Path(model_path)
        self._verify_model(model_path)

        cpu_config = CPUConfig(engine_cfg)
        core.set_property("CPU", cpu_config.get_config())

        cache_dir = engine_cfg.get("cache_dir")
        if cache_dir not in (None, ""):
            core.set_property({"CACHE_DIR": str(cache_dir)})

        compile_cfg: Dict[str, str] = {}
        for key, ov_key in (
            ("performance_hint", "PERFORMANCE_HINT"),
            ("performance_num_requests", "PERFORMANCE_HINT_NUM_REQUESTS"),
            ("num_streams", "NUM_STREAMS"),
        ):
            value = engine_cfg.get(key)
            if value in (None, "", -1):
                continue
            compile_cfg[ov_key] = str(value)

        if selected_device.startswith("AUTO:"):
            device_priorities = ",".join(_split_openvino_device_expr(selected_device))
            if device_priorities:
                compile_cfg[_OV_DEVICE_PRIORITIES_KEY] = device_priorities
            compile_cfg[_OV_AUTO_STARTUP_FALLBACK_KEY] = "false"
            compile_cfg[_OV_AUTO_RUNTIME_FALLBACK_KEY] = "false"

        try:
            self.model = core.read_model(model_path)
            compiled_model = core.compile_model(
                model=self.model,
                device_name=selected_device,
                config=compile_cfg,
            )
        except Exception as exc:
            raise RuntimeError(
                f"RapidOCR OpenVINO compile_model failed on device={selected_device}. "
                "No silent fallback is allowed."
            ) from exc

        self._mt_core = core
        self._mt_compiled_model = compiled_model
        self._mt_request_local = threading.local()
        self.session = compiled_model.create_infer_request()
        self._mt_device_name = selected_device
        self._mt_configured_device_name = configured_device
        self._mt_execution_devices = tuple(_get_compiled_model_execution_devices(compiled_model))

    def _get_request(self) -> ov.InferRequest:
        request = getattr(self._mt_request_local, "request", None)
        if request is None:
            request = self._mt_compiled_model.create_infer_request()
            self._mt_request_local.request = request
        return request

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        try:
            prepared = np.ascontiguousarray(input_content)
            infer_request = self._get_request()
            infer_request.set_input_tensor(0, ov.Tensor(prepared, shared_memory=True))
            infer_request.infer()
            return np.asarray(infer_request.get_output_tensor(0).data)
        except Exception as exc:
            error_info = traceback.format_exc()
            raise self._openvino_error_cls(error_info) from exc

    def have_key(self, key: str = "character") -> bool:
        try:
            self.get_character_list(key)
            return True
        except Exception:
            return False

    def get_character_list(self, key: str = "character") -> List[str]:
        framework_info = self.get_rt_info_framework()
        if framework_info is None:
            raise self._openvino_error_cls("Failed to get runtime framework info")
        if key not in framework_info:
            raise self._openvino_error_cls(f"Key '{key}' not found in framework info")
        value_node = framework_info[key]
        value = getattr(value_node, "value", None)
        if value is None:
            raise self._openvino_error_cls(f"Value is None for key '{key}'")
        return str(value).splitlines()

    def get_rt_info_framework(self) -> Any:
        rt_info = self.model.get_rt_info()
        if "framework" not in rt_info:
            return None
        return rt_info["framework"]


class RapidOCRMixin:
    core: ov.Core
    ov_cache_dir: Optional[Path]
    rapidocr_config_path: Path
    rapidocr_model_dir_path: Path
    rapidocr_font_path: str
    _rapidocr_load_lock: Any
    _rapidocr_engine: Optional[RapidOCR]
    _rapidocr_engines: Optional[Tuple[RapidOCR, ...]]
    _rapidocr_engine_pool: Optional[LifoQueue]
    _rapidocr_runtime_cfg: Optional[Dict[str, Any]]
    _ocr_opencl_device_name: Optional[str]
    _ocr_opencl_device_vendor: Optional[str]
    _ocr_stage_worker_count: int
    _ocr_det_executor: ThreadPoolExecutor
    _ocr_cls_executor: ThreadPoolExecutor
    _ocr_rec_executor: ThreadPoolExecutor
    _shared_cpu_executor: ThreadPoolExecutor
    _ocr_admission: _AdmissionController
    _ocr_execution_timeout_seconds: int

    if TYPE_CHECKING:
        def _acquire_image_request_slot(self, label: str) -> None: ...
        def _release_image_request_slot(self) -> None: ...
        def _load_family_with_process_lock(self, family: str, loader: Any) -> None: ...
        def _acquire_non_text_family_lease(self, family: str) -> bool: ...
        async def _acquire_non_text_family_lease_async(self, family: str) -> bool: ...
        def _release_non_text_family_lease(self, family: str) -> None: ...
        def _acquire_admission(self, admission: _AdmissionController, label: str) -> None: ...
        async def _acquire_admission_async(
            self,
            admission: _AdmissionController,
            label: str,
        ) -> None: ...
        def _run_control(
            self,
            func: Any,
            *args: Any,
        ) -> asyncio.Future[Any]: ...
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
        async def _await_with_timeout_and_cooperative_cancel(
            self,
            awaitable: asyncio.Future[Any] | asyncio.Task[Any],
            *,
            cancel_event: threading.Event,
            timeout_seconds: float,
            task_name: str,
        ) -> Any: ...
        @staticmethod
        def _log_detached_async_task_failure(task: "asyncio.Task[Any]", task_name: str) -> None: ...

    def _require_rapidocr_config_path(self) -> Path:
        if not self.rapidocr_config_path.is_file():
            raise FileNotFoundError(
                "RapidOCR OpenVINO config file is required and must exist. "
                f"Configured path: {self.rapidocr_config_path}"
            )
        return self.rapidocr_config_path

    @staticmethod
    def _configure_rapidocr_logger() -> None:
        if getattr(RAPIDOCR_LOGGER, "_mt_expected_no_text_filter", None) is not None:
            return

        message_filter = _SuppressExpectedRapidOCRNoTextFilter()
        RAPIDOCR_LOGGER.addFilter(message_filter)
        for handler in RAPIDOCR_LOGGER.handlers:
            handler.addFilter(message_filter)
        setattr(RAPIDOCR_LOGGER, "_mt_expected_no_text_filter", message_filter)

    def _load_rapidocr_openvino_config(self) -> Dict[str, Any]:
        requested_device = _normalize_non_text_openvino_device(
            os.environ.get("RAPIDOCR_DEVICE", "CPU")
        )
        if requested_device != "CPU":
            LOG.warning(
                "RapidOCR upstream OpenVINO backend is CPU-only; coercing RAPIDOCR_DEVICE=%s to CPU.",
                requested_device,
            )
        config: Dict[str, Any] = {
            "requested_device_name": requested_device,
            "device_name": "CPU",
            "det_device_name": "CPU",
            "cls_device_name": "CPU",
            "rec_device_name": "CPU",
            "inference_num_threads": _as_int(os.environ.get("RAPIDOCR_INFERENCE_NUM_THREADS"), -1),
            "performance_hint": os.environ.get("RAPIDOCR_PERFORMANCE_HINT", "THROUGHPUT"),
            "performance_num_requests": _as_int(
                os.environ.get("RAPIDOCR_PERFORMANCE_NUM_REQUESTS"), 2
            ),
            "enable_cpu_pinning": _as_bool(os.environ.get("RAPIDOCR_ENABLE_CPU_PINNING"), True),
            "num_streams": _as_int(os.environ.get("RAPIDOCR_NUM_STREAMS"), 2),
            "enable_hyper_threading": _as_bool(
                os.environ.get("RAPIDOCR_ENABLE_HYPER_THREADING"), True
            ),
            "scheduling_core_type": os.environ.get("RAPIDOCR_SCHEDULING_CORE_TYPE", "ANY_CORE"),
            "use_cls": _as_bool(os.environ.get("RAPIDOCR_USE_CLS"), True),
            "max_side_len": _as_int(os.environ.get("RAPIDOCR_MAX_SIDE_LEN"), 960),
            "det_limit_side_len": _as_int(os.environ.get("RAPIDOCR_DET_LIMIT_SIDE_LEN"), 960),
            "det_limit_type": _normalize_rapidocr_limit_type(
                os.environ.get("RAPIDOCR_DET_LIMIT_TYPE", "max")
            ),
            "rec_batch_num": max(1, _as_int(os.environ.get("RAPIDOCR_REC_BATCH_NUM"), 8)),
            "cls_batch_num": max(1, _as_int(os.environ.get("RAPIDOCR_CLS_BATCH_NUM"), 8)),
        }

        config_path = self._require_rapidocr_config_path()
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        ov_cfg = loaded.get("EngineConfig", {}).get("openvino", {})
        for key in list(config.keys()):
            if key in ov_cfg:
                config[key] = ov_cfg[key]
        config["requested_device_name"] = requested_device
        config["device_name"] = "CPU"

        global_cfg = loaded.get("Global", {})
        if "use_cls" in global_cfg:
            config["use_cls"] = _as_bool(global_cfg.get("use_cls"), config["use_cls"])
        if "max_side_len" in global_cfg:
            config["max_side_len"] = _as_int(
                global_cfg.get("max_side_len"), config["max_side_len"]
            )

        det_cfg = loaded.get("Det", {})
        if "limit_side_len" in det_cfg:
            config["det_limit_side_len"] = _as_int(
                det_cfg.get("limit_side_len"), config["det_limit_side_len"]
            )
        if "limit_type" in det_cfg:
            config["det_limit_type"] = _normalize_rapidocr_limit_type(
                det_cfg.get("limit_type"),
                config["det_limit_type"],
            )

        rec_cfg = loaded.get("Rec", {})
        if "rec_batch_num" in rec_cfg:
            config["rec_batch_num"] = max(
                1, _as_int(rec_cfg.get("rec_batch_num"), config["rec_batch_num"])
            )

        cls_cfg = loaded.get("Cls", {})
        if "cls_batch_num" in cls_cfg:
            config["cls_batch_num"] = max(
                1, _as_int(cls_cfg.get("cls_batch_num"), config["cls_batch_num"])
            )

        env_override_map: Dict[str, Tuple[str, Any]] = {
            "RAPIDOCR_INFERENCE_NUM_THREADS": (
                "inference_num_threads",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_PERFORMANCE_HINT": (
                "performance_hint",
                lambda value, default: str(value).strip() or default,
            ),
            "RAPIDOCR_PERFORMANCE_NUM_REQUESTS": (
                "performance_num_requests",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_ENABLE_CPU_PINNING": (
                "enable_cpu_pinning",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_NUM_STREAMS": (
                "num_streams",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_ENABLE_HYPER_THREADING": (
                "enable_hyper_threading",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_SCHEDULING_CORE_TYPE": (
                "scheduling_core_type",
                lambda value, default: str(value).strip() or default,
            ),
            "RAPIDOCR_USE_CLS": (
                "use_cls",
                lambda value, default: _as_bool(value, default),
            ),
            "RAPIDOCR_MAX_SIDE_LEN": (
                "max_side_len",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_DET_LIMIT_SIDE_LEN": (
                "det_limit_side_len",
                lambda value, default: _as_int(value, default),
            ),
            "RAPIDOCR_DET_LIMIT_TYPE": (
                "det_limit_type",
                lambda value, default: _normalize_rapidocr_limit_type(value, default),
            ),
            "RAPIDOCR_REC_BATCH_NUM": (
                "rec_batch_num",
                lambda value, default: max(1, _as_int(value, default)),
            ),
            "RAPIDOCR_CLS_BATCH_NUM": (
                "cls_batch_num",
                lambda value, default: max(1, _as_int(value, default)),
            ),
        }
        for env_name, (cfg_name, parser) in env_override_map.items():
            raw_env = os.environ.get(env_name)
            if raw_env is None or str(raw_env).strip() == "":
                continue
            config[cfg_name] = parser(raw_env, config[cfg_name])

        for env_name in ("RAPIDOCR_DET_DEVICE", "RAPIDOCR_CLS_DEVICE", "RAPIDOCR_REC_DEVICE"):
            raw_env = os.environ.get(env_name)
            if raw_env is None or str(raw_env).strip() == "":
                continue
            normalized = _normalize_non_text_openvino_device(raw_env)
            if normalized != "CPU":
                LOG.warning(
                    "%s=%s is ignored; RapidOCR upstream OpenVINO backend is CPU-only.",
                    env_name,
                    normalized,
                )

        config["device_name"] = "CPU"
        config["det_device_name"] = "CPU"
        config["cls_device_name"] = "CPU"
        config["rec_device_name"] = "CPU"
        if self.ov_cache_dir is not None:
            config["cache_dir"] = str(self.ov_cache_dir)
        config["runtime_device_name"] = "CPU"
        config["det_runtime_device_name"] = "CPU"
        config["cls_runtime_device_name"] = "CPU"
        config["rec_runtime_device_name"] = "CPU"
        return config

    def _require_rapidocr_local_assets(self) -> Dict[str, Path]:
        required_files = {
            "det": RAPIDOCR_V5_MOBILE_DET_FILE,
            "rec": RAPIDOCR_V5_MOBILE_REC_FILE,
            "dict": RAPIDOCR_V5_DICT_FILE,
            "cls": RAPIDOCR_CLS_MOBILE_V2_FILE,
        }
        resolved: Dict[str, Path] = {}
        missing: List[str] = []
        for key, filename in required_files.items():
            candidate = self.rapidocr_model_dir_path / filename
            if candidate.exists() and candidate.is_file():
                resolved[key] = candidate
            else:
                missing.append(str(candidate))

        if missing:
            joined = "; ".join(missing)
            raise FileNotFoundError(
                "RapidOCR local assets missing; online download fallback is disabled. "
                f"Missing files: {joined}"
            )
        return resolved

    def _build_rapidocr_runtime_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
            "EngineConfig.openvino.device_name": str(
                cfg.get("runtime_device_name", cfg.get("device_name", "AUTO"))
            ),
            "Det.device_name": str(
                cfg.get("det_runtime_device_name", cfg.get("det_device_name", "AUTO"))
            ),
            "Cls.device_name": str(
                cfg.get("cls_runtime_device_name", cfg.get("cls_device_name", "AUTO"))
            ),
            "Rec.device_name": str(
                cfg.get("rec_runtime_device_name", cfg.get("rec_device_name", "AUTO"))
            ),
            "EngineConfig.openvino.inference_num_threads": _as_int(
                cfg.get("inference_num_threads"), -1
            ),
            "EngineConfig.openvino.performance_num_requests": _as_int(
                cfg.get("performance_num_requests"), -1
            ),
            "EngineConfig.openvino.enable_cpu_pinning": _as_bool(
                cfg.get("enable_cpu_pinning"), True
            ),
            "EngineConfig.openvino.num_streams": _as_int(cfg.get("num_streams"), -1),
            "EngineConfig.openvino.enable_hyper_threading": _as_bool(
                cfg.get("enable_hyper_threading"), True
            ),
            "EngineConfig.openvino.scheduling_core_type": str(
                cfg.get("scheduling_core_type", "ANY_CORE")
            ),
            "Global.use_cls": _as_bool(cfg.get("use_cls"), True),
            "Global.max_side_len": _as_int(cfg.get("max_side_len"), 960),
            "Det.limit_side_len": _as_int(cfg.get("det_limit_side_len"), 960),
            "Det.limit_type": _normalize_rapidocr_limit_type(cfg.get("det_limit_type"), "max"),
            "Rec.rec_batch_num": max(1, _as_int(cfg.get("rec_batch_num"), 6)),
            "Cls.cls_batch_num": max(1, _as_int(cfg.get("cls_batch_num"), 6)),
        }
        cache_dir = cfg.get("cache_dir")
        if cache_dir not in (None, ""):
            params["EngineConfig.openvino.cache_dir"] = str(cache_dir)
        if cfg.get("performance_hint") not in (None, ""):
            params["EngineConfig.openvino.performance_hint"] = str(
                cfg.get("performance_hint", "LATENCY")
            )

        if self.rapidocr_font_path:
            params["Global.font_path"] = self.rapidocr_font_path

        assets = self._require_rapidocr_local_assets()
        params["Det.model_path"] = str(assets["det"])
        params["Rec.model_path"] = str(assets["rec"])
        params["Rec.rec_keys_path"] = str(assets["dict"])
        params["Cls.model_path"] = str(assets["cls"])
        return params

    @staticmethod
    def _resolve_rapidocr_stage_worker_count(cfg: Dict[str, Any]) -> int:
        return max(1, _as_int(cfg.get("performance_num_requests"), 1))

    def _reconfigure_rapidocr_stage_executors(self, cfg: Dict[str, Any]) -> None:
        worker_count = self._resolve_rapidocr_stage_worker_count(cfg)
        self._ocr_admission.resize(self._resolve_ocr_request_capacity(worker_count))
        if worker_count == self._ocr_stage_worker_count:
            return

        old_executors = (
            self._ocr_det_executor,
            self._ocr_cls_executor,
            self._ocr_rec_executor,
        )
        self._ocr_det_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-det",
        )
        self._ocr_cls_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-cls",
        )
        self._ocr_rec_executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ocr-rec",
        )
        self._ocr_stage_worker_count = worker_count

        for executor in old_executors:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _validate_rapidocr_backend(engine: RapidOCR) -> None:
        cfg = getattr(engine, "cfg", None)
        if cfg is None:
            raise RuntimeError("RapidOCR initialized without runtime cfg metadata.")

        backend_errors: List[str] = []
        for section_name in ("Det", "Cls", "Rec"):
            section_cfg = getattr(cfg, section_name, None)
            engine_type = getattr(section_cfg, "engine_type", None)
            engine_name = str(getattr(engine_type, "value", engine_type)).strip().lower()
            if engine_name != EngineType.OPENVINO.value:
                backend_errors.append(f"{section_name}.engine_type={engine_name or 'missing'}")

        if backend_errors:
            raise RuntimeError(
                "RapidOCR backend validation failed after initialization: "
                f"{', '.join(backend_errors)}. No silent fallback is allowed."
            )

    @staticmethod
    def _configure_rapidocr_openvino_sessions(engine: RapidOCR) -> None:
        cfg = getattr(engine, "cfg", None)
        if cfg is None:
            raise RuntimeError("RapidOCR initialized without runtime cfg metadata.")

        stage_sessions = {
            "det": _RapidOCROpenVINOInferSession(cfg.Det),
            "cls": _RapidOCROpenVINOInferSession(cfg.Cls),
            "rec": _RapidOCROpenVINOInferSession(cfg.Rec),
        }
        engine.text_det.session = stage_sessions["det"]
        engine.text_cls.session = stage_sessions["cls"]
        engine.text_rec.session = stage_sessions["rec"]

    def _instantiate_rapidocr(self, params: Dict[str, Any]) -> RapidOCR:
        rapidocr_device = str(params.get("EngineConfig.openvino.device_name", "CPU")).upper()
        config_path = self._require_rapidocr_config_path()
        self._configure_rapidocr_logger()
        try:
            engine = RapidOCR(config_path=str(config_path), params=params)
        except Exception as exc:
            raise RuntimeError(
                f"RapidOCR 初始化失败，无法以 OpenVINO({rapidocr_device}) 配置启动。"
            ) from exc
        self._validate_rapidocr_backend(engine)
        return engine

    @staticmethod
    def _collect_rapidocr_execution_devices(engine: RapidOCR) -> Dict[str, List[str]]:
        execution_devices: Dict[str, List[str]] = {}
        for stage_name, attr_name in (
            ("det", "text_det"),
            ("cls", "text_cls"),
            ("rec", "text_rec"),
        ):
            stage = getattr(engine, attr_name, None)
            session = getattr(stage, "session", None)
            devices = _get_openvino_session_execution_devices(session)
            if devices:
                execution_devices[stage_name] = devices
        return execution_devices

    @staticmethod
    def _rapidocr_execution_matches_requested_device(
        requested_device: str,
        actual_devices: List[str],
    ) -> bool:
        normalized_actual = _normalize_openvino_devices(actual_devices)
        if not normalized_actual:
            return False
        if _openvino_device_expr_requests_gpu(requested_device):
            return _has_openvino_gpu_device(normalized_actual)
        if _openvino_device_expr_requests_cpu(requested_device):
            return any(device_name.startswith("CPU") for device_name in normalized_actual)
        if _openvino_device_expr_requests_npu(requested_device):
            return any(device_name.startswith("NPU") for device_name in normalized_actual)
        return False

    @staticmethod
    def _validate_rapidocr_execution_devices(
        cfg: Dict[str, Any],
        execution_devices: Dict[str, List[str]],
    ) -> None:
        mismatched_stages = [
            stage_name
            for stage_name in ("det", "cls", "rec")
            if not RapidOCRMixin._rapidocr_execution_matches_requested_device(
                str(
                    cfg.get(
                        f"{stage_name}_runtime_device_name",
                        cfg.get(f"{stage_name}_device_name", cfg.get("runtime_device_name", "AUTO")),
                    )
                ),
                execution_devices.get(stage_name, []),
            )
        ]
        if mismatched_stages:
            expected_devices = {
                stage_name: str(
                    cfg.get(
                        f"{stage_name}_runtime_device_name",
                        cfg.get(f"{stage_name}_device_name", cfg.get("runtime_device_name", "AUTO")),
                    )
                )
                for stage_name in ("det", "cls", "rec")
            }
            raise RuntimeError(
                "RapidOCR execution device validation failed: "
                f"expected_devices={expected_devices}, execution_devices={execution_devices}, "
                f"mismatched_stages={mismatched_stages}. No silent fallback is allowed."
            )

    def _ensure_ocr_opencl_preprocess(self, cfg: Dict[str, Any]) -> None:
        preprocess_backends = cfg.get("preprocess_backends", {}) or {}
        if "opencl" not in preprocess_backends.values():
            return

        name, vendor = _ensure_intel_opencl_device("RapidOCR preprocessing")
        self._ocr_opencl_device_name = name
        self._ocr_opencl_device_vendor = vendor

    @staticmethod
    def _resize_image_opencl(image: np.ndarray, width: int, height: int) -> np.ndarray:
        resized = cv2.resize(_to_opencv_umat(image), (int(width), int(height)))
        if isinstance(resized, cv2.UMat):
            return _as_contiguous_bgr_uint8(resized.get(), context="RapidOCR OpenCL resize")
        return _as_contiguous_bgr_uint8(np.asarray(resized), context="RapidOCR OpenCL resize")

    @staticmethod
    def _rapidocr_blob_from_image_opencl(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        mean_values: List[float],
        std_values: List[float],
    ) -> np.ndarray:
        prepared = _as_contiguous_bgr_uint8(image, context="RapidOCR OpenCL preprocess")
        mean = [float(value) for value in mean_values]
        std = [float(value) for value in std_values]
        if not std or max(std) - min(std) > 1e-6:
            raise RuntimeError(
                "RapidOCR OpenCL preprocess requires uniform std values per channel. "
                f"Got std={std_values}"
            )
        scalefactor = 1.0 / (255.0 * std[0])
        mean_pixels = tuple(float(value) * 255.0 for value in mean)
        blob = cv2.dnn.blobFromImage(
            _to_opencv_umat(prepared),
            scalefactor=scalefactor,
            size=(int(target_width), int(target_height)),
            mean=mean_pixels,
            swapRB=False,
            crop=False,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    @staticmethod
    def _rapidocr_resize_image_within_bounds_opencl(
        image: np.ndarray,
        min_side_len: float,
        max_side_len: float,
    ) -> Tuple[np.ndarray, float, float]:
        img = _as_contiguous_bgr_uint8(image, context="RapidOCR preprocess")
        h, w = img.shape[:2]
        ratio_h = ratio_w = 1.0

        max_value = max(h, w)
        if max_value > max_side_len:
            scale = float(max_side_len) / float(max_value)
            resize_h = int(round((h * scale) / 32.0) * 32)
            resize_w = int(round((w * scale) / 32.0) * 32)
            if resize_h <= 0 or resize_w <= 0:
                raise RapidOCRError("RapidOCR preprocess resize target is invalid.")
            img = RapidOCRMixin._resize_image_opencl(img, resize_w, resize_h)
            ratio_h = h / float(resize_h)
            ratio_w = w / float(resize_w)

        h, w = img.shape[:2]
        min_value = min(h, w)
        if min_value < min_side_len:
            scale = float(min_side_len) / float(min_value)
            resize_h = int(round((h * scale) / 32.0) * 32)
            resize_w = int(round((w * scale) / 32.0) * 32)
            if resize_h <= 0 or resize_w <= 0:
                raise RapidOCRError("RapidOCR preprocess resize target is invalid.")
            img = RapidOCRMixin._resize_image_opencl(img, resize_w, resize_h)
            ratio_h = h / float(resize_h)
            ratio_w = w / float(resize_w)
        return img, ratio_h, ratio_w

    @staticmethod
    def _rapidocr_det_resize_shape(
        image: np.ndarray,
        limit_side_len: int,
        limit_type: str,
    ) -> Optional[Tuple[int, int]]:
        h, w = image.shape[:2]
        if limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / float(max(h, w))
            else:
                ratio = 1.0
        else:
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / float(min(h, w))
            else:
                ratio = 1.0

        resize_h = int(round((h * ratio) / 32.0) * 32)
        resize_w = int(round((w * ratio) / 32.0) * 32)
        if resize_h <= 0 or resize_w <= 0:
            return None
        return resize_h, resize_w

    @staticmethod
    def _rapidocr_detect_opencl(detector: Any, image: np.ndarray) -> TextDetOutput:
        start_time = time.perf_counter()
        if image is None:
            raise ValueError("img is None")

        image_bgr = _as_contiguous_bgr_uint8(image, context="RapidOCR det")
        ori_img_shape = image_bgr.shape[0], image_bgr.shape[1]
        preprocess_op = detector.get_preprocess(max(image_bgr.shape[0], image_bgr.shape[1]))
        detector.preprocess_op = preprocess_op
        resize_shape = RapidOCRMixin._rapidocr_det_resize_shape(
            image_bgr,
            int(preprocess_op.limit_side_len),
            str(preprocess_op.limit_type),
        )
        if resize_shape is None:
            return TextDetOutput()
        resize_h, resize_w = resize_shape
        prepro_img = RapidOCRMixin._rapidocr_blob_from_image_opencl(
            image_bgr,
            target_width=resize_w,
            target_height=resize_h,
            mean_values=preprocess_op.mean.tolist(),
            std_values=preprocess_op.std.tolist(),
        )
        preds = detector.session(prepro_img)
        boxes, scores = detector.postprocess_op(preds, ori_img_shape)
        if len(boxes) < 1:
            return TextDetOutput()

        boxes = detector.sorted_boxes(boxes)
        return TextDetOutput(
            img=image_bgr,
            boxes=np.asarray(boxes, dtype=np.float32),
            scores=[float(score) for score in scores],
            elapse=time.perf_counter() - start_time,
        )

    def _rapidocr_preprocess_img(
        self,
        engine: RapidOCR,
        ori_img: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        runtime_cfg = self._rapidocr_runtime_cfg or {}
        preprocess_backends = runtime_cfg.get("preprocess_backends", {}) or {}
        if preprocess_backends.get("det") != "opencl":
            return engine.preprocess_img(ori_img)

        op_record: Dict[str, Any] = {}
        img, ratio_h, ratio_w = self._rapidocr_resize_image_within_bounds_opencl(
            ori_img,
            engine.min_side_len,
            engine.max_side_len,
        )
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        return img, op_record

    def _warmup_rapidocr_locked(self) -> None:
        engines = self._rapidocr_engines
        if not engines:
            raise RuntimeError("RapidOCR model is not loaded.")
        runtime_cfg = self._rapidocr_runtime_cfg or {}
        warmup_side = max(
            64,
            min(
                1280,
                max(
                    _as_int(runtime_cfg.get("max_side_len"), 960),
                    _as_int(runtime_cfg.get("det_limit_side_len"), 960),
                ),
            ),
        )
        warmup_image = np.full((warmup_side, warmup_side, 3), 255, dtype=np.uint8)
        font_scale = max(1.2, warmup_side / 512.0)
        thickness = max(2, warmup_side // 320)
        baseline_y = max(48, warmup_side // 3)
        cv2.putText(
            warmup_image,
            "rapidocr warmup",
            (max(12, warmup_side // 16), baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            warmup_image,
            "12345",
            (max(12, warmup_side // 8), min(warmup_side - 24, baseline_y * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (32, 32, 32),
            thickness,
            cv2.LINE_AA,
        )
        for engine in engines:
            self._run_rapidocr_builtin(engine, warmup_image)

    def _load_rapidocr_locked(self) -> None:
        config = self._load_rapidocr_openvino_config()
        rapidocr_params = self._build_rapidocr_runtime_params(config)
        engine_pool_size = max(1, self._ocr_stage_worker_count)
        engines = tuple(
            self._instantiate_rapidocr(dict(rapidocr_params)) for _ in range(engine_pool_size)
        )
        engine_pool: LifoQueue = LifoQueue(maxsize=len(engines))
        for engine in engines:
            engine_pool.put_nowait(engine)
        runtime_cfg = dict(config)
        runtime_cfg["engine_pool_size"] = len(engines)
        self._rapidocr_engine = engines[0]
        self._rapidocr_engines = engines
        self._rapidocr_engine_pool = engine_pool
        self._rapidocr_runtime_cfg = runtime_cfg
        if config.get("det_limit_type") == "min":
            LOG.warning(
                "RapidOCR Det.limit_type=min will upscale small images and may increase latency."
            )
        LOG.info(
            "RapidOCR ready: config=%s requested_device=%s runtime_device=%s hint=%s use_cls=%s max_side_len=%s "
            "det_limit=%s/%s rec_batch_num=%s cls_batch_num=%s ocr_admission=%s engine_pool=%s",
            self.rapidocr_config_path,
            config.get("requested_device_name"),
            config.get("device_name"),
            config.get("performance_hint"),
            config.get("use_cls"),
            config.get("max_side_len"),
            config.get("det_limit_type"),
            config.get("det_limit_side_len"),
            config.get("rec_batch_num"),
            config.get("cls_batch_num"),
            self._ocr_admission.capacity,
            len(engines),
        )

    def _unload_rapidocr_model_locked(self) -> None:
        self._rapidocr_engine = None
        self._rapidocr_engines = None
        self._rapidocr_engine_pool = None
        self._rapidocr_runtime_cfg = None
        self._ocr_opencl_device_name = None
        self._ocr_opencl_device_vendor = None

    def _unload_rapidocr_model(self) -> None:
        with self._rapidocr_load_lock:
            self._unload_rapidocr_model_locked()

    def _ensure_rapidocr_loaded(self) -> None:
        with self._rapidocr_load_lock:
            if self._rapidocr_engine is not None:
                return
            self._load_family_with_process_lock("ocr", self._load_rapidocr_locked)

    @contextmanager
    def _borrow_rapidocr_engine(self) -> Iterator[RapidOCR]:
        pool = self._rapidocr_engine_pool
        if pool is None:
            raise RuntimeError("RapidOCR model pool is not loaded.")
        engine = pool.get(block=True)
        try:
            yield cast(RapidOCR, engine)
        finally:
            pool.put_nowait(engine)

    def _run_rapidocr_with_pooled_engine(
        self,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> OCRResult:
        if self._rapidocr_engine_pool is None:
            raise RuntimeError("RapidOCR model is not loaded.")
        prepared = _as_contiguous_bgr_uint8(image, context="OCR")
        with self._borrow_rapidocr_engine() as engine:
            return self._run_rapidocr_builtin(engine, prepared, cancel_event=cancel_event)

    def _infer_ocr(self, image: np.ndarray) -> OCRResult:
        return self._run_rapidocr_with_pooled_engine(image)

    @staticmethod
    def _ocr_result_from_raw(raw_result: Any) -> OCRResult:
        ocr_items: Any
        if all(hasattr(raw_result, field) for field in ("boxes", "txts", "scores")):
            boxes_data = getattr(raw_result, "boxes", None)
            txts_data = getattr(raw_result, "txts", None) or ()
            scores_data = getattr(raw_result, "scores", None) or ()
            if boxes_data is None:
                ocr_items = []
            else:
                count = min(len(boxes_data), len(txts_data), len(scores_data))
                ocr_items = [
                    (boxes_data[idx], txts_data[idx], scores_data[idx]) for idx in range(count)
                ]
        elif isinstance(raw_result, tuple):
            ocr_items = raw_result[0]
        else:
            ocr_items = raw_result

        if not ocr_items:
            return OCRResult(texts=[], scores=[], boxes=[])

        texts: List[str] = []
        scores: List[str] = []
        boxes: List[OCRBox] = []

        for item in ocr_items:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            points = np.array(item[0], dtype=np.float32)
            if points.shape != (4, 2):
                continue
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            boxes.append(
                OCRBox(
                    x=str(round(float(x_min), 2)),
                    y=str(round(float(y_min), 2)),
                    width=str(round(float(x_max - x_min), 2)),
                    height=str(round(float(y_max - y_min), 2)),
                )
            )
            texts.append(str(item[1]))
            scores.append(f"{float(item[2]):.2f}")

        return OCRResult(texts=texts, scores=scores, boxes=boxes)

    def _rapidocr_detect_and_pad(
        self,
        engine: RapidOCR,
        image: np.ndarray,
        op_record: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any], TextDetOutput]:
        padded, updated_record = apply_vertical_padding(
            image,
            op_record,
            engine.width_height_ratio,
            engine.min_height,
        )
        detector = engine.text_det
        preprocess_backend = str(getattr(detector, "_mt_preprocess_backend", "cpu")).strip().lower()
        if preprocess_backend == "opencl":
            return padded, updated_record, self._rapidocr_detect_opencl(detector, padded)
        return padded, updated_record, detector(padded)

    def _rapidocr_crop_regions(self, image: np.ndarray, det_boxes: np.ndarray) -> List[np.ndarray]:
        boxes = np.asarray(det_boxes, dtype=np.float32)
        if boxes.size == 0:
            return []
        return [
            get_rotate_crop_image(
                image,
                np.array(box, dtype=np.float32, copy=True),
            )
            for box in boxes
        ]

    @staticmethod
    def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _InferenceCancelled("OCR task cancelled")

    async def _rapidocr_crop_regions_async(
        self,
        image: np.ndarray,
        det_boxes: np.ndarray,
        cancel_event: Optional[threading.Event],
    ) -> List[np.ndarray]:
        boxes = np.asarray(det_boxes, dtype=np.float32)
        if boxes.size == 0:
            return []

        max_inflight = max(1, getattr(self._shared_cpu_executor, "_max_workers", 1))
        cropped: List[np.ndarray] = []
        for start in range(0, len(boxes), max_inflight):
            self._raise_if_cancelled(cancel_event)
            current_boxes = boxes[start : start + max_inflight]
            current_results = await asyncio.gather(
                *[
                    self._run_in_executor(
                        self._shared_cpu_executor,
                        get_rotate_crop_image,
                        image,
                        np.array(box, dtype=np.float32, copy=True),
                    )
                    for box in current_boxes
                ]
            )
            cropped.extend(current_results)
        return cropped

    @staticmethod
    def _rapidocr_classify_single_image(classifier: Any, image: np.ndarray) -> Tuple[str, float]:
        norm_img = classifier.resize_norm_img(image)[np.newaxis, ...].astype(np.float32)
        prob_out = classifier.session(norm_img)
        cls_result = list(classifier.postprocess_op(prob_out))
        if len(cls_result) != 1:
            raise RapidOCRError(
                "RapidOCR cls single-image fallback output mismatch: "
                f"expected=1 got={len(cls_result)}"
            )
        label, score = cls_result[0]
        return str(label), float(score)

    @staticmethod
    def _rapidocr_build_sorted_batch_indices(
        images: List[np.ndarray],
        batch_num: int,
    ) -> List[List[int]]:
        if not images:
            return []

        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        indices = np.argsort(np.array(width_list))
        batch_size = max(1, int(batch_num))
        return [
            [int(indices[ino]) for ino in range(beg_img_no, min(len(images), beg_img_no + batch_size))]
            for beg_img_no in range(0, len(images), batch_size)
        ]

    @staticmethod
    def _rapidocr_classify_batch(
        engine: RapidOCR,
        images: List[np.ndarray],
        batch_indices: List[int],
    ) -> Tuple[List[int], List[np.ndarray], List[Tuple[str, float]]]:
        classifier = engine.text_cls
        batch_images = [images[original_index] for original_index in batch_indices]
        if not batch_images:
            return batch_indices, [], []

        norm_img_batch = np.stack(
            [classifier.resize_norm_img(image) for image in batch_images],
            axis=0,
        ).astype(np.float32)
        prob_out = classifier.session(norm_img_batch)
        cls_result = list(classifier.postprocess_op(prob_out))
        if len(cls_result) != len(batch_indices):
            LOG.warning(
                "RapidOCR cls output size mismatch: expected=%s got=%s; "
                "falling back to single-image classification.",
                len(batch_indices),
                len(cls_result),
            )
            cls_result = [
                RapidOCRMixin._rapidocr_classify_single_image(classifier, image)
                for image in batch_images
            ]

        rotated_images = list(batch_images)
        normalized_results: List[Tuple[str, float]] = []
        for relative_index, (label, score) in enumerate(cls_result):
            label_str = str(label)
            score_float = float(score)
            normalized_results.append((label_str, score_float))
            if "180" in label_str and score_float > classifier.cls_thresh:
                rotated_images[relative_index] = cv2.rotate(
                    rotated_images[relative_index],
                    cv2.ROTATE_180,
                )
        return batch_indices, rotated_images, normalized_results

    @staticmethod
    def _rapidocr_recognize_single_batch(
        recognizer: Any,
        images: List[np.ndarray],
        *,
        return_word_box: bool,
    ) -> TextRecOutput:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            return _empty_text_rec_output()

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        img_num = len(img_list)
        rec_res: List[Tuple[_TextClsResult, Optional[_TextRecWordResult]]] = [
            (("", 0.0), None) for _ in range(img_num)
        ]

        img_c, img_h, img_w = recognizer.rec_image_shape[:3]
        max_wh_ratio = img_w / img_h
        wh_ratio_list: List[float] = []
        ordered_images = [img_list[int(indices[ino])] for ino in range(img_num)]
        for image in ordered_images:
            h, w = image.shape[:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)

        target_width = int(max(img_w, round(img_h * max_wh_ratio)))
        preprocess_backend = str(getattr(recognizer, "_mt_preprocess_backend", "cpu")).strip().lower()
        norm_img_batch: List[np.ndarray] = []
        for image in ordered_images:
            if preprocess_backend == "opencl":
                ratio = image.shape[1] / float(image.shape[0])
                resized_w = min(target_width, int(np.ceil(img_h * ratio)))
                blob = RapidOCRMixin._rapidocr_blob_from_image_opencl(
                    image,
                    target_width=resized_w,
                    target_height=int(img_h),
                    mean_values=[0.5, 0.5, 0.5],
                    std_values=[0.5, 0.5, 0.5],
                )[0]
                padding_im = np.zeros((img_c, img_h, target_width), dtype=np.float32)
                padding_im[:, :, :resized_w] = blob[:, :, :resized_w]
                norm_img_batch.append(padding_im)
            else:
                norm_img_batch.append(recognizer.resize_norm_img(image, max_wh_ratio))

        batch = np.stack(norm_img_batch, axis=0).astype(np.float32)
        preds = recognizer.session(batch)
        line_results, word_results = recognizer.postprocess_op(
            preds,
            return_word_box,
            wh_ratio_list=wh_ratio_list,
            max_wh_ratio=max_wh_ratio,
        )

        for result_index, one_res in enumerate(line_results):
            original_index = int(indices[result_index])
            if return_word_box:
                rec_res[original_index] = (
                    (str(one_res[0]), float(one_res[1])),
                    cast(_TextRecWordResult, word_results[result_index]),
                )
            else:
                rec_res[original_index] = ((str(one_res[0]), float(one_res[1])), None)

        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        if recognizer.cfg.lang_type == LangRec.ARABIC:
            txts = reorder_bidi_for_display(txts)

        normalized_word_results: List[_TextRecWordResult] = []
        for word_result in all_word_results:
            normalized_word_results.append(word_result or ("", 1.0, None))

        return TextRecOutput(
            imgs=img_list,
            txts=cast(Any, tuple(str(text) for text in txts)),
            scores=[float(score) for score in scores],
            word_results=cast(Any, tuple(normalized_word_results)),
            elapse=time.perf_counter() - start_time,
            viser=VisRes(lang_type=recognizer.cfg.lang_type, font_path=recognizer.cfg.font_path),
        )

    @staticmethod
    def _rapidocr_recognize(engine: RapidOCR, images: List[np.ndarray]) -> TextRecOutput:
        rec_res = RapidOCRMixin._rapidocr_recognize_single_batch(
            engine.text_rec,
            images,
            return_word_box=engine.return_word_box,
        )
        if rec_res.txts is None:
            raise RapidOCRError("The text recognize result is empty")
        return rec_res

    @staticmethod
    def _rapidocr_recognize_batch(
        engine: RapidOCR,
        images: List[np.ndarray],
        batch_indices: List[int],
    ) -> Tuple[List[int], TextRecOutput]:
        batch_images = [images[original_index] for original_index in batch_indices]
        return batch_indices, RapidOCRMixin._rapidocr_recognize(engine, batch_images)

    async def _rapidocr_cls_and_rotate_async(
        self,
        engine: RapidOCR,
        images: List[np.ndarray],
        cancel_event: Optional[threading.Event],
    ) -> Tuple[List[np.ndarray], TextClsOutput]:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            output = TextClsOutput(img_list=[], cls_res=[], elapse=0.0)
            return [], output

        batch_indices_list = self._rapidocr_build_sorted_batch_indices(
            img_list,
            engine.text_cls.cls_batch_num,
        )
        batch_outputs = []
        max_inflight = max(1, self._ocr_stage_worker_count)
        for start in range(0, len(batch_indices_list), max_inflight):
            self._raise_if_cancelled(cancel_event)
            batch_outputs.extend(
                await asyncio.gather(
                    *[
                        self._run_in_executor(
                            self._ocr_cls_executor,
                            self._rapidocr_classify_batch,
                            engine,
                            img_list,
                            batch_indices,
                        )
                        for batch_indices in batch_indices_list[start : start + max_inflight]
                    ]
                )
            )

        cls_res: List[Tuple[str, float]] = [("", 0.0)] * len(img_list)
        for batch_indices, rotated_images, batch_results in batch_outputs:
            if len(rotated_images) != len(batch_indices) or len(batch_results) != len(batch_indices):
                raise RapidOCRError(
                    "RapidOCR cls batch output mismatch: "
                    f"expected={len(batch_indices)} got_images={len(rotated_images)} "
                    f"got_results={len(batch_results)}"
                )
            for relative_index, original_index in enumerate(batch_indices):
                img_list[original_index] = rotated_images[relative_index]
                cls_res[original_index] = batch_results[relative_index]

        output = TextClsOutput(
            img_list=img_list,
            cls_res=cls_res,
            elapse=time.perf_counter() - start_time,
        )
        if output.img_list is None:
            raise RapidOCRError("The text classifier is empty")
        return output.img_list, output

    async def _rapidocr_recognize_async(
        self,
        engine: RapidOCR,
        images: List[np.ndarray],
        cancel_event: Optional[threading.Event],
    ) -> TextRecOutput:
        start_time = time.perf_counter()
        img_list = list(images)
        if not img_list:
            return _empty_text_rec_output()

        batch_indices_list = self._rapidocr_build_sorted_batch_indices(
            img_list,
            engine.text_rec.rec_batch_num,
        )
        batch_outputs = []
        max_inflight = max(1, self._ocr_stage_worker_count)
        for start in range(0, len(batch_indices_list), max_inflight):
            self._raise_if_cancelled(cancel_event)
            batch_outputs.extend(
                await asyncio.gather(
                    *[
                        self._run_in_executor(
                            self._ocr_rec_executor,
                            self._rapidocr_recognize_batch,
                            engine,
                            img_list,
                            batch_indices,
                        )
                        for batch_indices in batch_indices_list[start : start + max_inflight]
                    ]
                )
            )

        texts: List[str] = [""] * len(img_list)
        scores: List[float] = [0.0] * len(img_list)
        word_results: List[_TextRecWordResult] = [("", 1.0, None) for _ in range(len(img_list))]
        viser: Any = None
        for batch_indices, batch_output in batch_outputs:
            batch_texts = list(batch_output.txts or ())
            batch_scores = [float(score) for score in (batch_output.scores or ())]
            batch_word_results = [
                cast(_TextRecWordResult, item)
                for item in list(batch_output.word_results or ())
            ]
            if (
                len(batch_texts) != len(batch_indices)
                or len(batch_scores) != len(batch_indices)
                or len(batch_word_results) != len(batch_indices)
            ):
                raise RapidOCRError(
                    "RapidOCR rec batch output mismatch: "
                    f"expected={len(batch_indices)} got_txts={len(batch_texts)} "
                    f"got_scores={len(batch_scores)} got_words={len(batch_word_results)}"
                )
            if viser is None:
                viser = batch_output.viser
            for relative_index, original_index in enumerate(batch_indices):
                texts[original_index] = str(batch_texts[relative_index])
                scores[original_index] = batch_scores[relative_index]
                word_results[original_index] = batch_word_results[relative_index]

        return TextRecOutput(
            imgs=img_list,
            txts=cast(Any, tuple(texts)),
            scores=scores,
            word_results=cast(Any, tuple(word_results)),
            elapse=time.perf_counter() - start_time,
            viser=viser,
        )

    def _run_rapidocr_builtin(
        self,
        engine: RapidOCR,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> OCRResult:
        runtime_cfg = self._rapidocr_runtime_cfg or {}
        self._raise_if_cancelled(cancel_event)
        total_started_at = time.perf_counter()
        self._raise_if_cancelled(cancel_event)
        raw_result = engine(image)
        total_ms = (time.perf_counter() - total_started_at) * 1000.0
        slow_threshold_ms = max(
            1000.0,
            min(float(self._ocr_execution_timeout_seconds) * 500.0, 5000.0),
        )
        if total_ms >= slow_threshold_ms:
            result_boxes = getattr(raw_result, "boxes", None)
            box_count = 0 if result_boxes is None else int(len(result_boxes))
            LOG.warning(
                "RapidOCR slow request: total=%.1fms runtime_device=%s use_cls=%s max_side_len=%s det_limit=%s/%s rec_batch_num=%s cls_batch_num=%s boxes=%s",
                total_ms,
                runtime_cfg.get("runtime_device_name", "CPU"),
                runtime_cfg.get("use_cls"),
                runtime_cfg.get("max_side_len"),
                runtime_cfg.get("det_limit_type"),
                runtime_cfg.get("det_limit_side_len"),
                runtime_cfg.get("rec_batch_num"),
                runtime_cfg.get("cls_batch_num"),
                box_count,
            )
        return self._ocr_result_from_raw(raw_result)

    async def _infer_ocr_async(
        self,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> OCRResult:
        if self._rapidocr_engine_pool is None:
            raise RuntimeError("RapidOCR model is not loaded.")
        return await self._run_in_executor(
            self._ocr_det_executor,
            self._run_rapidocr_with_pooled_engine,
            image,
            cancel_event,
        )

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self._acquire_image_request_slot("OCR")
        try:
            self._acquire_non_text_family_lease("ocr")
            try:
                self._ensure_rapidocr_loaded()
                self._acquire_admission(self._ocr_admission, "OCR")
                try:
                    return self._infer_ocr(image)
                finally:
                    self._ocr_admission.release()
            finally:
                self._release_non_text_family_lease("ocr")
        finally:
            self._release_image_request_slot()

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        self._acquire_image_request_slot("OCR")
        cancel_event = threading.Event()
        try:
            await self._acquire_non_text_family_lease_async("ocr")
            try:
                await self._run_control(self._ensure_rapidocr_loaded)
                await self._acquire_admission_async(self._ocr_admission, "OCR")
                try:
                    task: asyncio.Task[OCRResult] = asyncio.create_task(
                        self._infer_ocr_async(image, cancel_event=cancel_event)
                    )
                    return await self._await_with_timeout_and_cooperative_cancel(
                        task,
                        cancel_event=cancel_event,
                        timeout_seconds=self._ocr_execution_timeout_seconds,
                        task_name="OCR task",
                    )
                finally:
                    self._ocr_admission.release()
            finally:
                self._release_non_text_family_lease("ocr")
        finally:
            self._release_image_request_slot()
