import asyncio
import logging
import os
import threading
import time
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
from rapidocr.utils.log import logger as RAPIDOCR_LOGGER
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
    _normalize_non_text_openvino_device,
    _normalize_rapidocr_limit_type,
)
from .constants import (
    LOG,
    RAPIDOCR_CLS_MOBILE_V2_FILE,
    RAPIDOCR_V5_DICT_FILE,
    RAPIDOCR_V5_MOBILE_DET_FILE,
    RAPIDOCR_V5_MOBILE_REC_FILE,
)


class _SuppressExpectedRapidOCRNoTextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().strip() != "The text detection result is empty"


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
    _ocr_stage_worker_count: int
    _ocr_det_executor: ThreadPoolExecutor
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

    @staticmethod
    def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _InferenceCancelled("OCR task cancelled")

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

