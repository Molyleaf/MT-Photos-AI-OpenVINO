import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from rapidocr import EngineType, RapidOCR
from rapidocr.utils.log import logger as RAPIDOCR_LOGGER

from .common import (
    _AdmissionController,
    _InferenceCancelled,
    NonTextFamily,
    _as_contiguous_bgr_uint8,
)
from .constants import LOG
from .schemas import OCRBox, OCRResult


class _SuppressExpectedRapidOCRNoTextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().strip() != "The text detection result is empty"


class RapidOCRMixin(ABC):
    _rapidocr_load_lock: Any
    _rapidocr_run_lock: Any
    _rapidocr_engine: Optional[RapidOCR]
    _rapidocr_runtime_cfg: Optional[Dict[str, Any]]
    _ocr_executor: ThreadPoolExecutor
    _ocr_admission: _AdmissionController
    _ocr_execution_timeout_seconds: int

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
    def _ensure_non_text_task_executors_ready(self) -> None:
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
    def _configure_rapidocr_logger() -> None:
        if getattr(RAPIDOCR_LOGGER, "_mt_expected_no_text_filter", None) is not None:
            return

        message_filter = _SuppressExpectedRapidOCRNoTextFilter()
        RAPIDOCR_LOGGER.addFilter(message_filter)
        for handler in RAPIDOCR_LOGGER.handlers:
            handler.addFilter(message_filter)
        setattr(RAPIDOCR_LOGGER, "_mt_expected_no_text_filter", message_filter)

    @staticmethod
    def _cfg_value(section: Any, key: str, default: Any = None) -> Any:
        if section is None:
            return default
        getter = getattr(section, "get", None)
        if callable(getter):
            value = getter(key, default)
        else:
            value = getattr(section, key, default)
        return default if value is None else value

    @staticmethod
    def _normalize_engine_name(value: Any) -> str:
        return str(getattr(value, "value", value)).strip().lower()

    @staticmethod
    def _format_model_name(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        return Path(raw).name

    def _build_rapidocr_init_params(self) -> Dict[str, Any]:
        return {
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
        }

    def _snapshot_rapidocr_runtime_config(self, engine: RapidOCR) -> Dict[str, Any]:
        cfg = getattr(engine, "cfg", None)
        if cfg is None:
            raise RuntimeError("RapidOCR initialized without runtime cfg metadata.")

        det_cfg = getattr(cfg, "Det", None)
        cls_cfg = getattr(cfg, "Cls", None)
        rec_cfg = getattr(cfg, "Rec", None)
        return {
            "det_engine_type": self._normalize_engine_name(
                self._cfg_value(det_cfg, "engine_type")
            ),
            "cls_engine_type": self._normalize_engine_name(
                self._cfg_value(cls_cfg, "engine_type")
            ),
            "rec_engine_type": self._normalize_engine_name(
                self._cfg_value(rec_cfg, "engine_type")
            ),
            "det_model_name": self._format_model_name(self._cfg_value(det_cfg, "model_path")),
            "cls_model_name": self._format_model_name(self._cfg_value(cls_cfg, "model_path")),
            "rec_model_name": self._format_model_name(self._cfg_value(rec_cfg, "model_path")),
            "rec_keys_name": self._format_model_name(self._cfg_value(rec_cfg, "rec_keys_path")),
        }

    @staticmethod
    def _validate_rapidocr_backend(engine: RapidOCR) -> None:
        cfg = getattr(engine, "cfg", None)
        if cfg is None:
            raise RuntimeError("RapidOCR initialized without runtime cfg metadata.")

        backend_errors: List[str] = []
        for section_name in ("Det", "Cls", "Rec"):
            section_cfg = getattr(cfg, section_name, None)
            engine_name = RapidOCRMixin._normalize_engine_name(
                getattr(section_cfg, "engine_type", None)
            )
            if engine_name != EngineType.OPENVINO.value:
                backend_errors.append(f"{section_name}.engine_type={engine_name or 'missing'}")

        if backend_errors:
            raise RuntimeError(
                "RapidOCR backend validation failed after initialization: "
                f"{', '.join(backend_errors)}. No silent fallback is allowed."
            )

    def _instantiate_rapidocr(self) -> RapidOCR:
        self._configure_rapidocr_logger()
        init_params = self._build_rapidocr_init_params()
        try:
            engine = RapidOCR(params=init_params)
        except Exception as exc:
            LOG.warning(
                "RapidOCR native initialization/download check failed: %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError("RapidOCR 初始化失败，无法以 OpenVINO 默认配置启动。") from exc
        self._validate_rapidocr_backend(engine)
        return engine

    def _warmup_rapidocr_locked(self) -> None:
        engine = self._rapidocr_engine
        if engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")

        warmup_image = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.putText(
            warmup_image,
            "rapidocr warmup",
            (24, 176),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            warmup_image,
            "12345",
            (48, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (32, 32, 32),
            3,
            cv2.LINE_AA,
        )
        self._run_rapidocr_builtin(engine, warmup_image)

    def _load_rapidocr_locked(self) -> None:
        engine = self._instantiate_rapidocr()
        runtime_cfg = self._snapshot_rapidocr_runtime_config(engine)
        self._rapidocr_engine = engine
        self._rapidocr_runtime_cfg = runtime_cfg
        LOG.info(
            "RapidOCR ready: det=%s(%s) cls=%s(%s) rec=%s(%s) dict=%s ocr_admission=%s",
            runtime_cfg.get("det_engine_type"),
            runtime_cfg.get("det_model_name") or "default",
            runtime_cfg.get("cls_engine_type"),
            runtime_cfg.get("cls_model_name") or "default",
            runtime_cfg.get("rec_engine_type"),
            runtime_cfg.get("rec_model_name") or "default",
            runtime_cfg.get("rec_keys_name") or "default",
            self._ocr_admission.capacity,
        )

    def _unload_rapidocr_model_locked(self) -> None:
        self._rapidocr_engine = None
        self._rapidocr_runtime_cfg = None

    def _unload_rapidocr_model(self) -> None:
        with self._rapidocr_load_lock:
            self._unload_rapidocr_model_locked()

    def _ensure_rapidocr_loaded(self) -> None:
        with self._rapidocr_load_lock:
            if self._rapidocr_engine is not None and self._rapidocr_runtime_cfg is not None:
                return
            self._load_family_serialized("ocr", self._load_rapidocr_locked)

    def _run_rapidocr(
        self,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> OCRResult:
        engine = self._rapidocr_engine
        if engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")
        prepared = _as_contiguous_bgr_uint8(image, context="OCR")
        with self._rapidocr_run_lock:
            return self._run_rapidocr_builtin(engine, prepared, cancel_event=cancel_event)

    def _infer_ocr(self, image: np.ndarray) -> OCRResult:
        return self._run_rapidocr(image)

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
            LOG.warning("RapidOCR slow request: total=%.1fms boxes=%s", total_ms, box_count)
        return self._ocr_result_from_raw(raw_result)

    async def _infer_ocr_async(
        self,
        image: np.ndarray,
        cancel_event: Optional[threading.Event] = None,
    ) -> OCRResult:
        if self._rapidocr_engine is None:
            raise RuntimeError("RapidOCR model is not loaded.")
        self._ensure_non_text_task_executors_ready()
        return await self._run_in_executor(
            self._ocr_executor,
            self._run_rapidocr,
            image,
            cancel_event,
        )

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        with self._non_text_request_scope(
            family="ocr",
            label="OCR",
            admission=self._ocr_admission,
            ensure_loaded=self._ensure_rapidocr_loaded,
        ):
            return self._infer_ocr(image)

    async def get_ocr_results_async(self, image: np.ndarray) -> OCRResult:
        cancel_event = threading.Event()
        async with self._non_text_request_scope_async(
            family="ocr",
            label="OCR",
            admission=self._ocr_admission,
            ensure_loaded=self._ensure_rapidocr_loaded,
        ):
            task: asyncio.Task[OCRResult] = asyncio.create_task(
                self._infer_ocr_async(image, cancel_event=cancel_event)
            )
            return await self._await_with_timeout_and_cooperative_cancel(
                task,
                cancel_event=cancel_event,
                timeout_seconds=self._ocr_execution_timeout_seconds,
                task_name="OCR task",
            )
