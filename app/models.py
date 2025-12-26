# app/models.py
import logging
import os
import threading
import gc
import time
from typing import List, Callable, Optional

import numpy as np
import torch
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from rapidocr_onnxruntime import RapidOCR

# 引入项目自带的 clip 包
import clip
from schemas import (
    OCRBox,
    OCRResult,
    FacialArea,
    RepresentResult
)

# --- 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.warning(f"当前推理设备: {DEVICE} (2GB 显存极致优化版)")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", os.path.join(_PROJECT_ROOT, "models"))

MODEL_NAME = os.environ.get("MODEL_NAME", "antelopev2")
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

# 限制图片最大边长，防止预处理 OOM
MAX_IMAGE_SIDE = 2560

logging.getLogger(__name__).setLevel(logging.WARNING)

class AIModels:
    """
    针对 MX150 (2GB) 优化的单例模型管理类。
    """
    def __init__(self):
        logging.warning(f"初始化 AIModels (Single Slot Mode).")

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.clip_cache_root = os.path.join(MODEL_BASE_PATH, "qa-clip")

        # 移除了 self.ocr_root 的定义，因为使用内置模型不再需要指定外部路径

        os.makedirs(self.clip_cache_root, exist_ok=True)

        # 模型容器
        self.clip_model = None
        self.clip_preprocess = None
        self.ocr_engine = None
        self.face_engine = None

        # 状态标记
        self.current_loaded_type: Optional[str] = None

        # 线程锁与定时器
        self._lock = threading.RLock()
        self._release_timer: Optional[threading.Timer] = None

    def _clean_gpu_memory(self):
        """强制清理 GPU 显存"""
        if DEVICE == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _cancel_timer(self):
        if self._release_timer:
            self._release_timer.cancel()
            self._release_timer = None

    def release_all_models(self, reason: str = "主动释放"):
        with self._lock:
            self._cancel_timer()
            cleaned = False

            if self.ocr_engine:
                del self.ocr_engine
                self.ocr_engine = None
                cleaned = True

            if self.face_engine:
                del self.face_engine
                self.face_engine = None
                cleaned = True

            if self.clip_model:
                del self.clip_model
                self.clip_model = None
                self.clip_preprocess = None
                cleaned = True

            if cleaned:
                logging.info(f"[{reason}] 已释放所有模型资源。")
                self.current_loaded_type = None
                self._clean_gpu_memory()

    def release_models(self):
        self.release_all_models(reason="外部调用 release")

    # --- 内部加载逻辑 ---

    def _load_clip_safe(self):
        logging.info(f"正在加载 CLIP ({CLIP_MODEL_NAME})...")
        try:
            # 1. Load CLIP
            model, preprocess = clip.load_from_name(
                CLIP_MODEL_NAME,
                device=DEVICE,
                download_root=self.clip_cache_root
            )
            model.eval()

            # 不再手动调用 .half()，依赖加载时的自动处理

            self.clip_model = model
            self.clip_preprocess = preprocess
            self.current_loaded_type = "clip"
        except Exception as e:
            logging.error(f"CLIP 加载失败: {e}")
            raise e

    def _load_ocr_safe(self):
        logging.info("正在加载 RapidOCR (使用内置模型)...")
        use_cuda = (DEVICE == "cuda")

        try:
            # 【核心修改】：不传入 model_path，默认使用 pip 包内自带的模型
            self.ocr_engine = RapidOCR(
                det_use_cuda=use_cuda,
                cls_use_cuda=use_cuda,
                rec_use_cuda=use_cuda
            )
            self.current_loaded_type = "ocr"
            logging.info(f"RapidOCR 加载成功 (CUDA={use_cuda})")
        except Exception as e:
            logging.error(f"OCR 加载失败 (详情): {repr(e)}")
            # 尝试回退到 CPU 模式
            if use_cuda:
                logging.warning("尝试回退到 CPU 模式加载 OCR...")
                try:
                    self.ocr_engine = RapidOCR(
                        det_use_cuda=False,
                        cls_use_cuda=False,
                        rec_use_cuda=False
                    )
                    self.current_loaded_type = "ocr"
                    logging.info("RapidOCR CPU 回退模式加载成功")
                    return
                except Exception as ex_cpu:
                    logging.error(f"OCR CPU 回退也失败: {ex_cpu}")
            raise e

    def _load_face_safe(self):
        logging.info(f"正在加载 InsightFace...")
        try:
            providers = ['CUDAExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
            app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)

            # 2GB 显存核心配置：det_size 320 是 MX150 的极限
            app.prepare(ctx_id=0, det_size=(320, 320))

            self.face_engine = app
            self.current_loaded_type = "face"
        except Exception as e:
            logging.error(f"InsightFace 加载失败: {e}")
            raise e

    def _switch_to(self, target_type: str):
        with self._lock:
            self._cancel_timer()
            if self.current_loaded_type == target_type:
                return

            if self.current_loaded_type is not None:
                self.release_all_models(reason="模型切换抢占")

            self._clean_gpu_memory()
            if target_type == "clip":
                self._load_clip_safe()
            elif target_type == "ocr":
                self._load_ocr_safe()
            elif target_type == "face":
                self._load_face_safe()

    def _schedule_auto_release(self, delay: float = 5.0):
        self._cancel_timer()
        self._release_timer = threading.Timer(delay, lambda: self.release_all_models(reason="超时自动释放"))
        self._release_timer.start()

    def _preprocess_image_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def ensure_clip_text_model_loaded(self):
        pass

    # --- 业务接口 ---

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        with self._lock:
            try:
                self._switch_to("ocr")
                image = self._preprocess_image_size(image)
                # RapidOCR 返回值结构: result, elapse
                result, _ = self.ocr_engine(image)

                if not result: return OCRResult(texts=[], scores=[], boxes=[])

                texts, scores, boxes = [], [], []
                for line in result:
                    # line format: [coordinates, text, score]
                    coords, text, score = line
                    if not text: continue
                    texts.append(str(text))
                    scores.append(f"{float(score):.2f}")
                    coords = np.array(coords, dtype=np.int32)
                    x_min, y_min = np.min(coords, axis=0)
                    x_max, y_max = np.max(coords, axis=0)
                    boxes.append(OCRBox(
                        x=str(x_min), y=str(y_min),
                        width=str(x_max - x_min), height=str(y_max - y_min)
                    ))
                return OCRResult(texts=texts, scores=scores, boxes=boxes)
            except Exception as e:
                logging.error(f"OCR Inference Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return OCRResult(texts=[], scores=[], boxes=[])

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        with self._lock:
            try:
                self._switch_to("face")
                image = self._preprocess_image_size(image)
                faces = self.face_engine.get(image)
                results = []
                for face in faces:
                    bbox = np.array(face.bbox).astype(int)
                    results.append(RepresentResult(
                        embedding=[float(x) for x in face.normed_embedding],
                        facial_area=FacialArea(x=bbox[0], y=bbox[1], w=bbox[2]-bbox[0], h=bbox[3]-bbox[1]),
                        face_confidence=float(face.det_score)
                    ))
                return results
            except Exception as e:
                logging.error(f"Face Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return []

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        with self._lock:
            try:
                self._switch_to("clip")
                image_input = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

                # 兼容性处理：如果设备是 CUDA，输入转为 Half
                if DEVICE == "cuda":
                    image_input = image_input.half()

                with torch.no_grad():
                    features = self.clip_model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)

                self._schedule_auto_release(5.0)
                return features.cpu().numpy().flatten().tolist()
            except Exception as e:
                logging.error(f"CLIP Image Inference Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return [0.0] * CLIP_EMBEDDING_DIMS

    def get_text_embedding(self, text: str) -> List[float]:
        with self._lock:
            try:
                self._switch_to("clip")
                text_input = clip.tokenize([text], context_length=CONTEXT_LENGTH).to(DEVICE)

                with torch.no_grad():
                    features = self.clip_model.encode_text(text_input)
                    features = features / features.norm(dim=-1, keepdim=True)

                self._schedule_auto_release(5.0)
                return features.cpu().numpy().flatten().tolist()
            except Exception as e:
                logging.error(f"CLIP Text Inference Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return [0.0] * CLIP_EMBEDDING_DIMS