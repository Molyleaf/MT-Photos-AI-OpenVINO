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

    原则：
    1. 同一时间显存中只能有一个模型 (Single Slot Strategy)。
    2. CLIP 文本请求后保留 5 秒，以应对连续搜索；其他请求强制抢占。
    3. 所有大图在推理前强制缩放。
    """
    def __init__(self):
        logging.warning(f"初始化 AIModels (Single Slot Mode).")

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.clip_cache_root = os.path.join(MODEL_BASE_PATH, "qa-clip")
        os.makedirs(self.clip_cache_root, exist_ok=True)

        # 模型容器
        self.clip_model = None
        self.clip_preprocess = None
        self.ocr_engine = None
        self.face_engine = None

        # 状态标记
        # "clip", "ocr", "face", None
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
        """取消待执行的释放任务"""
        if self._release_timer:
            self._release_timer.cancel()
            self._release_timer = None

    def release_all_models(self, reason: str = "主动释放"):
        """
        释放所有模型，清空显存槽位。
        """
        with self._lock:
            self._cancel_timer()

            cleaned = False

            # 释放 OCR
            if self.ocr_engine:
                del self.ocr_engine
                self.ocr_engine = None
                cleaned = True

            # 释放 Face
            if self.face_engine:
                del self.face_engine
                self.face_engine = None
                cleaned = True

            # 释放 CLIP
            if self.clip_model:
                del self.clip_model
                self.clip_model = None
                self.clip_preprocess = None
                cleaned = True

            if cleaned:
                logging.info(f"[{reason}] 已释放所有模型资源。")
                self.current_loaded_type = None
                self._clean_gpu_memory()

    # 兼容 server.py 调用的接口
    def release_models(self):
        self.release_all_models(reason="外部调用 release")

    # --- 内部加载逻辑 (优化版) ---

    def _load_clip_safe(self):
        """优化：先载入 CPU 转 FP16，再移入 GPU，防止峰值爆炸"""
        logging.info(f"正在加载 CLIP ({CLIP_MODEL_NAME})...")
        try:
            # 1. Load to CPU first
            model, preprocess = clip.load_from_name(
                CLIP_MODEL_NAME,
                device="cpu",
                download_root=self.clip_cache_root
            )
            model.eval()

            # 2. Convert to Half & Move to CUDA
            if DEVICE == "cuda":
                model.half()
                model.to(DEVICE)

            self.clip_model = model
            self.clip_preprocess = preprocess
            self.current_loaded_type = "clip"
        except Exception as e:
            logging.error(f"CLIP 加载失败: {e}")
            raise e

    def _load_ocr_safe(self):
        logging.info("正在加载 RapidOCR...")
        use_cuda = (DEVICE == "cuda")
        try:
            self.ocr_engine = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
            self.current_loaded_type = "ocr"
        except Exception as e:
            logging.error(f"OCR 加载失败: {e}")
            raise e

    def _load_face_safe(self):
        logging.info(f"正在加载 InsightFace...")
        try:
            providers = ['CUDAExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
            app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)

            # 【优化】2GB 显存核心保命配置：det_size 降为 320
            # 640x640 在 MX150 上极易 OOM
            app.prepare(ctx_id=0, det_size=(320, 320))

            self.face_engine = app
            self.current_loaded_type = "face"
        except Exception as e:
            logging.error(f"InsightFace 加载失败: {e}")
            raise e

    def _switch_to(self, target_type: str):
        """
        核心调度器：切换显存槽位到指定模型。
        如果当前已经是该模型，则跳过；否则先清空再加载。
        """
        with self._lock:
            # 1. 任何新请求都会取消之前的释放定时器（如果是 CLIP 续费，或者被抢占）
            self._cancel_timer()

            # 2. 如果已经加载了目标模型，直接返回
            if self.current_loaded_type == target_type:
                return

            # 3. 如果加载了其他模型，先强制释放
            if self.current_loaded_type is not None:
                logging.warning(f"切换模型: {self.current_loaded_type} -> {target_type}")
                self.release_all_models(reason="模型切换抢占")

            # 4. 执行加载
            self._clean_gpu_memory() # 加载前确保干净
            if target_type == "clip":
                self._load_clip_safe()
            elif target_type == "ocr":
                self._load_ocr_safe()
            elif target_type == "face":
                self._load_face_safe()

    def _schedule_auto_release(self, delay: float = 5.0):
        """计划在 delay 秒后释放模型（主要用于 CLIP）"""
        self._cancel_timer()
        self._release_timer = threading.Timer(delay, lambda: self.release_all_models(reason="超时自动释放"))
        self._release_timer.start()

    def _preprocess_image_size(self, image: np.ndarray) -> np.ndarray:
        """
        主动大图缩放：防止超大分辨率图片在推理时撑爆显存。
        限制最大边长为 MAX_IMAGE_SIDE (2560)。
        """
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            logging.info(f"图片尺寸过大 ({w}x{h})，主动缩放至 {new_w}x{new_h}")
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    # --- 兼容性空方法 (server.py 启动时会调用) ---
    def ensure_clip_text_model_loaded(self):
        # 启动时不预加载，改为 lazy loading
        pass

        # --- 业务接口 ---

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        with self._lock: # 锁住整个推理过程
            try:
                # 1. 抢占槽位
                self._switch_to("ocr")

                # 2. 缩放检查
                image = self._preprocess_image_size(image)

                # 3. 推理
                result, _ = self.ocr_engine(image)

                # 4. OCR 往往是单次任务，推理完建议直接释放，或者稍微保留等待后续?
                # 为了极致稳定，建议用完即走，或者让 server.py 的 idle handle 处理
                # 这里我们选择保留，等待 server 的 idle_timeout 或者被抢占

                if not result: return OCRResult(texts=[], scores=[], boxes=[])

                texts, scores, boxes = [], [], []
                for line in result:
                    coords, text, score = line
                    texts.append(str(text))
                    scores.append(f"{float(score):.2f}")
                    # 坐标处理
                    coords = np.array(coords, dtype=np.int32)
                    x_min, y_min = np.min(coords, axis=0)
                    x_max, y_max = np.max(coords, axis=0)
                    boxes.append(OCRBox(
                        x=str(x_min), y=str(y_min),
                        width=str(x_max - x_min), height=str(y_max - y_min)
                    ))
                return OCRResult(texts=texts, scores=scores, boxes=boxes)
            except Exception as e:
                logging.error(f"OCR Error: {e}")
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
                # CLIP 的 resize 已经在 preprocess 里了，但如果是超大图，PIL to Tensor 也可能炸
                # 这里假设传入的是 PIL，preprocess 会负责 resize 到 224

                # 预处理
                image_input = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
                if DEVICE == "cuda":
                    image_input = image_input.half()

                with torch.no_grad():
                    features = self.clip_model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)

                # CLIP 处理完，如果是图像搜索，后续可能还有，保留5秒
                self._schedule_auto_release(5.0)

                return features.cpu().numpy().flatten().tolist()
            except Exception as e:
                logging.error(f"CLIP Image Error: {e}")
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

                # 文本搜索通常是连续的，保留 5 秒等待下一次搜索
                self._schedule_auto_release(5.0)

                return features.cpu().numpy().flatten().tolist()
            except Exception as e:
                logging.error(f"CLIP Text Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return [0.0] * CLIP_EMBEDDING_DIMS