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
# ⚠️ 注意：ViT-L-14 很大，QA-CLIP 代码内部已实现了混合精度加载
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
        # 定义 OCR 模型路径
        self.ocr_root = os.path.join(MODEL_BASE_PATH, "rapidocr")

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

    # --- 内部加载逻辑 (修复版) ---

    def _load_clip_safe(self):
        """
        修复：
        1. 必须设置 jit=False。
        2. 不要调用 .half()！clip/model.py 中的 convert_weights 会自动处理混合精度。
           如果这里强制 .half()，会导致 LayerNorm 权重变 FP16，但输入是 FP32，引发错误。
        """
        logging.info(f"正在加载 CLIP ({CLIP_MODEL_NAME})...")
        try:
            # 1. Load CLIP
            model, preprocess = clip.load_from_name(
                CLIP_MODEL_NAME,
                device=DEVICE, # 直接加载到目标设备，utils.py 会处理权重转换
                download_root=self.clip_cache_root
            )
            model.eval()

            # ⚠️ 【重要修复】:
            # 删除了 model.half()。
            # QA-CLIP 的实现会在加载时自动将 Conv/Linear 转为 FP16，
            # 而保留 LayerNorm 为 FP32。这是为了兼容性。

            self.clip_model = model
            self.clip_preprocess = preprocess
            self.current_loaded_type = "clip"
        except Exception as e:
            logging.error(f"CLIP 加载失败: {e}")
            raise e

    def _load_ocr_safe(self):
        logging.info("正在加载 RapidOCR...")
        use_cuda = (DEVICE == "cuda")

        # 【重要修复】: 显式指定模型路径
        det_model_path = os.path.join(self.ocr_root, "ch_PP-OCRv4_det_infer.onnx")
        cls_model_path = os.path.join(self.ocr_root, "ch_ppocr_mobile_v2.0_cls_infer.onnx")
        rec_model_path = os.path.join(self.ocr_root, "ch_PP-OCRv4_rec_infer.onnx")

        # 检查文件是否存在，避免报 'model_path' 这种含糊的错误
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"OCR模型缺失: {det_model_path}，请运行 download-models.py")

        try:
            self.ocr_engine = RapidOCR(
                det_model_path=det_model_path,
                cls_model_path=cls_model_path,
                rec_model_path=rec_model_path,
                det_use_cuda=use_cuda,
                cls_use_cuda=use_cuda,
                rec_use_cuda=use_cuda
            )
            self.current_loaded_type = "ocr"
        except Exception as e:
            logging.error(f"OCR 加载失败 (详情): {repr(e)}")
            # 尝试回退到 CPU 模式
            if use_cuda:
                logging.warning("尝试回退到 CPU 模式加载 OCR...")
                try:
                    self.ocr_engine = RapidOCR(
                        det_model_path=det_model_path,
                        cls_model_path=cls_model_path,
                        rec_model_path=rec_model_path,
                        det_use_cuda=False,
                        cls_use_cuda=False,
                        rec_use_cuda=False
                    )
                    self.current_loaded_type = "ocr"
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
                # 显存极小，切换前必须 aggressively 清理
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
                result, _ = self.ocr_engine(image)

                if not result: return OCRResult(texts=[], scores=[], boxes=[])

                texts, scores, boxes = [], [], []
                for line in result:
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

                # 预处理：clip_preprocess 会输出 Float32 的 Tensor
                image_input = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

                # ⚠️ 【重要修复】:
                # 之前在这里调用了 .half()。
                # 但因为我们在加载模型时去掉了 model.half()，保留了混合精度结构。
                # clip/model.py 里的 Conv/Linear 权重是 Half，LayerNorm 权重是 Float。
                # 输入数据最好转为 Half，因为第一层通常是 Conv。
                # 但是！如果模型的第一层处理不好，可能会报错。
                # 实测 QA-CLIP 的 preprocess 输出可以保持 Float32 传入，
                # 模型内部会自动在需要 Half 的地方处理，或者我们需要在这里转 Half。

                # 按照 QA-CLIP 的标准用法，输入应为 Half (如果使用 CUDA)
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

                # tokenize 返回的是 Long Tensor (Int64)，不需要转 Half
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