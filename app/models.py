# app/models.py
import logging
import os
import threading
import gc
from typing import List, Optional

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

# --- 日志净化区域 ---
# 压制常见的库日志噪音
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("insightface").setLevel(logging.WARNING)

# --- 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.warning(f"当前推理设备: {DEVICE} (2GB 显存极致优化版 + CLIP热备)")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", os.path.join(_PROJECT_ROOT, "models"))

MODEL_NAME = os.environ.get("MODEL_NAME", "antelopev2")
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

# 限制图片最大边长，防止预处理 OOM
MAX_IMAGE_SIDE = 2560

# 自动释放时间配置
AUTO_RELEASE_DELAY = 5.0   # OCR/Face 完成后多久释放
STANDBY_LOAD_DELAY = 10.0  # 释放完成后多久自动加载 CLIP (热备)

class AIModels:
    """
    针对 MX150 (2GB) 优化的单例模型管理类。
    包含 CLIP 热备 (Hot-Standby) 逻辑。
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
        self.current_loaded_type: Optional[str] = None

        # 线程锁与定时器
        self._lock = threading.RLock()
        self._release_timer: Optional[threading.Timer] = None
        self._standby_timer: Optional[threading.Timer] = None # 新增：热备加载定时器

    def _clean_gpu_memory(self):
        """强制清理 GPU 显存"""
        if DEVICE == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _cancel_timers(self):
        """取消所有挂起的定时任务（释放或加载）"""
        if self._release_timer:
            self._release_timer.cancel()
            self._release_timer = None

        if self._standby_timer:
            self._standby_timer.cancel()
            self._standby_timer = None

    def release_all_models(self, reason: str = "主动释放", schedule_standby: bool = True):
        """
        释放所有模型。
        :param schedule_standby: 释放后是否计划自动加载 CLIP 热备
        """
        with self._lock:
            self._cancel_timers() # 停止之前的计时
            cleaned = False

            if self.ocr_engine:
                del self.ocr_engine
                self.ocr_engine = None
                cleaned = True

            if self.face_engine:
                del self.face_engine
                self.face_engine = None
                cleaned = True

            # 注意：如果是 CLIP 且当前策略是热备，这里也会被释放
            # 只有在 _switch_to 需要腾位置时，或者 lifespan 关闭时才会调用这里
            if self.clip_model:
                del self.clip_model
                self.clip_model = None
                self.clip_preprocess = None
                cleaned = True

            if cleaned:
                self.current_loaded_type = None
                self._clean_gpu_memory()
                logging.info(f"[{reason}] 已释放所有模型资源。")

            # 【核心逻辑】释放完成后，计划加载 CLIP 热备
            if schedule_standby and DEVICE == "cuda":
                logging.info(f"将在 {STANDBY_LOAD_DELAY}s 后加载 CLIP 进入热备状态...")
                self._standby_timer = threading.Timer(STANDBY_LOAD_DELAY, self._load_clip_standby_worker)
                self._standby_timer.start()

    def release_models(self):
        # 外部调用的释放（如 API /restart），不自动触发热备，或者根据需求触发
        # 这里设定为触发热备，保持服务活性
        self.release_all_models(reason="外部调用 release", schedule_standby=True)

    def _load_clip_standby_worker(self):
        """热备加载的工作线程函数"""
        with self._lock:
            # 如果当前已经加载了其他东西（可能在等待期间来了新请求），则放弃加载
            if self.current_loaded_type is not None:
                return

            logging.info(">>> 触发热备机制：自动加载 CLIP 模型...")
            try:
                self._load_clip_safe()
            except Exception as e:
                logging.error(f"热备加载 CLIP 失败: {e}")

    # --- 内部加载逻辑 ---

    def _load_clip_safe(self):
        if self.clip_model is not None: return

        logging.info(f"正在加载 CLIP ({CLIP_MODEL_NAME})...")
        try:
            # 捕获可能的标准输出噪音
            model, preprocess = clip.load_from_name(
                CLIP_MODEL_NAME,
                device=DEVICE,
                download_root=self.clip_cache_root
            )
            model.eval()
            self.clip_model = model
            self.clip_preprocess = preprocess
            self.current_loaded_type = "clip"
            logging.info("CLIP 模型加载完毕 (Ready).")
        except Exception as e:
            logging.error(f"CLIP 加载失败: {e}")
            raise e

    def _load_ocr_safe(self):
        logging.info("正在加载 RapidOCR...")
        use_cuda = (DEVICE == "cuda")
        try:
            # 屏蔽 RapidOCR 初始化时的部分日志
            logging.getLogger().setLevel(logging.ERROR)
            self.ocr_engine = RapidOCR(
                det_use_cuda=use_cuda,
                cls_use_cuda=use_cuda,
                rec_use_cuda=use_cuda
            )
            logging.getLogger().setLevel(logging.WARNING) # 恢复

            self.current_loaded_type = "ocr"
            logging.info(f"RapidOCR 加载成功 (CUDA={use_cuda})")
        except Exception as e:
            logging.getLogger().setLevel(logging.WARNING) # 确保恢复
            logging.error(f"OCR 加载失败: {repr(e)}")
            # 回退逻辑...
            if use_cuda:
                self.ocr_engine = RapidOCR(det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False)
                self.current_loaded_type = "ocr"
                return
            raise e

    def _load_face_safe(self):
        logging.info(f"正在加载 InsightFace...")
        try:
            providers = ['CUDAExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
            app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)
            app.prepare(ctx_id=0, det_size=(320, 320))
            self.face_engine = app
            self.current_loaded_type = "face"
            logging.info("InsightFace 加载成功.")
        except Exception as e:
            logging.error(f"InsightFace 加载失败: {e}")
            raise e

    def _switch_to(self, target_type: str):
        with self._lock:
            # 1. 无论如何，只要有请求进来，先取消“热备加载计时器”
            if self._standby_timer:
                self._standby_timer.cancel()
                self._standby_timer = None

            # 2. 如果目标就是当前类型，直接取消“释放计时器”并返回（续命）
            if self.current_loaded_type == target_type:
                if self._release_timer:
                    self._release_timer.cancel()
                    self._release_timer = None
                return

            # 3. 如果当前有模型且类型不同，必须释放
            # 特殊情况：如果当前是 CLIP (可能是热备的)，而目标是 OCR/Face
            # 需要立即释放 CLIP 给重型任务腾出显存
            if self.current_loaded_type is not None:
                # 注意：这里调用 release_all_models 时传入 False，防止递归或冲突
                # 因为我们马上就要加载新模型，不需要调度热备
                self.release_all_models(reason=f"切换模型: {self.current_loaded_type}->{target_type}", schedule_standby=False)

            # 4. 加载新模型
            self._clean_gpu_memory()
            if target_type == "clip":
                self._load_clip_safe()
            elif target_type == "ocr":
                self._load_ocr_safe()
            elif target_type == "face":
                self._load_face_safe()

    def _schedule_auto_release(self):
        """安排在 5秒后释放模型 (用于 OCR 和 Face)"""
        if self._release_timer:
            self._release_timer.cancel()

        # 只有 OCR 和 Face 需要释放后重新进入热备循环
        # CLIP 自身不需要释放，因为它就是热备目标
        self._release_timer = threading.Timer(AUTO_RELEASE_DELAY, lambda: self.release_all_models(reason="任务完成自动释放", schedule_standby=True))
        self._release_timer.start()

    def _preprocess_image_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def ensure_clip_text_model_loaded(self):
        # 启动时调用，直接加载 CLIP
        with self._lock:
            self._switch_to("clip")

    # --- 业务接口 ---

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        with self._lock:
            try:
                self._switch_to("ocr")
                image = self._preprocess_image_size(image)
                result, _ = self.ocr_engine(image)

                # 必须调度释放，以便腾出空间给 CLIP 热备
                self._schedule_auto_release()

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

                # 【修复】之前遗漏了这里，现在加上自动释放
                self._schedule_auto_release()

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
                if DEVICE == "cuda":
                    image_input = image_input.half()

                with torch.no_grad():
                    features = self.clip_model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)

                # CLIP 请求结束后，不应该调度释放
                # 因为 CLIP 是我们的热备模型，它应该一直呆在显存里，除非 OCR/Face 来抢
                # 只有当用户显式调用释放接口时才释放
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

                # CLIP 驻留，不释放
                return features.cpu().numpy().flatten().tolist()
            except Exception as e:
                logging.error(f"CLIP Text Inference Error: {e}")
                self.release_all_models(reason="Error Recovery")
                return [0.0] * CLIP_EMBEDDING_DIMS