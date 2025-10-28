# app/models.py
import logging
import os
import queue
import threading
from typing import List, Tuple, Optional, Callable

import numpy as np
import openvino as ov
from PIL import Image
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR

import clip
from clip.utils import image_transform, MODEL_INFO
from schemas import (
    OCRBox,
    OCRResult,
    FacialArea,
    RepresentResult
)
import gc # 垃圾回收

# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
logging.info(f"模型根目录 (MODEL_BASE_PATH): {MODEL_BASE_PATH}")

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "1"))
MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

# 注意：logging.basicConfig 应该在 server.py 中配置
logging.getLogger(__name__).setLevel(logging.WARNING)


class AIModels:
    """封装所有 AI 模型加载和推理逻辑的类（并发安全版）。"""
    def __init__(self):
        logging.warning(f"正在初始化AI模型实例 (尚未加载)，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.qa_clip_path = os.path.join(MODEL_BASE_PATH, "qa-clip", "openvino")

        # 【修复】基础编译模型
        self.clip_vision_model: Optional[ov.CompiledModel] = None
        self.clip_text_model: Optional[ov.CompiledModel] = None
        self.clip_image_preprocessor = None

        # 【修复】实例池
        self.face_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.ocr_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.clip_vision_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.clip_text_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)

        # 【修复】使用可重入锁(RLock)解决死锁问题
        self._load_lock = threading.RLock()

        # 【修复】使用独立的加载标志
        self.clip_text_model_loaded = False
        self.clip_vision_model_loaded = False
        self.ocr_models_loaded = False
        self.face_models_loaded = False

    def _empty_queue(self, q: queue.Queue):
        """辅助函数：安全清空队列"""
        if q is None: return
        while not q.empty():
            try:
                item = q.get_nowait()
                del item
            except queue.Empty:
                break
            except Exception as e:
                logging.warning(f"清空队列时出错: {e}")

    def release_models(self):
        """
        (空闲释放)
        从内存中释放所有“按需”模型 (Face, OCR, Vision CLIP)。
        【修复】此方法 *不会* 释放 Text CLIP 模型。
        """
        if not (self.face_models_loaded or self.ocr_models_loaded or self.clip_vision_model_loaded):
            logging.debug("没有按需模型需要释放。")
            return

        logging.warning("--- 正在释放 (按需) AI 模型和实例池 ---")
        try:
            self._empty_queue(self.face_pool)
            self._empty_queue(self.ocr_pool)
            self._empty_queue(self.clip_vision_pool)
            logging.warning("按需实例池已清空。")

            if self.clip_vision_model:
                del self.clip_vision_model
                self.clip_vision_model = None

            self.clip_image_preprocessor = None

            gc.collect()

            logging.warning("(按需) 模型已成功从内存中释放。")
        except Exception as e:
            logging.warning(f"释放 (按需) 模型时出现错误: {e}", exc_info=True)
        finally:
            # 【修复】重置“按需”模型的标志
            self.face_models_loaded = False
            self.ocr_models_loaded = False
            self.clip_vision_model_loaded = False

    def release_all_models(self):
        """
        (关闭释放)
        释放 *所有* 模型，包括常驻的 Text CLIP 模型。
        仅在应用关闭时调用。
        """
        logging.warning("--- 正在释放 *所有* AI 模型 (应用关闭) ---")
        # 1. 释放所有按需模型
        self.release_models()

        # 2. 释放常驻的 Text CLIP 模型
        try:
            if self.clip_text_model_loaded:
                logging.warning("释放 Text CLIP 模型...")
                self._empty_queue(self.clip_text_pool)
                if self.clip_text_model:
                    del self.clip_text_model
                    self.clip_text_model = None
                gc.collect()
                logging.warning("Text CLIP 模型已释放。")
        except Exception as e:
            logging.warning(f"释放 Text CLIP 模型时出现错误: {e}", exc_info=True)
        finally:
            self.clip_text_model_loaded = False

    def _load_insightface(self) -> FaceAnalysis:
        # --- 【修复 v5】: 解决 Error 127 静默失败和日志误导问题 ---
        logging.warning("OpenVINO EP (for Insightface) 在 Server 2025 上不兼容 (将跳过)。")
        logging.warning("Insightface 将显式加载通用的 'CPUExecutionProvider' (无 OpenVINO 加速)。")
        try:
            providers_generic_cpu = ['CPUExecutionProvider']
            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers_generic_cpu)
            face_app.prepare(ctx_id=0, det_size=(64, 64))
            logging.warning("InsightFace 实例已在通用 CPU (回退模式) 上成功加载。")
            return face_app
        except Exception as final_fallback_e:
            # 如果连这个都失败了，那就是个大问题
            logging.critical(f"InsightFace 在通用 CPU (回退) 模式下也加载失败: {final_fallback_e}", exc_info=True)
            raise final_fallback_e
        # --- 修复结束 ---

    @staticmethod
    def _load_rapidocr() -> RapidOCR:
        # (此处的逻辑已正确：AUTO -> CPU (OpenVINO)
        try:
            logging.debug(f"加载 RapidOCR (OpenVINO) (尝试使用 {INFERENCE_DEVICE})...")
            # RapidOCR 会自动处理 'AUTO' 设备
            ocr = RapidOCR(device_name=INFERENCE_DEVICE)
            logging.warning(f"RapidOCR 实例在 {INFERENCE_DEVICE} (OpenVINO) 上加载成功。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 实例失败 (尝试使用 {INFERENCE_DEVICE}): {e}", exc_info=True)
            logging.warning("RapidOCR (OpenVINO) 加载失败，将回退到 CPU (OpenVINO)...")
            try:
                ocr = RapidOCR(device_name="CPU")
                logging.warning("RapidOCR 实例已在 CPU (OpenVINO 回退模式) 上成功加载。")
                return ocr
            except Exception as fallback_e:
                logging.critical(f"RapidOCR 在 CPU (回退) 模式下也加载失败: {fallback_e}", exc_info=True)
                raise fallback_e

    # 【修复】拆分 _load_qa_clip

    def _compile_clip_vision_model(self) -> Tuple[Callable, ov.CompiledModel]:
        """编译并返回 Vision 模型及预处理器"""
        logging.warning(f"正在加载 QA-CLIP Vision ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            vision_model_path = os.path.join(self.qa_clip_path, "openvino_image_fp16.xml")
            if not os.path.exists(vision_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 Vision 模型文件。")

            config_vision = {"PERFORMANCE_HINT": "THROUGHPUT"}
            logging.warning(f"编译 Vision 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_vision['PERFORMANCE_HINT']})...")
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config_vision)

            image_preprocessor = image_transform(MODEL_INFO[MODEL_ARCH]['input_resolution'])

            if vision_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型维度不匹配！")

            logging.warning(f"QA-CLIP Vision ({MODEL_ARCH}) 基础模型编译成功。")
            return image_preprocessor, vision_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP Vision 模型时发生严重错误: {e}", exc_info=True)
            raise

    def _compile_clip_text_model(self) -> ov.CompiledModel:
        """编译并返回 Text 模型"""
        logging.warning(f"正在加载 QA-CLIP Text ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            text_model_path = os.path.join(self.qa_clip_path, "openvino_text_fp16.xml")
            if not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 Text 模型文件。")

            # 【修复】为 Text 模型硬编码 LATENCY 模式
            config_text = {"PERFORMANCE_HINT": "LATENCY"}
            logging.warning(f"编译 Text 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_text['PERFORMANCE_HINT']})...")
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config_text)

            if text_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型维度不匹配！")

            logging.warning(f"QA-CLIP Text ({MODEL_ARCH}) 基础模型编译成功。")
            return text_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP Text 模型时发生严重错误: {e}", exc_info=True)
            raise

    # 【修复】按需加载 (Ensure) 方法

    def ensure_clip_text_model_loaded(self):
        """(常驻) 确保 Text CLIP 模型已加载。"""
        if self.clip_text_model_loaded:
            return
        with self._load_lock:
            if self.clip_text_model_loaded:
                return
            logging.warning("--- 正在加载 CLIP Text 模型 (常驻) ---")
            try:
                self.clip_text_model = self._compile_clip_text_model()
                for i in range(INFERENCE_WORKERS):
                    self.clip_text_pool.put(self.clip_text_model.create_infer_request())
                self.clip_text_model_loaded = True
                logging.warning(f"--- CLIP Text 模型已准备就绪 ({INFERENCE_WORKERS} 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 Text CLIP 模型失败: {e}", exc_info=True)
                raise

    def ensure_clip_vision_model_loaded(self):
        """(按需) 确保 Vision CLIP 模型已加载。"""
        if self.clip_vision_model_loaded:
            return
        with self._load_lock:
            if self.clip_vision_model_loaded:
                return
            logging.warning("--- 正在加载 CLIP Vision 模型 (按需) ---")
            try:
                self.clip_image_preprocessor, self.clip_vision_model = self._compile_clip_vision_model()
                for i in range(INFERENCE_WORKERS):
                    self.clip_vision_pool.put(self.clip_vision_model.create_infer_request())
                self.clip_vision_model_loaded = True
                logging.warning(f"--- CLIP Vision 模型已准备就绪 ({INFERENCE_WORKERS} 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 Vision CLIP 模型失败: {e}", exc_info=True)
                raise

    def ensure_ocr_models_loaded(self):
        """(按需) 确保 OCR 模型已加载。"""
        if self.ocr_models_loaded:
            return
        with self._load_lock:
            if self.ocr_models_loaded:
                return
            logging.warning("--- 正在加载 RapidOCR 模型 (按需) ---")
            try:
                for i in range(INFERENCE_WORKERS):
                    logging.warning(f"加载 RapidOCR 实例 {i+1}/{INFERENCE_WORKERS}...")
                    self.ocr_pool.put(self._load_rapidocr())
                self.ocr_models_loaded = True
                logging.warning(f"--- RapidOCR 模型已准备就绪 ({INFERENCE_WORKERS} 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 RapidOCR 模型失败: {e}", exc_info=True)
                raise

    def ensure_face_models_loaded(self):
        """(按需) 确保 FaceAnalysis 模型已加载。"""
        if self.face_models_loaded:
            return
        with self._load_lock:
            if self.face_models_loaded:
                return
            logging.warning("--- 正在加载 FaceAnalysis 模型 (按需) ---")
            try:
                for i in range(INFERENCE_WORKERS):
                    logging.warning(f"加载 FaceAnalysis 实例 {i+1}/{INFERENCE_WORKERS}...")
                    self.face_pool.put(self._load_insightface())
                self.face_models_loaded = True
                logging.warning(f"--- FaceAnalysis 模型已准备就绪 ({INFERENCE_WORKERS} 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 FaceAnalysis 模型失败: {e}", exc_info=True)
                raise

    # 【修复】修改 get_... 方法以调用其各自的 ensure

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        self.ensure_face_models_loaded() # 按需加载
        face_analyzer = None
        try:
            face_analyzer = self.face_pool.get(timeout=10)
            faces = face_analyzer.get(image)
            results = []
            for face in faces:
                bbox = np.array(face.bbox).astype(int)
                x1, y1, x2, y2 = bbox
                facial_area = FacialArea(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
                results.append(RepresentResult(
                    embedding=[float(x) for x in face.normed_embedding],
                    facial_area=facial_area,
                    face_confidence=float(face.det_score)
                ))
            return results
        except queue.Empty:
            logging.error("获取 FaceAnalysis 实例超时（池已空且无法加载）")
            raise Exception("FaceAnalysis model pool timeout")
        except Exception as e:
            # 调试级别，因为“未找到人脸”是正常情况
            logging.debug(f"处理人脸识别时出错或未找到人脸: {e}", exc_info=False)
            return []
        finally:
            if face_analyzer:
                self.face_pool.put(face_analyzer)

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_ocr_models_loaded() # 按需加载
        ocr_engine = None
        try:
            ocr_engine = self.ocr_pool.get(timeout=10)
            ocr_raw_output = ocr_engine(image)

            if not isinstance(ocr_raw_output, tuple) or len(ocr_raw_output) < 1:
                logging.warning(f"RapidOCR 返回了意外的格式: {type(ocr_raw_output)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            ocr_result, _ = ocr_raw_output

            if ocr_result is None:
                # 这是正常情况（没有文本）
                logging.debug("RapidOCR 返回 None 结果 (无文本)。")
                return OCRResult(texts=[], scores=[], boxes=[])

            if not isinstance(ocr_result, list):
                logging.warning(f"RapidOCR 的结果部分不是列表: {type(ocr_result)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            if len(ocr_result) == 0:
                # 正常情况（无文本）
                return OCRResult(texts=[], scores=[], boxes=[])

            texts, scores, boxes = [], [], []
            def to_fixed(num): return str(round(num, 2))

            for res in ocr_result:
                if not (isinstance(res, list) or isinstance(res, tuple)) or len(res) < 3:
                    logging.warning(f"RapidOCR 返回了格式不正确的识别结果项: {res}")
                    continue
                if not (isinstance(res[0], list) or isinstance(res[0], np.ndarray)) or len(res[0]) != 4:
                    logging.warning(f"RapidOCR 返回了格式不正确的坐标点: {res[0]}")
                    continue

                texts.append(str(res[1]))
                scores.append(f"{float(res[2]):.2f}")
                try:
                    points = np.array(res[0], dtype=np.int32)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                except Exception as box_err:
                    logging.warning(f"处理 OCR 边界框时出错: {box_err} - 原始点: {res[0]}")
                    continue

                boxes.append(OCRBox(
                    x=to_fixed(x_min),
                    y=to_fixed(y_min),
                    width=to_fixed(x_max - x_min),
                    height=to_fixed(y_max - y_min)
                ))

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except queue.Empty:
            logging.error("获取 RapidOCR 实例超时（池已空且无法加载）")
            raise Exception("RapidOCR model pool timeout")
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return OCRResult(texts=[], scores=[], boxes=[])
        finally:
            if ocr_engine:
                self.ocr_pool.put(ocr_engine)

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_clip_vision_model_loaded() # 按需加载
        infer_request = None
        try:
            if not self.clip_image_preprocessor or not self.clip_vision_model:
                logging.error("CLIP Vision 模型未正确加载。")
                raise Exception("CLIP Vision model not loaded")

            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            infer_request = self.clip_vision_pool.get(timeout=10)

            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]
            return [float(x) for x in embedding.flatten()]
        except queue.Empty:
            logging.error("获取 CLIP Vision 实例超时（池已空且无法加载）")
            raise Exception("CLIP Vision model pool timeout")
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if infer_request:
                self.clip_vision_pool.put(infer_request)

    def get_text_embedding(self, text: str) -> List[float]:
        self.ensure_clip_text_model_loaded() # (常驻) 检查以确保加载
        infer_request = None
        try:
            if not self.clip_text_model:
                logging.error("CLIP Text 模型未正确加载。")
                raise Exception("CLIP Text model not loaded")

            inputs_tensor = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs_tensor.numpy()
            pad_index = clip._tokenizer.vocab['[PAD]']
            attention_mask = np.array(input_ids != pad_index).astype(np.int64)

            infer_request = self.clip_text_pool.get(timeout=10)

            input_name_0 = self.clip_text_model.inputs[0].any_name
            input_name_1 = self.clip_text_model.inputs[1].any_name
            inputs_dict = {
                input_name_0: input_ids,
                input_name_1: attention_mask
            }
            results = infer_request.infer(inputs_dict)
            embedding = results[self.clip_text_model.outputs[0]]
            return [float(x) for x in embedding.flatten()]
        except queue.Empty:
            logging.error("获取 CLIP Text 实例超时（池已空且无法加载）")
            raise Exception("CLIP Text model pool timeout")
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text[:50]}...' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if infer_request:
                self.clip_text_pool.put(infer_request)