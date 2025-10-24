# app/models.py
import os
from typing import List, Any, Dict
import logging

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from PIL import Image

# --- 修正 1: 导入 app/clip 目录下的 QA-CLIP 库 ---
import clip
from clip.utils import image_transform, _MODEL_INFO
# --- 结束修正 1 ---

# --- 修正: 导入共享的 schemas.py 文件 ---
from schemas import (
    OCRBox,
    OCRResult,
    FacialArea,
    RepresentResult
)
# --- 结束修正 ---


# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

# 假定 QA-CLIP 结构同 Chinese-CLIP ViT-L-14
MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768

# QA-CLIP (ViT-L-14) 使用 77 的上下文长度
CONTEXT_LENGTH = 77

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic 模型定义 (用于 API 响应) ---
# --- 修正: 所有模型定义已移至 schemas.py ---
# --- 结束修正 ---


class AIModels:
    """封装所有 AI 模型加载和推理逻辑的类。"""
    def __init__(self):
        logging.info(f"正在初始化AI模型实例 (尚未加载)，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # 定义模型文件的根路径
        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        # 路径指向 qa-clip (由新的转换脚本创建)
        self.qa_clip_path = os.path.join(MODEL_BASE_PATH, "qa-clip", "openvino")

        # 初始化模型持有者为 None
        self.face_analyzer = None
        self.ocr_engine = None
        self.clip_image_preprocessor = None
        self.clip_vision_model = None
        self.clip_text_model = None

        # 标志位
        self.models_loaded = False

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        # 这个函数现在不再被 get_image/text_embedding 调用
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1e-6
        return vector / norm

    def ensure_models_loaded(self):
        """确保模型已加载，如果未加载，则加载它们。"""
        if not self.models_loaded:
            logging.warning("模型未加载。正在触发按需加载...")
            self.load_models()

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        # ... (此函数不变) ...
        self.ensure_models_loaded()
        try:
            faces = self.face_analyzer.get(image)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                facial_area = FacialArea(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
                results.append(RepresentResult(
                    embedding=[float(x) for x in face.normed_embedding],
                    facial_area=facial_area,
                    face_confidence=float(face.det_score)
                ))
            return results
        except Exception as e:
            logging.warning(f"处理人脸识别时出错: {e}", exc_info=True)
            return []


    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_models_loaded()
        try:
            # --- 添加诊断日志 ---
            logging.debug(f"输入 OCR 引擎的图像 shape: {image.shape}")
            ocr_raw_output = self.ocr_engine(image)
            logging.debug(f"RapidOCR 原始输出: {ocr_raw_output}")
            # --- 结束添加 ---

            # 检查 ocr_raw_output 是否是预期的元组格式
            if not isinstance(ocr_raw_output, tuple) or len(ocr_raw_output) < 1:
                logging.error(f"RapidOCR 返回了意外的格式: {type(ocr_raw_output)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            ocr_result, _ = ocr_raw_output # 假设 ocr_engine 返回元组

            if ocr_result is None:
                logging.warning("RapidOCR 返回 None 结果。")
                return OCRResult(texts=[], scores=[], boxes=[])

            if not isinstance(ocr_result, list):
                logging.error(f"RapidOCR 的结果部分不是列表: {type(ocr_result)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            if len(ocr_result) == 0:
                logging.info("RapidOCR 未检测到文本。")
                # 返回空结果是正常的
                return OCRResult(texts=[], scores=[], boxes=[])


            texts, scores, boxes = [], [], []
            def to_fixed(num): return str(round(num, 2))

            for res in ocr_result:
                # 添加更健壮的检查
                if not (isinstance(res, list) or isinstance(res, tuple)) or len(res) < 3:
                    logging.warning(f"RapidOCR 返回了格式不正确的识别结果项: {res}")
                    continue
                if not (isinstance(res[0], list) or isinstance(res[0], np.ndarray)) or len(res[0]) != 4:
                    logging.warning(f"RapidOCR 返回了格式不正确的坐标点: {res[0]}")
                    continue

                texts.append(str(res[1])) # 确保是字符串
                scores.append(f"{float(res[2]):.2f}") # 格式化为字符串
                try:
                    points = np.array(res[0], dtype=np.int32)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                except Exception as box_err:
                    logging.warning(f"处理 OCR 边界框时出错: {box_err} - 原始点: {res[0]}")
                    continue # 跳过这个结果

                boxes.append(OCRBox(
                    x=to_fixed(x_min),
                    y=to_fixed(y_min),
                    width=to_fixed(x_max - x_min),
                    height=to_fixed(y_max - y_min)
                ))

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return OCRResult(texts=[], scores=[], boxes=[])

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_models_loaded()
        try:
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]

            # --- 修正: 移除 normalize 调用 ---
            # normalized_embedding = self._normalize(embedding)
            # return [float(x) for x in normalized_embedding.flatten()]
            return [float(x) for x in embedding.flatten()] # 直接返回原始 embedding
            # --- 结束修正 ---
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

    def get_text_embedding(self, text: str) -> List[float]:
        self.ensure_models_loaded()
        try:
            inputs_tensor = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs_tensor.numpy()
            pad_index = clip._tokenizer.vocab['[PAD]']
            attention_mask = (input_ids != pad_index).astype(np.int64)

            infer_request = self.clip_text_model.create_infer_request()
            input_name_0 = self.clip_text_model.inputs[0].any_name
            input_name_1 = self.clip_text_model.inputs[1].any_name
            inputs_dict = {
                input_name_0: input_ids,
                input_name_1: attention_mask
            }
            results = infer_request.infer(inputs_dict)
            embedding = results[self.clip_text_model.outputs[0]]

            # --- 修正: 移除 normalize 调用 ---
            # normalized_embedding = self._normalize(embedding)
            # return [float(x) for x in normalized_embedding.flatten()]
            return [float(x) for x in embedding.flatten()] # 直接返回原始 embedding
            # --- 结束修正 ---
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS