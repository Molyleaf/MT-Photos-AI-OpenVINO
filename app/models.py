# app/models.py
import os
from typing import List, Any, Dict
import logging

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from PIL import Image

# --- 新增: 导入 Chinese-CLIP 相关工具 ---
import cn_clip.clip as clip
# --- 修正：修复 ModuleNotFoundError ---
from cn_clip.clip.utils import image_transform, _MODEL_INFO
# --- 结束修正 ---


# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

# --- MODIFIED ---
# ViT-L-14 对应的模型架构
MODEL_ARCH = "ViT-L-14"
# Chinese-CLIP ViT-L-14 的原生维度是 768
CLIP_EMBEDDING_DIMS = 768
# Chinese-CLIP 使用的上下文长度
CONTEXT_LENGTH = 52
# --- END MODIFIED ---

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 新增：导入 server_openvino.py 中的 Pydantic 模型 ---
# 这样做是为了确保返回的数据结构与 FastAPI 的期望完全一致
# 注意：在实际项目中，更好的做法是将 Pydantic 模型定义在一个单独的文件中，
# 然后让 models.py 和 server_openvino.py 都从该文件导入。
# 但为了简单起见，我们暂时在这里重新定义（或确保它们一致）。
from pydantic import BaseModel

class OCRBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class OCRResult(BaseModel):
    texts: List[str]
    scores: List[float]
    boxes: List[OCRBox]

class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int

class RepresentResult(BaseModel):
    embedding: List[float]
    facial_area: FacialArea
    face_confidence: float
# --- 结束新增 ---


class AIModels:
    """一个封装所有 AI 模型加载和推理逻辑的类。"""
    def __init__(self):
        logging.info(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # 定义模型文件的根路径
        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.chinese_clip_path = os.path.join(MODEL_BASE_PATH, "chinese-clip", "openvino")

        # --- 模型加载 ---
        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_image_preprocessor, self.clip_vision_model, self.clip_text_model = self._load_chinese_clip()

        logging.info("所有模型已成功加载并准备就绪。")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        对输出向量进行 L2 归一化 (按批次)。
        """
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1e-6
        return vector / norm

    def _load_insightface(self) -> FaceAnalysis:
        """加载 InsightFace 人脸识别模型。"""
        try:
            logging.info(f"正在从以下路径加载 InsightFace 模型: {self.insightface_root}")
            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root)
            # --- 修正：显式指定 providers ---
            # 避免 UserWarning: Specified provider 'CUDAExecutionProvider' ...
            # OpenVINOExecutionProvider 可能需要额外配置，先用 CPU
            face_app.prepare(ctx_id=0, det_size=(640, 640), providers=['CPUExecutionProvider'])
            # --- 结束修正 ---
            logging.info("InsightFace 模型加载成功。")
            return face_app
        except Exception as e:
            logging.error(f"加载 InsightFace 模型失败: {e}", exc_info=True)
            raise

    def _load_rapidocr(self) -> RapidOCR:
        """加载 RapidOCR 模型。"""
        try:
            logging.info("正在加载 RapidOCR (OpenVINO) 模型...")
            ocr = RapidOCR()
            logging.info("RapidOCR 模型加载成功。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 模型失败: {e}", exc_info=True)
            raise

    def _load_chinese_clip(self):
        """加载 Chinese-CLIP ViT-L-14 (OpenVINO) 模型和预处理器。"""
        logging.info(f"正在从以下路径加载 Chinese-CLIP ({MODEL_ARCH}) 模型: {self.chinese_clip_path}")
        try:
            vision_model_path = os.path.join(self.chinese_clip_path, "openvino_image_fp16.xml")
            text_model_path = os.path.join(self.chinese_clip_path, "openvino_text_fp16.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(
                    f"未在 '{self.chinese_clip_path}' 路径下找到 OpenVINO 模型文件。"
                    "请确保模型文件已正确放置。"
                )

            config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            logging.info(f"使用性能提示 '{config['PERFORMANCE_HINT']}' 编译 Chinese-CLIP 模型...")

            image_preprocessor = image_transform(_MODEL_INFO[MODEL_ARCH]['input_resolution'])

            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config)
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config)

            # --- 验证步骤 ---
            # (验证代码不变)
            vision_output_partial_shape = vision_compiled.outputs[0].get_partial_shape()
            if vision_output_partial_shape.rank.get_length() != 2 or vision_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型输出维度不匹配！")
            text_output_partial_shape = text_compiled.outputs[0].get_partial_shape()
            if text_output_partial_shape.rank.get_length() != 2 or text_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型输出维度不匹配！")
            if len(text_compiled.inputs) != 1:
                raise RuntimeError(f"文本模型输入数量不匹配！")

            logging.info(f"Chinese-CLIP 模型维度验证通过 (期望维度: {CLIP_EMBEDDING_DIMS})。")
            logging.info(f"Chinese-CLIP ({MODEL_ARCH}) 模型及 Preprocessor 加载成功。")
            return image_preprocessor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 Chinese-CLIP 模型时发生严重错误: {e}", exc_info=True)
            raise

    # --- 修正：get_face_representation 返回值结构 ---
    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        """使用 InsightFace 提取人脸特征，返回符合 RepresentResult 结构的列表。"""
        try:
            faces = self.face_analyzer.get(image)
            results = []
            for face in faces:
                # face.bbox 是 [x1, y1, x2, y2]
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                # 转换为 FacialArea (x, y, w, h)
                facial_area = FacialArea(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

                results.append(RepresentResult(
                    embedding=[float(x) for x in face.normed_embedding],
                    facial_area=facial_area,
                    face_confidence=float(face.det_score) # 添加 face_confidence
                ))
            return results
        except Exception as e:
            logging.warning(f"处理人脸识别时出错: {e}", exc_info=True)
            return []
    # --- 结束修正 ---

    # --- 修正：get_ocr_results 返回值结构 ---
    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        """使用 RapidOCR 提取文本，返回符合 OCRResult 结构的对象。"""
        try:
            ocr_result, _ = self.ocr_engine(image)
            if ocr_result is None:
                # 返回空的 OCRResult 对象
                return OCRResult(texts=[], scores=[], boxes=[])

            texts = []
            scores = []
            boxes = []
            for res in ocr_result:
                texts.append(res[1])
                scores.append(float(res[2]))
                # res[0] 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                points = np.array(res[0], dtype=np.int32)
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                width = x_max - x_min
                height = y_max - y_min
                boxes.append(OCRBox(x=int(x_min), y=int(y_min), width=int(width), height=int(height)))

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            # 返回空的 OCRResult 对象
            return OCRResult(texts=[], scores=[], boxes=[])
    # --- 结束修正 ---

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        """为图像生成 768 维的 CLIP 嵌入向量 (已归一化)。"""
        try:
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]

            normalized_embedding = self._normalize(embedding)
            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding

        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

    def get_text_embedding(self, text: str) -> List[float]:
        """为文本生成 768 维的 CLIP 嵌入向量 (已归一化)。"""
        try:
            inputs = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs.numpy()

            infer_request = self.clip_text_model.create_infer_request()
            results = infer_request.infer({self.clip_text_model.inputs[0].any_name: input_ids})
            embedding = results[self.clip_text_model.outputs[0]]

            normalized_embedding = self._normalize(embedding)
            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS