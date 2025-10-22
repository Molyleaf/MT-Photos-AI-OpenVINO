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
# clip.utils 依赖于 chinese-clip 仓库中的 clip 目录
from clip.utils import image_transform, _MODEL_INFO
# --- 结束 ---


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

class AIModels:
    """一个封装所有 AI 模型加载和推理逻辑的类。"""
    def __init__(self):
        logging.info(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # 定义模型文件的根路径
        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        # --- MODIFIED: 路径更新 ---
        self.chinese_clip_path = os.path.join(MODEL_BASE_PATH, "chinese-clip", "openvino")
        # --- END MODIFIED ---

        # --- 模型加载 ---
        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        # --- MODIFIED: 重命名加载函数和返回变量 ---
        self.clip_image_preprocessor, self.clip_vision_model, self.clip_text_model = self._load_chinese_clip()
        # --- END MODIFIED ---

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
            face_app.prepare(ctx_id=0, det_size=(640, 640))
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

    # --- MODIFIED: 替换 _load_alt_clip ---
    def _load_chinese_clip(self):
        """加载 Chinese-CLIP ViT-L-14 (OpenVINO) 模型和预处理器。"""
        logging.info(f"正在从以下路径加载 Chinese-CLIP ({MODEL_ARCH}) 模型: {self.chinese_clip_path}")
        try:
            # 更改文件名以匹配新的 FP16 模型
            vision_model_path = os.path.join(self.chinese_clip_path, "openvino_image_fp16.xml")
            text_model_path = os.path.join(self.chinese_clip_path, "openvino_text_fp16.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(
                    f"未在 '{self.chinese_clip_path}' 路径下找到 OpenVINO 模型文件。"
                    "请确保 Docker build 期间 convert_models.py 脚本已成功运行。"
                )

            config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            logging.info(f"使用性能提示 '{config['PERFORMANCE_HINT']}' 编译 Chinese-CLIP 模型...")

            # --- 加载图像预处理器 ---
            # _MODEL_INFO 需要 'ViT-L-14' 架构名称
            image_preprocessor = image_transform(_MODEL_INFO[MODEL_ARCH]['input_resolution'])

            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config)
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config)

            # --- 最终验证步骤 ---
            vision_output_partial_shape = vision_compiled.outputs[0].get_partial_shape()
            if vision_output_partial_shape.rank.get_length() != 2 or vision_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    f"视觉模型输出维度不匹配！期望维度: (*, {CLIP_EMBEDDING_DIMS}), "
                    f"实际: {vision_output_partial_shape}。"
                )

            text_output_partial_shape = text_compiled.outputs[0].get_partial_shape()
            if text_output_partial_shape.rank.get_length() != 2 or text_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    f"文本模型输出维度不匹配！期望维度: (*, {CLIP_EMBEDDING_DIMS}), "
                    f"实际: {text_output_partial_shape}。"
                )

            # --- 验证文本模型输入数量 ---
            # 官方 Chinese-CLIP ONNX 文本模型只有 1 个输入
            if len(text_compiled.inputs) != 1:
                raise RuntimeError(
                    f"文本模型输入数量不匹配！期望 1 个输入 (input_ids)，"
                    f"实际: {len(text_compiled.inputs)} 个输入。"
                )

            logging.info(f"Chinese-CLIP 模型维度验证通过 (期望维度: {CLIP_EMBEDDING_DIMS})。")
            logging.info(f"Chinese-CLIP ({MODEL_ARCH}) 模型及 Preprocessor 加载成功。")
            return image_preprocessor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 Chinese-CLIP 模型时发生严重错误: {e}", exc_info=True)
            raise
    # --- END MODIFIED ---

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用 InsightFace 提取人脸特征。"""
        # (此函数不变)
        try:
            faces = self.face_analyzer.get(image)
            results = []
            for face in faces:
                results.append({
                    "embedding": [float(x) for x in face.normed_embedding],
                    "bbox": [int(x) for x in face.bbox.flatten()]
                })
            return results
        except Exception as e:
            logging.warning(f"处理人脸识别时出错: {e}", exc_info=True)
            return []

    def get_ocr_results(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用 RapidOCR 提取文本。"""
        # (此函数不变)
        try:
            ocr_result, _ = self.ocr_engine(image)
            if ocr_result is None:
                return []

            results = []
            for res in ocr_result:
                results.append({
                    "text": res[1],
                    "confidence": float(res[2]),
                    "bbox": [[int(p[0]), int(p[1])] for p in res[0]]
                })
            return results
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return []

    # --- MODIFIED: 更新图像 embedding 逻辑 ---
    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        """为图像生成 768 维的 CLIP 嵌入向量 (已归一化)。"""
        try:
            # 预处理 (image 应该是 PIL Image RGB 格式)
            # .unsqueeze(0) 增加 batch 维度
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            # 推理
            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]] # (1, 768)

            # L2 归一化
            normalized_embedding = self._normalize(embedding) # (1, 768)

            # 展平为列表
            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding

        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
    # --- END MODIFIED ---

    # --- MODIFIED: 更新文本 embedding 逻辑 ---
    def get_text_embedding(self, text: str) -> List[float]:
        """为文本生成 768 维的 CLIP 嵌入向量 (已归一化)。"""
        try:
            # 预处理
            # 使用 clip.tokenize，返回 torch.Tensor
            inputs = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs.numpy() # 转换为 numpy

            infer_request = self.clip_text_model.create_infer_request()

            # 传入 1 个输入
            results = infer_request.infer({
                self.clip_text_model.inputs[0].any_name: input_ids
            })

            embedding = results[self.clip_text_model.outputs[0]] # (1, 768)

            # L2 归一化
            normalized_embedding = self._normalize(embedding) # (1, 768)

            # 展平为列表
            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
    # --- END MODIFIED ---