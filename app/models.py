import os
from typing import List, Any, Dict
import logging

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import AltCLIPProcessor

# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

# 关键常量: 必须与 convert_models_fixed.py 的输出保持一致
CLIP_EMBEDDING_DIMS = 768

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIModels:
    """一个封装所有 AI 模型加载和推理逻辑的类。"""
    def __init__(self):
        logging.info(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # 定义模型文件的根路径
        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.alt_clip_path = os.path.join(MODEL_BASE_PATH, "alt-clip", "openvino")

        # --- 模型加载 ---
        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_processor, self.clip_vision_model, self.clip_text_model = self._load_alt_clip()

        logging.info("所有模型已成功加载并准备就绪。")

    def _load_insightface(self) -> FaceAnalysis:
        logging.info(f"正在从以下根路径加载 InsightFace 模型: {self.insightface_root}")
        try:
            app = FaceAnalysis(
                name=MODEL_NAME,
                root=self.insightface_root,
                providers=['OpenVINOExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            logging.info("InsightFace 模型加载成功。")
            return app
        except Exception as e:
            logging.error(f"加载 InsightFace 模型时发生严重错误: {e}", exc_info=True)
            raise

    def _load_rapidocr(self) -> RapidOCR:
        logging.info("正在加载 RapidOCR 模型...")
        try:
            engine = RapidOCR()
            logging.info("RapidOCR 模型加载成功。")
            return engine
        except Exception as e:
            logging.error(f"加载 RapidOCR 模型时发生严重错误: {e}", exc_info=True)
            raise

    def _load_alt_clip(self):
        logging.info(f"正在从以下路径加载 Alt-CLIP 模型: {self.alt_clip_path}")
        try:
            vision_model_path = os.path.join(self.alt_clip_path, "clip_vision.xml")
            text_model_path = os.path.join(self.alt_clip_path, "clip_text.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.alt_clip_path}' 路径下找到 Alt-CLIP 的 OpenVINO 模型文件。请确保已运行正确的模型转换脚本。")

            config = {"PERFORMANCE_HINT": "THROUGHPUT"}
            logging.info(f"使用性能提示 '{config['PERFORMANCE_HINT']}' 编译 Alt-CLIP 模型...")

            processor = AltCLIPProcessor.from_pretrained(self.alt_clip_path, use_fast=True)

            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config)
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config)

            # --- 最终验证步骤 (已修复) ---
            # 使用 get_partial_shape() 来安全地处理动态维度
            vision_output_partial_shape = vision_compiled.outputs[0].get_partial_shape()
            if vision_output_partial_shape.rank.get_length() != 2 or vision_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    f"视觉模型输出维度不匹配！期望维度: (*, {CLIP_EMBEDDING_DIMS}), "
                    f"实际加载的模型输出维度: {vision_output_partial_shape}。请确保您使用了正确的模型文件！"
                )

            text_output_partial_shape = text_compiled.outputs[0].get_partial_shape()
            if text_output_partial_shape.rank.get_length() != 2 or text_output_partial_shape[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    f"文本模型输出维度不匹配！期望维度: (*, {CLIP_EMBEDDING_DIMS}), "
                    f"实际加载的模型输出维度: {text_output_partial_shape}。请确保您使用了正确的模型文件！"
                )

            logging.info(f"Alt-CLIP 模型维度验证通过 (期望维度: {CLIP_EMBEDDING_DIMS})。")
            logging.info("Alt-CLIP 模型及 Processor 加载成功。")
            return processor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 Alt-CLIP 模型时发生严重错误: {e}", exc_info=True)
            raise

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.face_analyzer.get(image)
        if not faces:
            return []

        results = []
        for face in faces:
            if face.bbox is not None and face.embedding is not None and face.det_score is not None:
                bbox = face.bbox.astype(int)
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                result = {
                    "embedding": face.embedding.tolist(),
                    "facial_area": {"x": x, "y": y, "w": w, "h": h},
                    "face_confidence": float(face.det_score)
                }
                results.append(result)
        return results

    def get_ocr_results(self, image: np.ndarray) -> Dict[str, Any]:
        result, _ = self.ocr_engine(image)
        if not result:
            return {"texts": [], "scores": [], "boxes": []}

        texts, scores, boxes = [], [], []
        for item in result:
            box_points = np.array(item[0])
            x = int(np.min(box_points[:, 0]))
            y = int(np.min(box_points[:, 1]))
            w = int(np.max(box_points[:, 0]) - x)
            h = int(np.max(box_points[:, 1]) - y)
            texts.append(item[1])
            scores.append(float(item[2]))
            boxes.append({"x": x, "y": y, "width": w, "height": h})
        return {"texts": texts, "scores": scores, "boxes": boxes}

    def get_image_embedding(self, image: np.ndarray, filename: str = "unknown") -> List[float]:
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].numpy()

            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})

            embedding = results[self.clip_vision_model.outputs[0]]

            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                normalized_embedding = embedding / norm
            else:
                normalized_embedding = embedding

            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding

        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

    def get_text_embedding(self, text: str) -> List[float]:
        try:
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].numpy()
            attention_mask = inputs['attention_mask'].numpy()

            infer_request = self.clip_text_model.create_infer_request()
            results = infer_request.infer({
                self.clip_text_model.inputs[0].any_name: input_ids,
                self.clip_text_model.inputs[1].any_name: attention_mask
            })

            embedding = results[self.clip_text_model.outputs[0]]

            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                normalized_embedding = embedding / norm
            else:
                normalized_embedding = embedding

            final_embedding = [float(x) for x in normalized_embedding.flatten()]

            return final_embedding
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

