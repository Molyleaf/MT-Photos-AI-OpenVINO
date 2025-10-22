import os
from typing import List, Any, Dict

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import AltCLIPProcessor

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "CPU")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

class AIModels:
    def __init__(self):
        print(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # 修正 insightface 模型根目录的逻辑
        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.alt_clip_path = os.path.join(MODEL_BASE_PATH, "alt-clip", "openvino")

        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_processor, self.clip_vision_model, self.clip_text_model = self._load_alt_clip()
        print("所有模型已成功加载。")

    def _load_insightface(self) -> FaceAnalysis:
        print(f"正在从以下根路径加载 InsightFace 模型: {self.insightface_root}")
        print("为 InsightFace 启用 OpenVINO Execution Provider...")
        try:
            # FaceAnalysis 的 'root' 参数应指向包含模型名称目录 (如 'buffalo_l') 的父目录
            app = FaceAnalysis(
                name=MODEL_NAME,
                root=self.insightface_root,
                providers=['OpenVINOExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as e:
            print(f"加载 InsightFace 模型时出错: {e}")
            raise

    def _load_rapidocr(self) -> RapidOCR:
        print("正在加载 RapidOCR 模型...")
        try:
            return RapidOCR()
        except Exception as e:
            print(f"加载 RapidOCR 模型时出错: {e}")
            raise

    def _load_alt_clip(self):
        print(f"正在从以下路径加载 Alt-CLIP 模型: {self.alt_clip_path}")
        try:
            vision_model_path = os.path.join(self.alt_clip_path, "clip_vision.xml")
            text_model_path = os.path.join(self.alt_clip_path, "clip_text.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.alt_clip_path}' 路径下找到 Alt-CLIP 的 OpenVINO 模型文件。")

            # 性能优化配置
            config = {"PERFORMANCE_HINT": "LATENCY" if INFERENCE_DEVICE == "GPU" else "THROUGHPUT"}
            print(f"使用性能提示 '{config['PERFORMANCE_HINT']}' 编译 Alt-CLIP 模型...")

            processor = AltCLIPProcessor.from_pretrained(self.alt_clip_path, use_fast=True)
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config)
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config)
            return processor, vision_compiled, text_compiled
        except Exception as e:
            print(f"加载 Alt-CLIP 模型时出错: {e}")
            raise

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.face_analyzer.get(image)
        results = []
        if not faces:
            return []

        for face in faces:
            if face.bbox is not None and face.embedding is not None:
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
        """
        FIX: 运行 OCR 并返回一个字典，其中包含三个独立的列表：texts, scores, boxes，以匹配客户端需求。
        """
        result, _ = self.ocr_engine(image)
        # 即使没有识别结果，也返回空的标准结构
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
            boxes.append({
                "x": x,
                "y": y,
                "width": w,  # 确保字段名为 width
                "height": h  # 确保字段名为 height
            })

        return {"texts": texts, "scores": scores, "boxes": boxes}

    def get_image_embedding(self, image: np.ndarray) -> List[float]:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].numpy()
        results = self.clip_vision_model.infer_new_request({self.clip_vision_model.inputs[0].any_name: pixel_values})
        embedding = list(results.values())[0]
        embedding /= np.linalg.norm(embedding)
        return embedding.flatten().tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].numpy()
        attention_mask = inputs['attention_mask'].numpy()
        results = self.clip_text_model.infer_new_request({
            self.clip_text_model.inputs[0].any_name: input_ids,
            self.clip_text_model.inputs[1].any_name: attention_mask
        })
        embedding = list(results.values())[0]
        embedding /= np.linalg.norm(embedding)
        return embedding.flatten().tolist()
