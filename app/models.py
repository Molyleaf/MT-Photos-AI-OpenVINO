import os
from typing import List, Any, Dict
import logging

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import AltCLIPProcessor

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
# 新增: 定义模型的实际输出维度
CLIP_EMBEDDING_DIMS = 1280

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIModels:
    def __init__(self):
        logging.info(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.alt_clip_path = os.path.join(MODEL_BASE_PATH, "alt-clip", "openvino")

        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_processor, self.clip_vision_model, self.clip_text_model = self._load_alt_clip()

        # 自动识别正确的 vision model 输出层
        self.vision_embedding_output = None
        self._find_vision_output()

        logging.info("所有模型已成功加载。")

    def _load_insightface(self) -> FaceAnalysis:
        logging.info(f"正在从以下根路径加载 InsightFace 模型: {self.insightface_root}")
        try:
            app = FaceAnalysis(
                name=MODEL_NAME,
                root=self.insightface_root,
                providers=['OpenVINOExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as e:
            logging.error(f"加载 InsightFace 模型时出错: {e}", exc_info=True)
            raise

    def _load_rapidocr(self) -> RapidOCR:
        logging.info("正在加载 RapidOCR 模型...")
        try:
            return RapidOCR()
        except Exception as e:
            logging.error(f"加载 RapidOCR 模型时出错: {e}", exc_info=True)
            raise

    def _load_alt_clip(self):
        logging.info(f"正在从以下路径加载 Alt-CLIP 模型: {self.alt_clip_path}")
        try:
            vision_model_path = os.path.join(self.alt_clip_path, "clip_vision.xml")
            text_model_path = os.path.join(self.alt_clip_path, "clip_text.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.alt_clip_path}' 路径下找到 Alt-CLIP 的 OpenVINO 模型文件。")

            config = {"PERFORMANCE_HINT": "LATENCY" if "GPU" in INFERENCE_DEVICE.upper() else "THROUGHPUT"}
            logging.info(f"使用性能提示 '{config['PERFORMANCE_HINT']}' 编译 Alt-CLIP 模型...")

            processor = AltCLIPProcessor.from_pretrained(self.alt_clip_path, use_fast=True)
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config)
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config)
            return processor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 Alt-CLIP 模型时出错: {e}", exc_info=True)
            raise

    def _find_vision_output(self):
        """
        自动检测 Vision 模型中代表最终 embedding 的输出层。
        此方法现在寻找维度为 1280 的输出层。
        """
        logging.info(f"正在识别正确的 Vision 模型输出层（期望维度: {CLIP_EMBEDDING_DIMS}）...")
        found = False
        for output in self.clip_vision_model.outputs:
            partial_shape = output.get_partial_shape()

            # 检查维度数量（秩）是否为 2
            if partial_shape.rank.is_static and partial_shape.rank.get_length() == 2:
                second_dim = partial_shape.get_dimension(1)

                # 检查第二个维度是否是固定的，并且其长度是否为我们期望的维度
                if second_dim.is_static and second_dim.get_length() == CLIP_EMBEDDING_DIMS:
                    self.vision_embedding_output = output
                    found = True
                    break

        if found:
            logging.info(f"成功识别 Vision embedding 输出层: '{self.vision_embedding_output.get_any_name()}'，形状: {self.vision_embedding_output.get_partial_shape()}")
        else:
            logging.error(f"错误: 无法在 Vision 模型中找到期望形状为 (*, {CLIP_EMBEDDING_DIMS}) 的输出层。")
            logging.error(f"可用的输出层: {[(o.get_any_name(), o.get_partial_shape()) for o in self.clip_vision_model.outputs]}")
            raise RuntimeError(f"Vision 模型输出配置不正确，找不到 {CLIP_EMBEDDING_DIMS} 维的嵌入输出。")

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # (代码无变化)
        faces = self.face_analyzer.get(image)
        results = []
        if not faces:
            return []
        for face in faces:
            if face.bbox is not None and face.embedding is not None:
                bbox = face.bbox.astype(int)
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                result = {"embedding": face.embedding.tolist(), "facial_area": {"x": x, "y": y, "w": w, "h": h}, "face_confidence": float(face.det_score)}
                results.append(result)
        return results

    def get_ocr_results(self, image: np.ndarray) -> Dict[str, Any]:
        # (代码无变化)
        result, _ = self.ocr_engine(image)
        if not result: return {"texts": [], "scores": [], "boxes": []}
        texts, scores, boxes = [], [], []
        for item in result:
            box_points = np.array(item[0])
            x, y = int(np.min(box_points[:, 0])), int(np.min(box_points[:, 1]))
            w, h = int(np.max(box_points[:, 0]) - x), int(np.max(box_points[:, 1]) - y)
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

            embedding = results[self.vision_embedding_output]

            logging.debug(f"[{filename}] 成功从 '{self.vision_embedding_output.get_any_name()}' 层获取 Embedding，形状: {embedding.shape}")

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
            input_ids, attention_mask = inputs['input_ids'].numpy(), inputs['attention_mask'].numpy()

            infer_request = self.clip_text_model.create_infer_request()
            results = infer_request.infer({
                self.clip_text_model.inputs[0].any_name: input_ids,
                self.clip_text_model.inputs[1].any_name: attention_mask
            })
            embedding = results[self.clip_text_model.outputs[0]]

            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            final_embedding = [float(x) for x in embedding.flatten()]

            # 验证文本 embedding 长度
            if len(final_embedding) != CLIP_EMBEDDING_DIMS:
                logging.warning(f"文本 '{text[:30]}...' 的 embedding 长度为 {len(final_embedding)}，而不是预期的 {CLIP_EMBEDDING_DIMS}！")

            return final_embedding
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

