# app/common/models.py

import os
from typing import List, Any, Dict

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import AltCLIPProcessor

# --- 全局变量 ---
# 决定推理设备。如果未指定，则默认为 CPU。
# 可选值: "CPU", "GPU", "AUTO", "AUTO:CPU,GPU" 等。
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "CPU")
# 存放所有模型文件的基础路径
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

class AIModels:
    """一个用于加载和管理所有AI模型的类。"""
    def __init__(self):
        print(f"正在初始化AI模型，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # --- 模型路径定义 ---
        self.insightface_path = os.path.join(MODEL_BASE_PATH, "insightface", MODEL_NAME)
        self.alt_clip_path = os.path.join(MODEL_BASE_PATH, "alt-clip", "openvino")

        # --- 加载所有模型 ---
        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_processor, self.clip_vision_model, self.clip_text_model = self._load_alt_clip()

        print("所有模型已成功加载。")

    def _load_insightface(self) -> FaceAnalysis:
        """加载用于人脸识别的 InsightFace 模型 (使用 OpenVINO 加速)。""" # <- 更新注释
        print(f"正在从以下路径加载 InsightFace 模型: {self.insightface_path}")
        print("为 InsightFace 启用 OpenVINO Execution Provider...") # <- 添加日志
        try:
            # 指定使用 OpenVINOExecutionProvider 来利用 OpenVINO 进行推理
            app = FaceAnalysis(
                name = MODEL_NAME,
                root=os.path.dirname(self.insightface_path),
                providers=['OpenVINOExecutionProvider'] # <--- 核心修改
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as e:
            print(f"加载 InsightFace 模型时出错: {e}")
            raise

    def _load_rapidocr(self) -> RapidOCR:
        """加载 RapidOCR 模型。"""
        print("正在加载 RapidOCR 模型...")
        try:
            # OpenVINO 版本的 RapidOCR 会自动选择最佳设备
            return RapidOCR()
        except Exception as e:
            print(f"加载 RapidOCR 模型时出错: {e}")
            raise

    def _load_alt_clip(self):
        """加载 Alt-CLIP 的视觉和文本模型 (OpenVINO IR 格式)。"""
        print(f"正在从以下路径加载 Alt-CLIP 模型: {self.alt_clip_path}")
        try:
            vision_model_path = os.path.join(self.alt_clip_path, "clip_vision.xml")
            text_model_path = os.path.join(self.alt_clip_path, "clip_text.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError("未找到 Alt-CLIP 的 OpenVINO 模型文件。")

            # 加载 tokenizer 和 processor
            processor = AltCLIPProcessor.from_pretrained(self.alt_clip_path)

            # 编译模型到目标设备，并使用吞吐量性能提示
            vision_compiled = self.core.compile_model(
                vision_model_path, INFERENCE_DEVICE, {"PERFORMANCE_HINT": "THROUGHPUT"}
            )
            text_compiled = self.core.compile_model(
                text_model_path, INFERENCE_DEVICE, {"PERFORMANCE_HINT": "THROUGHPUT"}
            )

            return processor, vision_compiled, text_compiled
        except Exception as e:
            print(f"加载 Alt-CLIP 模型时出错: {e}")
            raise

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测人脸并提取特征向量。
        此函数输出格式与 API 要求一致。
        """
        faces = self.face_analyzer.get(image)
        results = []
        for face in faces:
            # Bbox 格式为 [x1, y1, x2, y2]
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
        对图像执行光学字符识别 (OCR)。
        此函数输出格式与 API 要求一致。
        """
        result, _ = self.ocr_engine(image)
        if not result:
            return {"texts": [], "scores": [], "boxes": []}

        texts = [item[1] for item in result]
        scores = [item[2] for item in result]
        boxes = [np.array(item[0]).astype(int).tolist() for item in result]

        return {"texts": texts, "scores": scores, "boxes": boxes}

    def get_image_embedding(self, image: np.ndarray) -> List[float]:
        """使用 Alt-CLIP 为图像生成特征向量。"""
        inputs = self.clip_processor(images=image, return_tensors="np")
        pixel_values = inputs['pixel_values']

        # 编译后的模型只有一个输入
        infer_request = self.clip_vision_model.create_infer_request()
        # 输入张量名称 'pixel_values' 是在模型导出时定义的
        results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
        # 输出张量名称为 'image_embeds'
        embedding = results[self.clip_vision_model.outputs[0]]

        # 对特征向量进行归一化
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.flatten().tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """使用 Alt-CLIP 为文本字符串生成特征向量。"""
        inputs = self.clip_processor(text=text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 编译后的模型有两个输入
        infer_request = self.clip_text_model.create_infer_request()
        # 输入张量名称为 'input_ids' 和 'attention_mask'
        results = infer_request.infer({
            self.clip_text_model.inputs[0].any_name: input_ids,
            self.clip_text_model.inputs[1].any_name: attention_mask
        })
        # 输出张量名称为 'text_embeds'
        embedding = results[self.clip_text_model.outputs[0]]

        # 对特征向量进行归一化
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.flatten().tolist()

# 在应用启动时，实例化一次模型管理类
try:
    models = AIModels()
except Exception as e:
    print(f"初始化 AI 模型失败: {e}")
    # 在这里可以选择退出程序或进行其他错误处理
    models = None

