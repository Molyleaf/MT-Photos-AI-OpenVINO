import os
import cv2
import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import CLIPProcessor, CLIPTokenizer
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ModelManager")

class ModelManager:
    def __init__(self, models_root_dir="/models"):
        self.models_root = Path(models_root_dir)
        self.device = os.getenv("INFERENCE_DEVICE", "AUTO")
        self.face_model_name = os.getenv("FACE_RECOGNITION_MODEL", "buffalo_l")

        log.info(f"Initializing ModelManager with models root: {self.models_root}")
        log.info(f"Inference device set to: {self.device}")
        log.info(f"Face recognition model set to: {self.face_model_name}")

        self._load_models()

    def _load_models(self):
        # 1. 加载 OpenVINO Core
        self.core = ov.Core()

        # 2. 加载 InsightFace 人脸识别模型
        face_model_path = self.models_root / "insightface" / self.face_model_name
        log.info(f"Loading InsightFace model from: {face_model_path}")
        self.face_analyzer = FaceAnalysis(
            name=self.face_model_name,
            root=self.models_root / "insightface",
            providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        log.info("InsightFace model loaded successfully.")

        # 3. 加载 RapidOCR 文本识别模型
        log.info("Loading RapidOCR model...")
        self.ocr_engine = RapidOCR()
        log.info("RapidOCR model loaded successfully.")

        # 4. 加载 Alt-CLIP 模型
        clip_model_dir = self.models_root / "alt-clip" / "openvino"
        log.info(f"Loading Alt-CLIP models from: {clip_model_dir}")

        # 加载 Tokenizer 和 Processor
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_dir)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)

        # 编译视觉模型
        vision_model_path = clip_model_dir / "clip_vision.xml"
        vision_model = self.core.read_model(model=vision_model_path)
        self.compiled_vision_model = self.core.compile_model(model=vision_model, device_name=self.device)
        log.info(f"Alt-CLIP vision model compiled for {self.device}.")

        # 编译文本模型
        text_model_path = clip_model_dir / "clip_text.xml"
        text_model = self.core.read_model(model=text_model_path)
        self.compiled_text_model = self.core.compile_model(model=text_model, device_name=self.device)
        log.info(f"Alt-CLIP text model compiled for {self.device}.")

    def get_face_representation(self, image: np.ndarray):
        faces = self.face_analyzer.get(image)
        results =
        for face in faces:
            facial_area = {
                "x": int(face.bbox),
                "y": int(face.bbox[1]),
                "w": int(face.bbox[2] - face.bbox),
                "h": int(face.bbox[3] - face.bbox[1]),
            }
            result = {
                "embedding": face.normed_embedding.tolist(),
                "facial_area": facial_area,
                "face_confidence": float(face.det_score),
            }
            results.append(result)

        return {
            "detector_backend": "insightface",
            "recognition_model": self.face_model_name,
            "result": results
        }

    def get_ocr_results(self, image: np.ndarray):
        ocr_result, _ = self.ocr_engine(image)
        if not ocr_result:
            return {"result": {"texts":, "scores":, "boxes":}}

            texts, scores, boxes =,,
            for item in ocr_result:
                box, text, score = item
                boxes.append([int(coord) for point in box for coord in point])
                texts.append(text)
                scores.append(score)

            return {"result": {"texts": texts, "scores": scores, "boxes": boxes}}

        def get_image_embedding(self, image: np.ndarray):
            inputs = self.clip_processor(images=image, return_tensors="np")
            pixel_values = inputs['pixel_values']
            result = self.compiled_vision_model([pixel_values])
            image_embeds = result[self.compiled_vision_model.output(0)]
            return {"results": image_embeds.flatten().tolist()}

        def get_text_embedding(self, text: str):
            inputs = self.clip_tokenizer([text], padding=True, return_tensors="np")
            input_ids = inputs['input_ids']
            result = self.compiled_text_model([input_ids])
            text_embeds = result[self.compiled_text_model.output(0)]
            return {"results": text_embeds.flatten().tolist()}