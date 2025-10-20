# models.py
# 模型加载逻辑

import os
import cv2
import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from transformers import AltCLIPProcessor
from typing import List, Any, Dict

# --- Global Variables ---
# Determine a device for inference. Default to CPU if not specified.
# Can be "CPU", "GPU", "AUTO", "AUTO:CPU, GPU", etc.
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "CPU")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")

class AIModels:
    """A class to load and manage all AI models."""
    def __init__(self):
        print(f"Initializing AI models on device: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        # --- Model Paths ---
        self.insightface_path = os.path.join(MODEL_BASE_PATH, "insightface", "buffalo_l")
        self.alt_clip_path = os.path.join(MODEL_BASE_PATH, "alt-clip", "openvino")

        # --- Load Models ---
        self.face_analyzer = self._load_insightface()
        self.ocr_engine = self._load_rapidocr()
        self.clip_processor, self.clip_vision_model, self.clip_text_model = self._load_alt_clip()

        print("All models loaded successfully.")

    def _load_insightface(self) -> FaceAnalysis:
        """Loads the InsightFace model for face recognition."""
        print(f"Loading InsightFace model from: {self.insightface_path}")
        try:
            # You can specify providers like ['OpenVINOExecutionProvider']
            # if you have a specific build of insightface.
            # The default CPU provider is generally sufficient.
            app = FaceAnalysis(name="buffalo_l", root=os.path.dirname(self.insightface_path))
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as e:
            print(f"Error loading InsightFace model: {e}")
            raise

    def _load_rapidocr(self) -> RapidOCR:
        """Loads the RapidOCR model."""
        print("Loading RapidOCR model...")
        try:
            # RapidOCR for OpenVINO will automatically use the best device.
            return RapidOCR()
        except Exception as e:
            print(f"Error loading RapidOCR model: {e}")
            raise

    def _load_alt_clip(self):
        """Loads the Alt-CLIP vision and text models (OpenVINO IR format)."""
        print(f"Loading Alt-CLIP models from: {self.alt_clip_path}")
        try:
            vision_model_path = os.path.join(self.alt_clip_path, "clip_vision.xml")
            text_model_path = os.path.join(self.alt_clip_path, "clip_text.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError("Alt-CLIP OpenVINO model files not found.")

            # Load tokenizer and processor
            processor = AltCLIPProcessor.from_pretrained(self.alt_clip_path)

            # Compile models for the target device with performance hints for throughput
            vision_compiled = self.core.compile_model(
                vision_model_path, INFERENCE_DEVICE, {"PERFORMANCE_HINT": "THROUGHPUT"}
            )
            text_compiled = self.core.compile_model(
                text_model_path, INFERENCE_DEVICE, {"PERFORMANCE_HINT": "THROUGHPUT"}
            )

            return processor, vision_compiled, text_compiled
        except Exception as e:
            print(f"Error loading Alt-CLIP model: {e}")
            raise

    def get_face_representation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces and extracts embeddings.
        Matches the required API output format.
        """
        faces = self.face_analyzer.get(image)
        results = []
        for face in faces:
            # Bbox is in [x1, y1, x2, y2] format
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
        Performs OCR on the image.
        Matches the required API output format.
        """
        result, _ = self.ocr_engine(image)
        if not result:
            return {"texts": [], "scores": [], "boxes": []}

        texts = [item[1] for item in result]
        scores = [item[2] for item in result]
        boxes = [np.array(item[0]).astype(int).tolist() for item in result]

        return {"texts": texts, "scores": scores, "boxes": boxes}

    def get_image_embedding(self, image: np.ndarray) -> List[float]:
        """Generates a feature vector for an image using Alt-CLIP."""
        inputs = self.clip_processor(images=image, return_tensors="np")
        pixel_values = inputs['pixel_values']

        # The compiled model has one input
        infer_request = self.clip_vision_model.create_infer_request()
        # The input tensor name is 'pixel_values' as defined during export
        results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
        # The output tensor name is 'image_embeds'
        embedding = results[self.clip_vision_model.outputs[0]]

        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.flatten().tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """Generates a feature vector for a text string using Alt-CLIP."""
        inputs = self.clip_processor(text=text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # The compiled model has two inputs
        infer_request = self.clip_text_model.create_infer_request()
        # Input tensor names are 'input_ids' and 'attention_mask'
        results = infer_request.infer({
            self.clip_text_model.inputs[0].any_name: input_ids,
            self.clip_text_model.inputs[1].any_name: attention_mask
        })
        # The output tensor name is 'text_embeds'
        embedding = results[self.clip_text_model.outputs[0]]

        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.flatten().tolist()

# Instantiate models once for the entire application
try:
    models = AIModels()
except Exception as e:
    print(f"Failed to initialize AI models: {e}")
    # You might want to exit or handle this gracefully
    models = None
