# app/models.py
import os
from typing import List, Any, Dict
import logging

import numpy as np
import openvino as ov
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR
from PIL import Image

import clip
from clip.utils import image_transform, _MODEL_INFO

from schemas import (
    OCRBox,
    OCRResult,
    FacialArea,
    RepresentResult
)

# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

# 配置日志记录器 - 可以考虑将 INFO 改为 WARNING 以减少生产日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AIModels:
    """封装所有 AI 模型加载和推理逻辑的类。"""
    def __init__(self):
        logging.info(f"正在初始化AI模型实例 (尚未加载)，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.qa_clip_path = os.path.join(MODEL_BASE_PATH, "qa-clip", "openvino")

        self.face_analyzer = None
        self.ocr_engine = None
        self.clip_image_preprocessor = None
        self.clip_vision_model = None
        self.clip_text_model = None
        self.models_loaded = False

    def load_models(self):
        """加载所有 AI 模型到内存中。"""
        if self.models_loaded:
            # logging.info("模型已加载，跳过重复加载。") # 频繁调用时可能冗余
            return

        logging.info("--- 正在加载所有 AI 模型 ---")
        try:
            self.face_analyzer = self._load_insightface()
            self.ocr_engine = self._load_rapidocr()
            self.clip_image_preprocessor, self.clip_vision_model, self.clip_text_model = self._load_qa_clip()
            self.models_loaded = True
            logging.info("--- 所有模型已成功加载并编译 ---")
        except Exception as e:
            logging.critical(f"模型加载失败: {e}", exc_info=True)
            # 尝试释放部分加载的模型（如果存在）
            self.release_models()
            raise

    def release_models(self):
        """从内存中释放所有已编译的模型。"""
        # 仅当模型确实已加载时才记录释放日志
        if not self.models_loaded and not any([self.face_analyzer, self.ocr_engine, self.clip_vision_model, self.clip_text_model]):
            return # 如果从未加载或已释放，则无需操作

        logging.info("--- 正在释放所有 AI 模型 ---")
        try:
            if self.face_analyzer:
                del self.face_analyzer
                self.face_analyzer = None
            if self.ocr_engine:
                del self.ocr_engine
                self.ocr_engine = None
            if self.clip_vision_model:
                del self.clip_vision_model
                self.clip_vision_model = None
            if self.clip_text_model:
                del self.clip_text_model
                self.clip_text_model = None
            self.clip_image_preprocessor = None

            import gc
            gc.collect()

            logging.info("模型已成功从内存中释放。")
        except Exception as e:
            logging.warning(f"释放模型时出现错误: {e}", exc_info=True)
        finally:
            self.models_loaded = False


    def _load_insightface(self) -> FaceAnalysis:
        try:
            logging.info(f"正在加载 InsightFace (使用 OpenVINOExecutionProvider)...")
            providers = ['OpenVINOExecutionProvider']
            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)
            # Prepare a small dummy input to check provider compatibility during load
            face_app.prepare(ctx_id=0, det_size=(64, 64))
            logging.info(f"InsightFace 模型加载成功 (使用: {providers})。")
            return face_app
        except Exception as e:
            logging.error(f"加载 InsightFace 模型失败: {e}", exc_info=True)
            logging.critical("这通常是 onnxruntime-openvino 和 openvino 版本不兼容导致的。")
            logging.critical("请确保 requirements.txt 中的版本匹配 (例如: onnxruntime-openvino==1.23.0 需配合 openvino==2024.5.0)。")
            raise

    def _load_rapidocr(self) -> RapidOCR:
        try:
            logging.info("正在加载 RapidOCR (OpenVINO)...")
            ocr = RapidOCR()
            # 可以考虑在这里做一次虚拟推理以触发编译
            # dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            # ocr(dummy_img)
            logging.info("RapidOCR 模型加载成功。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 模型失败: {e}", exc_info=True)
            raise

    def _load_qa_clip(self):
        logging.info(f"正在加载 QA-CLIP ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            vision_model_path = os.path.join(self.qa_clip_path, "openvino_image_fp16.xml")
            text_model_path = os.path.join(self.qa_clip_path, "openvino_text_fp16.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 OpenVINO 模型文件。请先运行转换脚本。")

            config_vision = {"PERFORMANCE_HINT": "THROUGHPUT"}
            config_text = {"PERFORMANCE_HINT": "LATENCY"}

            logging.info(f"编译 Vision 模型 (提示: {config_vision['PERFORMANCE_HINT']})...")
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config_vision)

            logging.info(f"编译 Text 模型 (提示: {config_text['PERFORMANCE_HINT']})...")
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config_text)

            image_preprocessor = image_transform(_MODEL_INFO[MODEL_ARCH]['input_resolution'])

            # 基本验证
            if vision_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型维度不匹配！")
            if text_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型维度不匹配！")
            if len(text_compiled.inputs) != 2:
                logging.warning(f"文本模型输入数量不匹配！预期: 2, 得到: {len(text_compiled.inputs)}")

            logging.info(f"QA-CLIP ({MODEL_ARCH}) 模型及 Preprocessor 加载成功。")
            return image_preprocessor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP 模型时发生严重错误: {e}", exc_info=True)
            raise

    def ensure_models_loaded(self):
        """确保模型已加载，如果未加载，则加载它们。"""
        if not self.models_loaded:
            logging.warning("模型未加载。正在触发按需加载...")
            self.load_models()

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
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
            # 减少日志级别，因为无人脸是正常情况
            logging.debug(f"处理人脸识别时出错或未找到人脸: {e}", exc_info=False)
            return []

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_models_loaded()
        try:
            # logging.debug(f"输入 OCR 引擎的图像 shape: {image.shape}") # 移除 Debug 日志
            ocr_raw_output = self.ocr_engine(image)
            # logging.debug(f"RapidOCR 原始输出: {ocr_raw_output}") # 移除 Debug 日志

            if not isinstance(ocr_raw_output, tuple) or len(ocr_raw_output) < 1:
                logging.warning(f"RapidOCR 返回了意外的格式: {type(ocr_raw_output)}") # 保留警告
                return OCRResult(texts=[], scores=[], boxes=[])

            ocr_result, _ = ocr_raw_output

            if ocr_result is None:
                logging.warning("RapidOCR 返回 None 结果。") # 保留警告
                return OCRResult(texts=[], scores=[], boxes=[])

            if not isinstance(ocr_result, list):
                logging.warning(f"RapidOCR 的结果部分不是列表: {type(ocr_result)}") # 保留警告
                return OCRResult(texts=[], scores=[], boxes=[])

            if len(ocr_result) == 0:
                logging.info("RapidOCR 未检测到文本。") # 保留 Info 日志
                return OCRResult(texts=[], scores=[], boxes=[])

            texts, scores, boxes = [], [], []
            def to_fixed(num): return str(round(num, 2))

            for res in ocr_result:
                if not (isinstance(res, list) or isinstance(res, tuple)) or len(res) < 3:
                    logging.warning(f"RapidOCR 返回了格式不正确的识别结果项: {res}") # 保留警告
                    continue
                if not (isinstance(res[0], list) or isinstance(res[0], np.ndarray)) or len(res[0]) != 4:
                    logging.warning(f"RapidOCR 返回了格式不正确的坐标点: {res[0]}") # 保留警告
                    continue

                texts.append(str(res[1]))
                scores.append(f"{float(res[2]):.2f}")
                try:
                    points = np.array(res[0], dtype=np.int32)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                except Exception as box_err:
                    logging.warning(f"处理 OCR 边界框时出错: {box_err} - 原始点: {res[0]}") # 保留警告
                    continue

                boxes.append(OCRBox(
                    x=to_fixed(x_min),
                    y=to_fixed(y_min),
                    width=to_fixed(x_max - x_min),
                    height=to_fixed(y_max - y_min)
                ))

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True) # 保留警告
            return OCRResult(texts=[], scores=[], boxes=[])

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_models_loaded()
        try:
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()
            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]
            return [float(x) for x in embedding.flatten()]
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True) # 保留错误
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
            return [float(x) for x in embedding.flatten()]
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True) # 保留错误
            return [0.0] * CLIP_EMBEDDING_DIMS