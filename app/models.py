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
        # 规范化 (L2-norm)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        # 防止除以零
        norm[norm < 1e-6] = 1e-6
        return vector / norm

    def load_models(self):
        """加载所有 AI 模型到内存中。"""
        if self.models_loaded:
            logging.info("模型已加载，跳过重复加载。")
            return

        logging.info("--- 正在加载所有 AI 模型 ---")
        try:
            # 任务 4 & 6: 加载 InsightFace 和 RapidOCR
            self.face_analyzer = self._load_insightface()
            self.ocr_engine = self._load_rapidocr()
            # 任务 5: 加载 QA-CLIP
            self.clip_image_preprocessor, self.clip_vision_model, self.clip_text_model = self._load_qa_clip()
            self.models_loaded = True
            logging.info("--- 所有模型已成功加载并编译 ---")
        except Exception as e:
            logging.critical(f"模型加载失败: {e}", exc_info=True)
            # 确保在失败时实例仍处于未加载状态
            self.release_models()
            raise

    def release_models(self):
        """从内存中释放所有已编译的模型。 (任务 7)"""
        logging.info("--- 正在释放所有 AI 模型 ---")
        try:
            if self.face_analyzer:
                # InsightFace (onnxruntime) 没有显式的 release
                del self.face_analyzer
                self.face_analyzer = None

            if self.ocr_engine:
                # RapidOCR (OpenVINO) 没有显式的 release
                del self.ocr_engine
                self.ocr_engine = None

            # OpenVINO 编译的模型可以被 del
            if self.clip_vision_model:
                del self.clip_vision_model
                self.clip_vision_model = None

            if self.clip_text_model:
                del self.clip_text_model
                self.clip_text_model = None

            self.clip_image_preprocessor = None

            # 强制垃圾回收
            import gc
            gc.collect()

            logging.info("模型已成功从内存中释放。")
        except Exception as e:
            logging.warning(f"释放模型时出现错误: {e}", exc_info=True)
        finally:
            self.models_loaded = False


    def _load_insightface(self) -> FaceAnalysis:
        """
        加载 InsightFace (吞吐量优化)。(任务 6)
        使用 OpenVINOExecutionProvider。
        """
        try:
            logging.info(f"正在加载 InsightFace (OpenVINOExecutionProvider)...")

            # --- 修正: 解决 ONNXRuntime-OpenVINO 的依赖冲突 ---
            # 根据你的日志 (Source 95, 100)，OpenVINOProvider 加载失败。
            # 在你按照步骤 1 修复版本兼容性之前，
            # 我们可以暂时回退到 'CPUExecutionProvider' 以避免日志错误。
            #
            # 如果你已修复版本依赖 (例如使用 openvino==2024.5.0)，
            # 请将 providers 改回 ['OpenVINOExecutionProvider']

            providers = ['OpenVINOExecutionProvider']
            try:
                # 尝试加载 OpenVINO
                face_app_test = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)
                face_app_test.prepare(ctx_id=0, det_size=(1, 1)) # 用小尺寸快速测试
                del face_app_test
                logging.info("OpenVINOExecutionProvider 可用。")
            except Exception as e:
                # 捕获日志 (Source 95) 中出现的错误
                logging.warning(f"加载 OpenVINOExecutionProvider 失败: {e}")
                logging.warning("ONNXRuntime OpenVINO provider 存在库冲突, 降级到 CPUExecutionProvider。")
                providers = ['CPUExecutionProvider']
            # --- 结束修正 ---

            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logging.info(f"InsightFace 模型加载成功 (使用: {providers})。")
            return face_app
        except Exception as e:
            logging.error(f"加载 InsightFace 模型失败: {e}", exc_info=True)
            raise

    def _load_rapidocr(self) -> RapidOCR:
        """加载 RapidOCR (吞吐量优化)。(任务 6)"""
        try:
            logging.info("正在加载 RapidOCR (OpenVINO)...")
            # rapidocr-openvino 默认已为 OpenVINO 吞吐量优化
            ocr = RapidOCR()
            logging.info("RapidOCR 模型加载成功。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 模型失败: {e}", exc_info=True)
            raise

    def _load_qa_clip(self):
        """
        加载 QA-CLIP (OpenVINO) 模型。
        图像：吞吐量优化。(任务 5)
        文本：延迟优化。(任务 5)
        """
        logging.info(f"正在加载 QA-CLIP ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            vision_model_path = os.path.join(self.qa_clip_path, "openvino_image_fp16.xml")
            text_model_path = os.path.join(self.qa_clip_path, "openvino_text_fp16.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 OpenVINO 模型文件。请先运行转换脚本。")

            # --- 优化: 为图像和文本设置不同的性能提示 (任务 5 & 6) ---
            config_vision = {"PERFORMANCE_HINT": "THROUGHPUT"} # 图像：吞吐量
            config_text = {"PERFORMANCE_HINT": "LATENCY"}     # 文本：延迟

            logging.info(f"编译 Vision 模型 (提示: {config_vision['PERFORMANCE_HINT']})...")
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config_vision)

            logging.info(f"编译 Text 模型 (提示: {config_text['PERFORMANCE_HINT']})...")
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config_text)
            # --- 结束优化 ---

            # 使用 QA-CLIP 库 (app/clip/utils.py) 中的 image_transform
            image_preprocessor = image_transform(_MODEL_INFO[MODEL_ARCH]['input_resolution'])

            # 验证 (与 convert_models.py 中的验证逻辑一致)
            if vision_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型维度不匹配！")
            if text_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型维度不匹配！")

            # 新的转换脚本 (transformers) 会产生 2 个文本输入 (input_ids, attention_mask)
            if len(text_compiled.inputs) != 2:
                logging.warning(f"文本模型输入数量不匹配！预期: 2, 得到: {len(text_compiled.inputs)}")
                # (如果使用旧的 cn_clip 转换脚本，这里会是 1)

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

                # --- 修正: 返回在 schemas.py 中定义的 Pydantic 模型 ---
                results.append(RepresentResult(
                    embedding=[float(x) for x in face.normed_embedding],
                    facial_area=facial_area,
                    face_confidence=float(face.det_score)
                ))
                # --- 结束修正 ---

            return results
        except Exception as e:
            logging.warning(f"处理人脸识别时出错: {e}", exc_info=True)
            return []

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_models_loaded()
        try:
            ocr_result, _ = self.ocr_engine(image)
            if ocr_result is None:
                return OCRResult(texts=[], scores=[], boxes=[])

            texts, scores, boxes = [], [], []
            def to_fixed(num): return str(round(num, 2))

            for res in ocr_result:
                texts.append(res[1])
                scores.append(f"{float(res[2]):.2f}") # 格式化为字符串
                points = np.array(res[0], dtype=np.int32)
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)

                # --- 修正: 返回在 schemas.py 中定义的 Pydantic 模型 ---
                boxes.append(OCRBox(
                    x=to_fixed(x_min),
                    y=to_fixed(y_min),
                    width=to_fixed(x_max - x_min),
                    height=to_fixed(y_max - y_min)
                ))
                # --- 结束修正 ---

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return OCRResult(texts=[], scores=[], boxes=[])

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_models_loaded()
        try:
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            # OpenVINO 推理
            infer_request = self.clip_vision_model.create_infer_request()
            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]

            normalized_embedding = self._normalize(embedding)
            return [float(x) for x in normalized_embedding.flatten()]
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS

    def get_text_embedding(self, text: str) -> List[float]:
        self.ensure_models_loaded()
        try:
            # --- 修正 3: 使用 QA-CLIP (app/clip) 的 tokenizer 和 CONTEXT_LENGTH ---
            # clip.tokenize (来自 app/clip/utils.py) 使用的是 BertTokenizer
            inputs_tensor = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs_tensor.numpy()

            # 注意：QA-CLIP (transformers) 版本需要 attention_mask
            # clip.tokenize (来自 app/clip/utils.py) 似乎没有返回 attention_mask
            # 让我们检查一下 app/clip/utils.py 中的 tokenize...
            # 它只返回 token IDs，并用 [PAD] (id 0) 填充。
            # 我们需要手动创建 attention_mask
            pad_index = clip._tokenizer.vocab['[PAD]']
            attention_mask = (input_ids != pad_index).astype(np.int64)
            # --- 结束修正 3 ---

            infer_request = self.clip_text_model.create_infer_request()

            # 准备输入字典 (新转换脚本需要 2 个输入)
            # { 0: input_ids, 1: attention_mask }
            # 或者按名称 { 'input_ids': ..., 'attention_mask': ... }
            # 为了稳健性，我们按索引查找名称
            input_name_0 = self.clip_text_model.inputs[0].any_name
            input_name_1 = self.clip_text_model.inputs[1].any_name

            # 假设顺序是 input_ids, then attention_mask (与 wrapper 一致)
            inputs_dict = {
                input_name_0: input_ids,
                input_name_1: attention_mask
            }

            results = infer_request.infer(inputs_dict)
            embedding = results[self.clip_text_model.outputs[0]]

            normalized_embedding = self._normalize(embedding)
            return [float(x) for x in normalized_embedding.flatten()]
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS