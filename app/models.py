# app/models.py
import logging
import os
import queue
import threading
from typing import List

# --- 【最终修复 v2】: 解决 OpenVINO DLL (Error 127) ---
# 必须在导入 insightface 和 rapidocr 之前执行
import sys
if sys.platform == "win32":
    try:
        import openvino

        ov_path = os.path.dirname(openvino.__file__)
        ov_libs_path = os.path.join(ov_path, "libs")
        ov_bin_path_old = os.path.join(ov_path, "runtime", "bin")

        found_path = None
        if os.path.isdir(ov_libs_path):
            # 新版 OpenVINO (2023+)
            found_path = ov_libs_path
        elif os.path.isdir(ov_bin_path_old):
            # 旧版 OpenVINO (2022-)
            found_path = ov_bin_path_old

        if found_path:
            # 1. (新) 使用 os.environ["PATH"]，这是最可靠的方法
            # 我们将其添加到 PATH 的最前面
            os.environ["PATH"] = found_path + os.pathsep + os.environ.get("PATH", "")

            # 2. (保留) 使用 add_dll_directory，作为双重保险
            # os.add_dll_directory 在 Python 3.8+ 上可用
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(found_path)

            # 3. (新) 将日志级别改为 WARNING，确保我们能在日志中看到此条消息
            logging.warning(f"已将 OpenVINO 运行时 (for ONNX) 添加到 PATH: {found_path}")
        else:
            # 如果两个路径都没找到，这是一个严重错误
            logging.error(f"严重错误: 未在 {ov_path} 中找到 'libs' 或 'runtime/bin' 目录。")

    except ImportError:
        logging.warning("未安装 'openvino' 包。onnxruntime-openvino 可能无法工作。")
    except Exception as e:
        logging.error(f"自动设置 OpenVINO DLL 路径时出错: {e}", exc_info=True)
# --- 修复结束 ---

import numpy as np
import openvino as ov
from PIL import Image
from insightface.app import FaceAnalysis
from rapidocr_openvino import RapidOCR

import clip
from clip.utils import image_transform, MODEL_INFO
from schemas import (
    OCRBox,
    OCRResult,
    FacialArea,
    RepresentResult
)

# --- 环境变量与常量定义 ---
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
logging.info(f"模型根目录 (MODEL_BASE_PATH): {MODEL_BASE_PATH}")

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "4"))
MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

# 注意：logging.basicConfig 应该在 server.py 中配置，以避免在库模式下产生副作用
# 这里保留它是为了确保日志输出，但在一个大型应用中，应由主入口点配置
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class AIModels:
    """封装所有 AI 模型加载和推理逻辑的类（并发安全版）。"""
    def __init__(self):
        logging.warning(f"正在初始化AI模型实例 (尚未加载)，使用设备: {INFERENCE_DEVICE}")
        self.core = ov.Core()

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.qa_clip_path = os.path.join(MODEL_BASE_PATH, "qa-clip", "openvino")

        self.clip_vision_model = None
        self.clip_text_model = None
        self.clip_image_preprocessor = None

        self.face_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.ocr_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.clip_vision_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)
        self.clip_text_pool: queue.Queue = queue.Queue(maxsize=INFERENCE_WORKERS)

        self.models_loaded = False
        self._load_lock = threading.Lock()

    def load_models(self):
        """加载所有 AI 模型并填充实例池。"""
        if self.models_loaded:
            return

        with self._load_lock:
            if self.models_loaded:
                return

            logging.warning("--- 正在加载所有 AI 模型 ---")
            try:
                if self.clip_vision_model is None:
                    logging.warning("编译基础 CLIP 模型...")
                    self.clip_image_preprocessor, self.clip_vision_model, self.clip_text_model = self._load_qa_clip()
                else:
                    logging.warning("基础 CLIP 模型已编译，跳过。")

                if self.face_pool.empty():
                    logging.warning(f"--- 正在填充 {INFERENCE_WORKERS} 个工作实例到池中... ---")
                    for i in range(INFERENCE_WORKERS):
                        logging.warning(f"加载 FaceAnalysis 实例 {i+1}/{INFERENCE_WORKERS}...")
                        self.face_pool.put(self._load_insightface())

                        logging.warning(f"加载 RapidOCR 实例 {i+1}/{INFERENCE_WORKERS}...")
                        self.ocr_pool.put(self._load_rapidocr())

                        self.clip_vision_pool.put(self.clip_vision_model.create_infer_request())
                        self.clip_text_pool.put(self.clip_text_model.create_infer_request())

                    logging.warning(f"--- {INFERENCE_WORKERS} 个工作实例已准备就绪 ---")
                else:
                    logging.warning("--- 工作实例池已填充，跳过填充。 ---")

                self.models_loaded = True
                logging.warning("--- 所有模型已成功加载并编译 ---")
            except Exception as e:
                logging.critical(f"模型加载失败: {e}", exc_info=True)
                self.release_models()
                raise

    def release_models(self):
        """从内存中释放所有已编译的模型和池化实例。"""
        if not self.models_loaded and self.face_pool.empty():
            return

        logging.warning("--- 正在释放所有 AI 模型和实例池 ---")
        try:
            def _empty_queue(q: queue.Queue):
                if q is None: return
                while not q.empty():
                    try:
                        item = q.get_nowait()
                        del item
                    except queue.Empty:
                        break
                    except Exception as e:
                        logging.warning(f"清空队列时出错: {e}")

            _empty_queue(self.face_pool)
            _empty_queue(self.ocr_pool)
            _empty_queue(self.clip_vision_pool)
            _empty_queue(self.clip_text_pool)
            logging.warning("实例池已清空。")

            if self.clip_vision_model:
                del self.clip_vision_model
                self.clip_vision_model = None
            if self.clip_text_model:
                del self.clip_text_model
                self.clip_text_model = None

            self.clip_image_preprocessor = None

            import gc
            gc.collect()

            logging.warning("模型已成功从内存中释放。")
        except Exception as e:
            logging.warning(f"释放模型时出现错误: {e}", exc_info=True)
        finally:
            self.models_loaded = False

    def _load_insightface(self) -> FaceAnalysis:
        # --- 【修复】: 实现多阶段回退 (AUTO OV -> CPU OV -> Generic CPU) ---

        # 尝试 1: 使用 INFERENCE_DEVICE (例如 "AUTO")
        primary_device_type = INFERENCE_DEVICE.split('.')[0]
        try:
            provider_options = {'device_type': primary_device_type}
            providers = [('OpenVINOExecutionProvider', provider_options)]

            logging.debug(f"加载 InsightFace (尝试 {providers})...")
            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers)
            face_app.prepare(ctx_id=0, det_size=(64, 64))
            logging.warning(f"InsightFace 实例在 {primary_device_type} (OpenVINO EP) 上加载成功。")
            return face_app
        except Exception as e:
            # 记录 Error 127 或其他错误
            logging.error(f"加载 InsightFace (OpenVINO EP, device={primary_device_type}) 失败。错误: {e}", exc_info=True)

            # 尝试 2: 回退到 OpenVINO CPU
            logging.warning(f"InsightFace {primary_device_type} (OpenVINO EP) 加载失败，将回退到 OpenVINO CPU Execution Provider...")
            try:
                provider_options_cpu = {'device_type': 'CPU'}
                providers_cpu = [('OpenVINOExecutionProvider', provider_options_cpu)]

                logging.debug(f"加载 InsightFace (尝试 {providers_cpu})...")
                face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers_cpu)
                face_app.prepare(ctx_id=0, det_size=(64, 64))
                logging.warning("InsightFace 实例已在 CPU (OpenVINO EP 回退模式) 上成功加载。")
                return face_app
            except Exception as fallback_e:
                logging.error(f"InsightFace 在 CPU (OpenVINO EP 回退) 模式下也加载失败: {fallback_e}", exc_info=True)

                # 尝试 3: 最终回退到通用的 CPUExecutionProvider (无 OpenVINO 加速)
                logging.warning("OpenVINO EP 彻底失败，将回退到通用的 CPUExecutionProvider (无 OpenVINO 加速)...")
                try:
                    providers_generic_cpu = ['CPUExecutionProvider']
                    face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers_generic_cpu)
                    face_app.prepare(ctx_id=0, det_size=(64, 64))
                    logging.warning("InsightFace 实例已在通用 CPU (回退模式) 上成功加载。")
                    return face_app
                except Exception as final_fallback_e:
                    logging.critical(f"InsightFace 在所有模式下均加载失败: {final_fallback_e}", exc_info=True)
                    raise final_fallback_e
        # --- 修复结束 ---

    @staticmethod
    def _load_rapidocr() -> RapidOCR:
        # (此处的逻辑已正确：AUTO -> CPU (OpenVINO)
        try:
            logging.debug(f"加载 RapidOCR (OpenVINO) (尝试使用 {INFERENCE_DEVICE})...")
            # RapidOCR 会自动处理 'AUTO' 设备
            ocr = RapidOCR(device_name=INFERENCE_DEVICE)
            logging.warning(f"RapidOCR 实例在 {INFERENCE_DEVICE} 上加载成功。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 实例失败 (尝试使用 {INFERENCE_DEVICE}): {e}", exc_info=True)
            logging.warning("RapidOCR (OpenVINO) 加载失败，将回退到 CPU...")
            try:
                ocr = RapidOCR(device_name="CPU")
                logging.warning("RapidOCR 实例已在 CPU 上成功加载 (回退模式)。")
                return ocr
            except Exception as fallback_e:
                logging.critical(f"RapidOCR 在 CPU (回退) 模式下也加载失败: {fallback_e}", exc_info=True)
                raise fallback_e

    def _load_qa_clip(self):
        logging.warning(f"正在加载 QA-CLIP ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            vision_model_path = os.path.join(self.qa_clip_path, "openvino_image_fp16.xml")
            text_model_path = os.path.join(self.qa_clip_path, "openvino_text_fp16.xml")

            if not os.path.exists(vision_model_path) or not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 OpenVINO 模型文件。请先运行转换脚本。")

            config_vision = {"PERFORMANCE_HINT": "THROUGHPUT"}
            config_text = {"PERFORMANCE_HINT": "LATENCY"}

            logging.warning(f"编译 Vision 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_vision['PERFORMANCE_HINT']})...")
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config_vision)

            logging.warning(f"编译 Text 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_text['PERFORMANCE_HINT']})...")
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config_text)

            image_preprocessor = image_transform(MODEL_INFO[MODEL_ARCH]['input_resolution'])

            if vision_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型维度不匹配！")
            if text_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型维度不匹配！")

            logging.warning(f"QA-CLIP ({MODEL_ARCH}) 基础模型编译成功。")
            return image_preprocessor, vision_compiled, text_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP 模型时发生严重错误: {e}", exc_info=True)
            raise

    def ensure_models_loaded(self):
        """确保模型已加载，如果未加载，则加载它们。（线程安全）"""
        if not self.models_loaded:
            with self._load_lock:
                if not self.models_loaded:
                    logging.warning("模型未加载。正在触发按需加载...")
                    self.load_models()

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        self.ensure_models_loaded()
        face_analyzer = None
        try:
            face_analyzer = self.face_pool.get(timeout=10)
            faces = face_analyzer.get(image)
            results = []
            for face in faces:
                bbox = np.array(face.bbox).astype(int)
                x1, y1, x2, y2 = bbox
                facial_area = FacialArea(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
                results.append(RepresentResult(
                    embedding=[float(x) for x in face.normed_embedding],
                    facial_area=facial_area,
                    face_confidence=float(face.det_score)
                ))
            return results
        except Exception as e:
            # 调试级别，因为“未找到人脸”是正常情况
            logging.debug(f"处理人脸识别时出错或未找到人脸: {e}", exc_info=False)
            return []
        finally:
            if face_analyzer:
                self.face_pool.put(face_analyzer)

    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_models_loaded()
        ocr_engine = None
        try:
            ocr_engine = self.ocr_pool.get(timeout=10)
            ocr_raw_output = ocr_engine(image)

            if not isinstance(ocr_raw_output, tuple) or len(ocr_raw_output) < 1:
                logging.warning(f"RapidOCR 返回了意外的格式: {type(ocr_raw_output)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            ocr_result, _ = ocr_raw_output

            if ocr_result is None:
                # 这是正常情况（没有文本）
                logging.debug("RapidOCR 返回 None 结果 (无文本)。")
                return OCRResult(texts=[], scores=[], boxes=[])

            if not isinstance(ocr_result, list):
                logging.warning(f"RapidOCR 的结果部分不是列表: {type(ocr_result)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            if len(ocr_result) == 0:
                # 正常情况（无文本）
                return OCRResult(texts=[], scores=[], boxes=[])

            texts, scores, boxes = [], [], []
            def to_fixed(num): return str(round(num, 2))

            for res in ocr_result:
                if not (isinstance(res, list) or isinstance(res, tuple)) or len(res) < 3:
                    logging.warning(f"RapidOCR 返回了格式不正确的识别结果项: {res}")
                    continue
                if not (isinstance(res[0], list) or isinstance(res[0], np.ndarray)) or len(res[0]) != 4:
                    logging.warning(f"RapidOCR 返回了格式不正确的坐标点: {res[0]}")
                    continue

                texts.append(str(res[1]))
                scores.append(f"{float(res[2]):.2f}")
                try:
                    points = np.array(res[0], dtype=np.int32)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                except Exception as box_err:
                    logging.warning(f"处理 OCR 边界框时出错: {box_err} - 原始点: {res[0]}")
                    continue

                boxes.append(OCRBox(
                    x=to_fixed(x_min),
                    y=to_fixed(y_min),
                    width=to_fixed(x_max - x_min),
                    height=to_fixed(y_max - y_min)
                ))

            return OCRResult(texts=texts, scores=scores, boxes=boxes)
        except Exception as e:
            logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return OCRResult(texts=[], scores=[], boxes=[])
        finally:
            if ocr_engine:
                self.ocr_pool.put(ocr_engine)

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_models_loaded()
        infer_request = None
        try:
            inputs = self.clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            infer_request = self.clip_vision_pool.get(timeout=10)

            results = infer_request.infer({self.clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[self.clip_vision_model.outputs[0]]
            return [float(x) for x in embedding.flatten()]
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if infer_request:
                self.clip_vision_pool.put(infer_request)

    def get_text_embedding(self, text: str) -> List[float]:
        self.ensure_models_loaded()
        infer_request = None
        try:
            inputs_tensor = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs_tensor.numpy()
            pad_index = clip._tokenizer.vocab['[PAD]']
            attention_mask = np.array(input_ids != pad_index).astype(np.int64)

            infer_request = self.clip_text_pool.get(timeout=10)

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
            logging.error(f"在 get_text_embedding 中处理 '{text[:50]}...' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if infer_request:
                self.clip_text_pool.put(infer_request)