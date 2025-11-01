# app/models.py
import logging
import os
import queue
import threading
from typing import List, Tuple, Optional, Callable

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
import gc

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "AUTO")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models")
MODEL_BASE_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
logging.info(f"模型根目录 (MODEL_BASE_PATH): {MODEL_BASE_PATH}")

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
MODEL_ARCH = "ViT-L-14"
CLIP_EMBEDDING_DIMS = 768
CONTEXT_LENGTH = 77

logging.getLogger(__name__).setLevel(logging.WARNING)


class AIModels:
    """封装所有 AI 模型加载和推理逻辑的类（并发安全版）。"""
    def __init__(self):
        logging.warning(f"正在初始化AI模型实例 (尚未加载)，使用设备: {INFERENCE_DEVICE}。所有模型将使用单实例队列。")
        self.core = ov.Core()

        self.insightface_root = os.path.join(MODEL_BASE_PATH, "insightface")
        self.qa_clip_path = os.path.join(MODEL_BASE_PATH, "qa-clip", "openvino")

        # 基础编译模型 (仍然保留对基础模型的引用，以便释放)
        self.clip_vision_model: Optional[ov.CompiledModel] = None
        self.clip_text_model: Optional[ov.CompiledModel] = None
        self.clip_image_preprocessor = None # 仅用于基础引用

        self.ocr_pool: queue.Queue = queue.Queue(maxsize=1)
        self.face_pool: queue.Queue = queue.Queue(maxsize=1)
        self.clip_vision_pool: queue.Queue = queue.Queue(maxsize=1)
        self.clip_text_pool: queue.Queue = queue.Queue(maxsize=1)

        self._load_lock = threading.RLock()

        self.clip_text_model_loaded = False
        self.clip_vision_model_loaded = False
        self.ocr_models_loaded = False
        self.face_models_loaded = False

    # --- 【修复：移除 _empty_queue，因为它不安全】 ---
    # def _empty_queue(self, q: queue.Queue): ... (移除)

    def release_models(self):
        """(按需) 释放模型。此函数由空闲计时器或 /restart 调用。"""

        # --- 【修复：使用安全的 'get_nowait' 逻辑】 ---
        with self._load_lock:
            if not (self.face_models_loaded or self.ocr_models_loaded or self.clip_vision_model_loaded):
                logging.debug("没有按需模型需要释放。")
                return

            logging.warning("--- 正在释放 (按需) AI 模型 ---")

            # 1. 释放 CLIP Vision (按需)
            if self.clip_vision_model_loaded:
                try:
                    # 尝试获取模型，如果队列为空（即模型正在使用），则会引发 Empty
                    pooled_item = self.clip_vision_pool.get_nowait()

                    # 成功获取：表示模型空闲，可以安全释放
                    logging.warning("释放空闲的 CLIP Vision (按需) 实例...")
                    del pooled_item
                    if self.clip_vision_model:
                        del self.clip_vision_model
                        self.clip_vision_model = None
                    self.clip_image_preprocessor = None
                    self.clip_vision_model_loaded = False

                except queue.Empty:
                    # 队列为空：表示模型正在使用中，跳过本次释放
                    logging.debug("CLIP Vision (按需) 模型正在使用中，跳过释放。")
                except Exception as e:
                    logging.error(f"释放 CLIP Vision 时出错: {e}", exc_info=True)

            # 2. 释放 RapidOCR (按需)
            if self.ocr_models_loaded:
                try:
                    ocr_instance = self.ocr_pool.get_nowait()
                    logging.warning("释放空闲的 RapidOCR (按需) 实例...")
                    del ocr_instance
                    self.ocr_models_loaded = False
                except queue.Empty:
                    logging.debug("RapidOCR (按需) 模型正在使用中，跳过释放。")
                except Exception as e:
                    logging.error(f"释放 RapidOCR 时出错: {e}", exc_info=True)

            # 3. 释放 FaceAnalysis (按需)
            if self.face_models_loaded:
                try:
                    face_instance = self.face_pool.get_nowait()
                    logging.warning("释放空闲的 FaceAnalysis (按需) 实例...")
                    del face_instance
                    self.face_models_loaded = False
                except queue.Empty:
                    logging.debug("FaceAnalysis (按需) 模型正在使用中，跳过释放。")
                except Exception as e:
                    logging.error(f"释放 FaceAnalysis 时出错: {e}", exc_info=True)

            gc.collect()
            logging.debug("按需模型释放检查完成。")
        # --- 修复结束 ---

    def release_all_models(self):
        """(关闭时) 释放所有模型，包括常驻模型。"""

        with self._load_lock:
            logging.warning("--- 正在释放 *所有* AI 模型 (应用关闭) ---")

            # 1. 释放所有按需模型
            # (因为是 RLock，所以 release_models() 在这里再次获取锁是安全的)
            # 注意：此时我们调用 release_models 只是为了清理池中 *空闲* 的项
            self.release_models()

            # 2. 释放常驻的 Text CLIP 模型 (使用 'get_nowait' 逻辑)
            if self.clip_text_model_loaded:
                logging.warning("释放 Text CLIP (常驻) 模型...")
                try:
                    pooled_item = self.clip_text_pool.get_nowait()
                    del pooled_item
                except queue.Empty:
                    logging.warning("Text CLIP 在关闭时正在使用中。")
                except Exception as e:
                    logging.error(f"清空 Text CLIP 池时出错: {e}", exc_info=True)

                # 无论如何都要删除基础模型
                if self.clip_text_model:
                    del self.clip_text_model
                    self.clip_text_model = None
                self.clip_text_model_loaded = False
                logging.warning("Text CLIP (常驻) 模型已释放。")

            # 3. (关闭时) 强制清理按需模型的基础引用（即使它们在 release_models 中正在被使用）
            # 这是为了防止 release_models 跳过它们
            if self.clip_vision_model:
                del self.clip_vision_model
                self.clip_vision_model = None
            self.clip_image_preprocessor = None

            # 确保池在退出前是空的（即使实例在别处被引用）
            while not self.clip_vision_pool.empty(): self.clip_vision_pool.get_nowait()
            while not self.ocr_pool.empty(): self.ocr_pool.get_nowait()
            while not self.face_pool.empty(): self.face_pool.get_nowait()

            gc.collect()
            logging.warning("所有模型清理完毕 (应用关闭)。")
        # --- 修复结束 ---

    def _load_insightface(self) -> FaceAnalysis:
        logging.warning("Insightface 将显式加载通用的 'CPUExecutionProvider' (无 OpenVINO 加速)。")
        try:
            providers_generic_cpu = ['CPUExecutionProvider']
            face_app = FaceAnalysis(name=MODEL_NAME, root=self.insightface_root, providers=providers_generic_cpu)
            face_app.prepare(ctx_id=0, det_size=(64, 64))
            logging.warning("InsightFace 实例已在通用 CPU (回退模式) 上成功加载。")
            return face_app
        except Exception as final_fallback_e:
            logging.critical(f"InsightFace 在通用 CPU (回退) 模式下也加载失败: {final_fallback_e}", exc_info=True)
            raise final_fallback_e

    @staticmethod
    def _load_rapidocr() -> RapidOCR:
        ov_config = {"PERFORMANCE_HINT": "THROUGHPUT"}

        try:
            logging.debug(f"加载 RapidOCR (OpenVINO) (尝试使用 {INFERENCE_DEVICE})...")
            ocr = RapidOCR(device_name=INFERENCE_DEVICE, ov_config=ov_config)
            logging.warning(f"RapidOCR 实例在 {INFERENCE_DEVICE} (OpenVINO) 上加载成功 (Config: {ov_config})。")
            return ocr
        except Exception as e:
            logging.error(f"加载 RapidOCR 实例失败 (尝试使用 {INFERENCE_DEVICE}, Config: {ov_config}): {e}", exc_info=True)
            logging.warning("RapidOCR (OpenVINO) 加载失败，将回退到 CPU (OpenVINO)...")
            try:
                ocr = RapidOCR(device_name="CPU", ov_config=ov_config)
                logging.warning(f"RapidOCR 实例已在 CPU (OpenVINO 回退模式) 上成功加载 (Config: {ov_config})。")
                return ocr
            except Exception as fallback_e:
                logging.critical(f"RapidOCR 在 CPU (回退) 模式下也加载失败: {fallback_e}", exc_info=True)
                raise fallback_e

    def _compile_clip_vision_model(self) -> Tuple[Callable, ov.CompiledModel]:
        """编译并返回 Vision 模型及预处理器"""
        logging.warning(f"正在加载 QA-CLIP Vision ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            vision_model_path = os.path.join(self.qa_clip_path, "openvino_image_fp16.xml")
            if not os.path.exists(vision_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 Vision 模型文件。")

            config_vision = {"PERFORMANCE_HINT": "THROUGHPUT"}
            logging.warning(f"编译 Vision 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_vision['PERFORMANCE_HINT']})...")
            vision_compiled = self.core.compile_model(vision_model_path, INFERENCE_DEVICE, config_vision)

            image_preprocessor = image_transform(MODEL_INFO[MODEL_ARCH]['input_resolution'])

            if vision_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"视觉模型维度不匹配！")

            logging.warning(f"QA-CLIP Vision ({MODEL_ARCH}) 基础模型编译成功。")
            return image_preprocessor, vision_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP Vision 模型时发生严重错误: {e}", exc_info=True)
            raise

    def _compile_clip_text_model(self) -> ov.CompiledModel:
        """编译并返回 Text 模型"""
        logging.warning(f"正在加载 QA-CLIP Text ({MODEL_ARCH}) 模型: {self.qa_clip_path}")
        try:
            text_model_path = os.path.join(self.qa_clip_path, "openvino_text_fp16.xml")
            if not os.path.exists(text_model_path):
                raise FileNotFoundError(f"未在 '{self.qa_clip_path}' 找到 Text 模型文件。")

            config_text = {"PERFORMANCE_HINT": "LATENCY"}
            logging.warning(f"编译 Text 模型 (设备: {INFERENCE_DEVICE}, 提示: {config_text['PERFORMANCE_HINT']})...")
            text_compiled = self.core.compile_model(text_model_path, INFERENCE_DEVICE, config_text)

            if text_compiled.outputs[0].get_partial_shape()[1].get_length() != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(f"文本模型维度不匹配！")

            logging.warning(f"QA-CLIP Text ({MODEL_ARCH}) 基础模型编译成功。")
            return text_compiled
        except Exception as e:
            logging.error(f"加载 QA-CLIP Text 模型时发生严重错误: {e}", exc_info=True)
            raise

    # 按需加载 (Ensure) 方法

    def ensure_clip_text_model_loaded(self):
        """(常驻) 确保 Text CLIP 模型已加载。"""
        if self.clip_text_model_loaded:
            return
        with self._load_lock:
            if self.clip_text_model_loaded:
                return
            logging.warning("--- 正在加载 CLIP Text 模型 (常驻, 1个实例) ---")
            try:
                self.clip_text_model = self._compile_clip_text_model()
                infer_request = self.clip_text_model.create_infer_request()
                self.clip_text_pool.put((self.clip_text_model, infer_request))
                self.clip_text_model_loaded = True
                logging.warning(f"--- CLIP Text 模型已准备就绪 (1 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 Text CLIP 模型失败: {e}", exc_info=True)
                raise

    def ensure_clip_vision_model_loaded(self):
        """(按需) 确保 Vision CLIP 模型已加载。"""
        if self.clip_vision_model_loaded:
            return
        with self._load_lock:
            if self.clip_vision_model_loaded:
                return

            # --- 【修复：增加检查，防止在 release_models 竞争时重复加载】 ---
            # 这是一个双重保险，防止在 'get_nowait' 逻辑失败时，
            # 两个线程都尝试加载。
            if not self.clip_vision_pool.empty():
                logging.warning("ensure_clip_vision_model_loaded: 发现池非空但标志为False。重置标志并跳过加载。")
                self.clip_vision_model_loaded = True
                return
            # --- 修复结束 ---

            logging.warning("--- 正在加载 CLIP Vision 模型 (按需, 1个实例) ---")
            try:
                preprocessor, model = self._compile_clip_vision_model()
                self.clip_image_preprocessor = preprocessor
                self.clip_vision_model = model

                infer_request = model.create_infer_request()
                self.clip_vision_pool.put((preprocessor, model, infer_request))

                self.clip_vision_model_loaded = True
                logging.warning(f"--- CLIP Vision 模型已准备就绪 (1 个实例) ---")
            except Exception as e:
                logging.critical(f"加载 Vision CLIP 模型失败: {e}", exc_info=True)
                raise

    def ensure_ocr_models_loaded(self):
        """(按需) 确保 OCR 模型已加载 (实例池)。"""
        if self.ocr_models_loaded:
            return
        with self._load_lock:
            if self.ocr_models_loaded:
                return

            if not self.ocr_pool.empty():
                logging.warning("ensure_ocr_models_loaded: 发现池非空但标志为False。重置标志并跳过加载。")
                self.ocr_models_loaded = True
                return

            logging.warning(f"--- 正在加载 RapidOCR 模型 (按需, 1个实例) ---")
            try:
                logging.warning(f"创建 RapidOCR 实例 1/1...")
                ocr_instance = self._load_rapidocr()
                self.ocr_pool.put(ocr_instance)

                self.ocr_models_loaded = True
                logging.warning(f"--- RapidOCR 实例池已准备就绪 ---")
            except Exception as e:
                logging.critical(f"加载 RapidOCR 实例池失败: {e}", exc_info=True)
                raise

    def ensure_face_models_loaded(self):
        """(按需) 确保 FaceAnalysis 模型已加载 (实例池)。"""
        if self.face_models_loaded:
            return
        with self._load_lock:
            if self.face_models_loaded:
                return

            if not self.face_pool.empty():
                logging.warning("ensure_face_models_loaded: 发现池非空但标志为False。重置标志并跳过加载。")
                self.face_models_loaded = True
                return

            logging.warning(f"--- 正在加载 FaceAnalysis 模型 (按需, 1个实例) ---")
            try:
                logging.warning(f"创建 FaceAnalysis 实例 1/1...")
                face_instance = self._load_insightface()
                self.face_pool.put(face_instance)

                self.face_models_loaded = True
                logging.warning(f"--- FaceAnalysis 实例池已准备就绪 ---")
            except Exception as e:
                logging.critical(f"加载 FaceAnalysis 实例池失败: {e}", exc_info=True)
                raise

    # 'get' 方法保持不变 (它们是正确的)

    def get_face_representation(self, image: np.ndarray) -> List[RepresentResult]:
        self.ensure_face_models_loaded() # 按需加载
        face_analyzer_engine = None

        try:
            face_analyzer_engine = self.face_pool.get(timeout=10)
            if not face_analyzer_engine:
                logging.error("从池中获取 FaceAnalysis 实例失败。")
                raise Exception("FaceAnalysis model not loaded")

            faces = face_analyzer_engine.get(image)
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
        except queue.Empty:
            logging.error("获取 FaceAnalysis 实例超时（池已空）")
            raise Exception("FaceAnalysis model pool timeout")
        except Exception as e:
            logging.debug(f"处理人脸识别时出错或未找到人脸: {e}", exc_info=False)
            return []
        finally:
            if face_analyzer_engine:
                self.face_pool.put(face_analyzer_engine)


    def get_ocr_results(self, image: np.ndarray) -> OCRResult:
        self.ensure_ocr_models_loaded() # 按需加载
        rapid_ocr_engine = None

        try:
            rapid_ocr_engine = self.ocr_pool.get(timeout=10)
            if not rapid_ocr_engine:
                logging.error("从池中获取 RapidOCR 实例失败。")
                raise Exception("RapidOCR model not loaded")

            ocr_raw_output = rapid_ocr_engine(image)

            if not isinstance(ocr_raw_output, tuple) or len(ocr_raw_output) < 1:
                logging.warning(f"RapidOCR 返回了意外的格式: {type(ocr_raw_output)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            ocr_result, _ = ocr_raw_output

            if ocr_result is None:
                logging.debug("RapidOCR 返回 None 结果 (无文本)。")
                return OCRResult(texts=[], scores=[], boxes=[])

            if not isinstance(ocr_result, list):
                logging.warning(f"RapidOCR 的结果部分不是列表: {type(ocr_result)}")
                return OCRResult(texts=[], scores=[], boxes=[])

            if len(ocr_result) == 0:
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
        except queue.Empty:
            logging.error("获取 RapidOCR 实例超时（池已空）")
            raise Exception("RapidOCR model pool timeout")
        except Exception as e:
            if isinstance(e, RuntimeError) and "Infer Request is busy" in str(e):
                logging.error(f"OCR 出现并发错误 (Infer Request is busy): {e}", exc_info=True)
            else:
                logging.warning(f"处理 OCR 时出错: {e}", exc_info=True)
            return OCRResult(texts=[], scores=[], boxes=[])
        finally:
            if rapid_ocr_engine:
                self.ocr_pool.put(rapid_ocr_engine)

    def get_image_embedding(self, image: Image.Image, filename: str = "unknown") -> List[float]:
        self.ensure_clip_vision_model_loaded() # 按需加载

        pooled_item = None
        local_clip_image_preprocessor = None
        local_clip_vision_model = None
        infer_request = None

        try:
            pooled_item = self.clip_vision_pool.get(timeout=10)
            local_clip_image_preprocessor, local_clip_vision_model, infer_request = pooled_item

            if not local_clip_image_preprocessor or not local_clip_vision_model or not infer_request:
                logging.error("CLIP Vision 模型未正确加载 (从池中获取了 None)。")
                raise Exception("CLIP Vision model not loaded")

            inputs = local_clip_image_preprocessor(image).unsqueeze(0)
            pixel_values = inputs.numpy()

            results = infer_request.infer({local_clip_vision_model.inputs[0].any_name: pixel_values})
            embedding = results[local_clip_vision_model.outputs[0]]

            return [float(x) for x in embedding.flatten()]
        except queue.Empty:
            logging.error("获取 CLIP Vision 实例超时（池已空且无法加载）")
            raise Exception("CLIP Vision model pool timeout")
        except Exception as e:
            logging.error(f"在 get_image_embedding 中处理 '{filename}' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if pooled_item:
                self.clip_vision_pool.put(pooled_item)

    def get_text_embedding(self, text: str) -> List[float]:
        self.ensure_clip_text_model_loaded() # (常驻) 检查以确保加载

        pooled_item = None
        local_clip_text_model = None
        infer_request = None

        try:
            pooled_item = self.clip_text_pool.get(timeout=10)
            local_clip_text_model, infer_request = pooled_item

            if not local_clip_text_model or not infer_request:
                logging.error("CLIP Text 模型未正确加载 (从池中获取了 None)。")
                raise Exception("CLIP Text model not loaded")

            inputs_tensor = clip.tokenize([text], context_length=CONTEXT_LENGTH)
            input_ids = inputs_tensor.numpy()
            pad_index = clip._tokenizer.vocab['[PAD]']
            attention_mask = np.array(input_ids != pad_index).astype(np.int64)

            input_name_0 = local_clip_text_model.inputs[0].any_name
            input_name_1 = local_clip_text_model.inputs[1].any_name
            inputs_dict = {
                input_name_0: input_ids,
                input_name_1: attention_mask
            }
            results = infer_request.infer(inputs_dict)
            embedding = results[local_clip_text_model.outputs[0]]
            return [float(x) for x in embedding.flatten()]
        except queue.Empty:
            logging.error("获取 CLIP Text 实例超时（池已空且无法加载）")
            raise Exception("CLIP Text model pool timeout")
        except Exception as e:
            logging.error(f"在 get_text_embedding 中处理 '{text[:50]}...' 时发生严重错误: {e}", exc_info=True)
            return [0.0] * CLIP_EMBEDDING_DIMS
        finally:
            if pooled_item:
                self.clip_text_pool.put(pooled_item)