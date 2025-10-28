# app/schemas.py
# --------------------------------------------------
# 这个文件包含所有在 server 和 models 之间共享的 Pydantic 模型
# --------------------------------------------------
import os
from typing import List
from pydantic import BaseModel

# --- /check 响应 ---
class CheckResponse(BaseModel):
    result: str
    title: str = "MT-Photos AI 统一服务 (OpenVINO 版本)"
    help: str = "https://github.com/Molyleaf/MT-Photos-AI-OpenVINO"

# --- /ocr 响应 ---
class OCRBox(BaseModel):
    x: str
    y: str
    width: str
    height: str

class OCRResult(BaseModel):
    texts: List[str]
    scores: List[str]
    boxes: List[OCRBox]

class OCRResponse(BaseModel):
    result: OCRResult
    msg: str = "ok"

# --- /clip 请求与响应 ---
class TextClipRequest(BaseModel):
    text: str

class ClipResponse(BaseModel):
    result: List[str]
    msg: str = "ok"

# --- /represent 响应 ---
class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int

class RepresentResult(BaseModel):
    embedding: List[float]
    facial_area: FacialArea
    face_confidence: float

class RepresentResponse(BaseModel):
    detector_backend: str = "insightface"
    recognition_model: str = os.environ.get("MODEL_NAME", "buffalo_l")
    result: List[RepresentResult]

# --- /restart 响应 ---
class RestartResponse(BaseModel):
    result: str