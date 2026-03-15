"""当前服务入口与模型运行时共享的 Pydantic 模型。"""

from typing import List

from pydantic import BaseModel


class CheckResponse(BaseModel):
    result: str
    title: str = "mt-photos-ai服务"
    help: str = "https://mtmt.tech/docs/advanced/ocr_api"


class OCRBox(BaseModel):
    x: str
    y: str
    width: str
    height: str


class OCRResult(BaseModel):
    texts: List[str]
    scores: List[str]
    boxes: List[OCRBox]


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int


class RepresentResult(BaseModel):
    embedding: List[float]
    facial_area: FacialArea
    face_confidence: float


class RestartResponse(BaseModel):
    result: str
