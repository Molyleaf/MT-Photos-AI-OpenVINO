import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import io
from typing import Optional, List, Dict, Any

# 导入模型管理类
from common.models import models, AIModels

# --- API 密钥认证 ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    """用于验证 API 密钥的依赖项。"""
    if not API_AUTH_KEY: # 如果环境变量中未设置密钥，则允许访问
        return
    if api_key_header != API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。",
    version="1.0.0",
    dependencies=[Depends(get_api_key)] # 对所有端点应用 API 密钥认证
)

# --- Pydantic 模型，用于定义请求和响应的数据结构 ---
class TextClipRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    result: str

class OCRResult(BaseModel):
    texts: List[str]
    scores: List[float]
    boxes: List[List[List[int]]]

class OCRResponse(BaseModel):
    result: OCRResult

class ClipResponse(BaseModel):
    results: List[float]

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
    recognition_model: str = "buffalo_l"
    result: List[RepresentResult]

# --- 辅助函数 ---
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """从上传的文件中读取数据并将其转换为 CV2 图像对象。"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无效的图像文件")
    return img

# --- API 端点定义 ---
@app.on_event("startup")
async def startup_event():
    """在应用启动时检查模型是否已成功加载。"""
    if models is None:
        # 如果模型加载失败，将阻止应用启动
        raise RuntimeError("严重错误: AI 模型无法初始化。请检查日志获取详细错误信息。")

@app.post("/check", response_model=CheckResponse)
async def check_service():
    """检查服务可用性和 API 密钥有效性。"""
    return {"result": "pass"}

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """对上传的图像执行 OCR。"""
    image = await read_image_from_upload(file)
    ocr_results = models.get_ocr_results(image)
    return {"result": ocr_results}

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    """从上传的图像中提取特征向量。"""
    image = await read_image_from_upload(file)
    # 将图像从 BGR (cv2 默认格式) 转换为 RGB (CLIP 模型需要)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embedding = models.get_image_embedding(image_rgb)
    return {"results": embedding}

@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    """从文本字符串中提取特征向量。"""
    embedding = models.get_text_embedding(request.text)
    return {"results": embedding}

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    """从上传的图像中检测人脸并提取特征向量。"""
    image = await read_image_from_upload(file)
    face_results = models.get_face_representation(image)
    return {"result": face_results}

# --- Uvicorn 服务器的程序入口 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)

