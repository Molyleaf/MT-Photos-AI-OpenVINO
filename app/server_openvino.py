import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional, Dict
from pydantic import BaseModel

from app.common.models import ModelManager

# 加载环境变量
load_dotenv()

API_AUTH_KEY = os.getenv("API_AUTH_KEY")
if not API_AUTH_KEY:
    raise ValueError("API_AUTH_KEY environment variable not set!")

# 初始化 FastAPI 应用
app = FastAPI(title="MT-Photos Unified AI Service (OpenVINO)")

# 在应用启动时加载模型
@app.on_event("startup")
def load_models_on_startup():
    app.state.model_manager = ModelManager(models_root_dir="./models")

# API 密钥验证依赖
async def verify_api_key(api_key: Optional[str] = Header(None)):
    if api_key!= API_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class TextClipRequest(BaseModel):
    text: str

# --- API 端点定义 ---

@app.post("/check", dependencies=)
async def check_service():
    """健康检查和API密钥验证端点"""
    return {"result": "pass"}

@app.post("/ocr", dependencies=)
async def ocr_endpoint(request: Request, file: UploadFile = File(...)):
    """从图像中提取文本 (OCR)"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model_manager: ModelManager = request.app.state.model_manager
    result = model_manager.get_ocr_results(img)
    return result

@app.post("/clip/img", dependencies=)
async def clip_image_endpoint(request: Request, file: UploadFile = File(...)):
    """提取图像的CLIP特征向量"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model_manager: ModelManager = request.app.state.model_manager
    result = model_manager.get_image_embedding(img_rgb)
    return result

@app.post("/clip/txt", dependencies=)
async def clip_text_endpoint(request: Request, payload: TextClipRequest):
    """提取文本的CLIP特征向量"""
    model_manager: ModelManager = request.app.state.model_manager
    result = model_manager.get_text_embedding(payload.text)
    return result

@app.post("/represent", dependencies=)
async def represent_endpoint(request: Request, file: UploadFile = File(...)):
    """人脸检测和识别"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model_manager: ModelManager = request.app.state.model_manager
    result = model_manager.get_face_representation(img)
    return result