import os
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# 假设 models.py 在同一目录下或在 python path 中
import models as ai_models

# --- API 密钥认证 ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt-photos-ai-openvino")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key":
        # 如果没有设置密钥或明确设置为 "no-key"，则不进行验证
        return
    if api_key_header != API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

# --- Lifespan 事件管理 ---
models: ai_models.AIModels

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行的代码
    global models
    print("应用启动... 开始加载 AI 模型。")
    try:
        models = ai_models.AIModels()
        print("AI 模型加载成功。")
    except Exception as e:
        print(f"严重错误: AI 模型无法初始化。错误详情: {e}")
        # 在实际生产中，您可能希望应用在这种情况下无法启动
        raise RuntimeError(f"AI 模型初始化失败: {e}") from e

    yield
    # 应用关闭时执行的代码
    print("应用关闭。")

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。",
    version="1.1.0",
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

# --- Pydantic 模型定义 ---
class TextClipRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    result: str

# FIX: 更新 OCR 响应模型以匹配 mt-photos-ai 的期望
class OCRBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class OCRResult(BaseModel):
    texts: List[str]
    scores: List[float]
    boxes: List[OCRBox]

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
    recognition_model: str = os.environ.get("MODEL_NAME", "buffalo_l")
    result: List[RepresentResult]

# --- 辅助函数 ---
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # 使用 cv2.IMREAD_UNCHANGED 来支持带有 alpha 通道的图像 (例如 PNG)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="无效的图像文件或不支持的格式")

    # 如果图像有 alpha 通道，将其转换为 BGR
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

# --- API 端点定义 ---
@app.post("/check", response_model=CheckResponse)
async def check_service(t: str = ""):
    return {"result": "pass"}

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    ocr_results = models.get_ocr_results(image)
    # 直接返回符合新模型格式的结果
    return {"result": ocr_results}

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embedding = models.get_image_embedding(image_rgb)
    return {"results": embedding}

@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    embedding = models.get_text_embedding(request.text)
    return {"results": embedding}

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    face_results = models.get_face_representation(image)

    # FIX: 即使没有检测到人脸，也返回 200 OK 和一个空的 result 列表
    # if not face_results:
    #     return Response(status_code=status.HTTP_204_NO_CONTENT)
    return {"result": face_results}

# Uvicorn 服务器入口
if __name__ == "__main__":
    import uvicorn
    # 允许通过环境变量配置端口
    port = int(os.environ.get("PORT", 8060))
    uvicorn.run(app, host="0.0.0.0", port=port)
