import os
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from common.models import AIModels, models

API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt-photos-ai-openvino")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not API_AUTH_KEY:
        return
    if api_key_header != API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。",
    version="1.0.0",
    dependencies=[Depends(get_api_key)]
)

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

async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无效的图像文件")
    return img

@app.on_event("startup")
async def startup_event():
    global models
    print("应用启动... 开始加载 AI 模型。")
    try:
        models = AIModels()
        print("AI 模型加载成功。")
    except Exception as e:
        print(f"严重错误: AI 模型无法初始化。错误详情: {e}")
        raise RuntimeError(f"AI 模型初始化失败: {e}") from e

@app.post("/check", response_model=CheckResponse)
async def check_service():
    return {"result": "pass"}

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    ocr_results = models.get_ocr_results(image)
    return {"result": oocr_results}

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
    if not face_results:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return {"result": face_results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)