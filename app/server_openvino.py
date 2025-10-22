import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import logging # 导入日志模块

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# 假设 models.py 在同一目录下或在 python path 中
import models as ai_models

# --- 日志配置 ---
# 配置日志记录器，以便我们可以看到详细的调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API 密钥认证 ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt-photos-ai-openvino")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key":
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
    global models
    logging.info("应用启动... 开始加载 AI 模型。")
    try:
        models = ai_models.AIModels()
        logging.info("AI 模型加载成功。")
    except Exception as e:
        logging.critical(f"严重错误: AI 模型无法初始化。错误详情: {e}", exc_info=True)
        raise RuntimeError(f"AI 模型初始化失败: {e}") from e

    yield
    logging.info("应用关闭。")

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。",
    version="1.2.0", # 版本号提升
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

# --- Pydantic 模型定义 (与之前保持一致) ---
class TextClipRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    result: str

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
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 无法被解码为图像。")

    if len(img.shape) == 2: # 灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # BGRA (带透明通道)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

# --- API 端点定义 ---
@app.post("/check", response_model=CheckResponse)
async def check_service(t: str = ""):
    return {"result": "pass"}

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        ocr_results = models.get_ocr_results(image)
        return {"result": ocr_results}
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理 OCR 失败: {e}")

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    # FIX: 增加完整的日志记录和错误处理
    logging.info(f"开始处理 CLIP 图像请求: {file.filename}")
    try:
        image = await read_image_from_upload(file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding = models.get_image_embedding(image_rgb, file.filename)

        # 准备返回的数据
        response_data = {"results": embedding}

        # 关键日志：记录返回给客户端的确切内容长度和前几个元素
        logging.info(f"成功为 '{file.filename}' 生成 embedding。向量长度: {len(embedding)}, 前5个元素: {embedding[:5]}")

        return response_data
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        # 即使发生错误，也抛出 HTTP 异常，而不是让连接中断
        raise HTTPException(status_code=500, detail=f"处理 CLIP 失败: {e}")


@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    try:
        embedding = models.get_text_embedding(request.text)
        return {"results": embedding}
    except Exception as e:
        logging.error(f"处理 CLIP 文本请求失败: {request.text}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理 CLIP 文本失败: {e}")


@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        face_results = models.get_face_representation(image)
        return {"result": face_results}
    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理人脸识别失败: {e}")

# Uvicorn 服务器入口
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8060))
    uvicorn.run(app, host="0.0.0.0", port=port)

