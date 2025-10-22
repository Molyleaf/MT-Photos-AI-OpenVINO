import os
from contextlib import asynccontextmanager
from typing import List
import logging

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# 导入我们重构后的模型处理模块
import models as ai_models

# --- 日志配置 ---
# 配置日志记录器，以便我们可以看到详细的调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API 密钥认证 ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt-photos-ai-openvino")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    # 如果没有设置 API_AUTH_KEY，则不进行验证
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key":
        return
    if api_key_header != API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

# --- Lifespan 事件管理 ---
# 创建一个全局变量来持有 AIModels 实例
models_instance: ai_models.AIModels

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    logging.info("应用启动... 开始加载 AI 模型。")
    try:
        # 在应用启动时，初始化 AIModels 类
        models_instance = ai_models.AIModels()
        logging.info("AI 模型加载成功。")
    except Exception as e:
        # 如果模型加载失败，记录严重错误并阻止应用启动
        logging.critical(f"严重错误: AI 模型无法初始化。错误详情: {e}", exc_info=True)
        # 在实际部署中，这会让 uvicorn 进程失败并退出，容器应该会重启
        raise RuntimeError(f"AI 模型初始化失败: {e}") from e

    yield

    logging.info("应用关闭。")

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。",
    version="2.0.0", # 版本号提升，代表重大重构
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

# --- Pydantic 模型定义 (请求与响应体) ---

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
    """从上传的文件中读取并解码图像，处理常见的颜色通道问题。"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 无法被解码为图像。")

    # 确保图像是 3 通道 BGR 格式
    if len(img.shape) == 2:  # 灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA (带透明通道)
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
        ocr_results = models_instance.get_ocr_results(image)
        return {"result": ocr_results}
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理 OCR 失败: {str(e)}")

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    logging.info(f"开始处理 CLIP 图像请求: {file.filename}")
    try:
        image = await read_image_from_upload(file)
        # 将 BGR 图像转换为 RGB 格式，因为 CLIP 模型是用 RGB 图像训练的
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding = models_instance.get_image_embedding(image_rgb, file.filename)

        response_data = {"results": embedding}
        logging.info(f"成功为 '{file.filename}' 生成 embedding。向量长度: {len(embedding)}")
        return response_data
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理 CLIP 失败: {str(e)}")

@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    try:
        embedding = models_instance.get_text_embedding(request.text)
        return {"results": embedding}
    except Exception as e:
        logging.error(f"处理 CLIP 文本请求失败: '{request.text[:50]}...', 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理 CLIP 文本失败: {str(e)}")

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        face_results = models_instance.get_face_representation(image)
        return {"result": face_results}
    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理人脸识别失败: {str(e)}")

# Uvicorn 服务器入口 (用于直接运行脚本进行测试)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8060))
    # 在 __main__ 块中运行 lifespan 管理器需要手动设置
    # 但在生产环境中，Uvicorn 会自动处理
    uvicorn.run("server_openvino:app", host="0.0.0.0", port=port, reload=True)
