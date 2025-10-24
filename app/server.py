# app/server.py
import os
from contextlib import asynccontextmanager
from typing import List
import logging
import asyncio
import threading # 用于实现空闲超时
import sys

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from PIL import Image

# 导入我们重构后的模型处理模块
import models as ai_models

# --- 修正: 导入共享的 schemas.py 文件 ---
from schemas import (
    CheckResponse,
    OCRResponse,
    OCRResult,
    OCRBox,
    ClipResponse,
    TextClipRequest,
    RepresentResponse,
    RestartResponse
)
# --- 结束修正 ---

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API 密钥认证 ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt_photos_ai_extra")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key":
        # 如果未设置密钥，则允许所有请求
        return
    if api_key_header != API_AUTH_KEY:
        logging.warning(f"拒绝了无效的 API 密钥: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

# --- 自动内存管理 (任务 7) ---
# 从环境变量读取超时时间，默认 300 秒 (5分钟)
SERVER_IDLE_TIMEOUT = int(os.environ.get("SERVER_IDLE_TIMEOUT", "300"))
idle_timer: threading.Timer = None
models_instance: ai_models.AIModels = None

def idle_timeout_handler():
    """当服务器空闲超时时调用此函数。"""
    global models_instance
    if models_instance and models_instance.models_loaded:
        logging.info(f"服务器已空闲 {SERVER_IDLE_TIMEOUT} 秒。正在释放模型以节省内存...")
        models_instance.release_models()

def reset_idle_timer():
    """重置空闲计时器。"""
    global idle_timer
    if idle_timer:
        idle_timer.cancel()
    # 仅当超时时间大于 0 时才启动计时器
    if SERVER_IDLE_TIMEOUT > 0:
        idle_timer = threading.Timer(SERVER_IDLE_TIMEOUT, idle_timeout_handler)
        idle_timer.start()

# --- Lifespan 事件管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    logging.info("应用启动... 初始化 AIModels 实例。")
    # 启动时不加载模型，实现按需加载
    models_instance = ai_models.AIModels()
    # 启动空闲计时器（如果超时 > 0）
    reset_idle_timer()
    yield
    # 应用关闭时
    global idle_timer
    if idle_timer:
        idle_timer.cancel()
    logging.info("应用关闭。正在释放所有模型...")
    if models_instance:
        models_instance.release_models()

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务 (支持自动内存释放)。",
    version="2.1.0",
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

# --- 中间件：重置空闲计时器 ---
@app.middleware("http")
async def reset_timer_middleware(request: Request, call_next):
    # 重置计时器以响应任何 API 调用
    # (排除 /check 和 / 端点，以免它们保持模型活动)
    if request.url.path not in ["/check", "/"]:
        reset_idle_timer()
    response = await call_next(request)
    return response

# --- 辅助函数 ---
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """从上传的文件中读取并解码图像，处理常见的颜色通道问题。"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 无法被解码为图像。")

    # 检查图像尺寸 (匹配参考脚本)
    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

    if width > 10000 or height > 10000:
        raise HTTPException(status_code=400, detail="height or width out of range")

    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

# --- API 端点定义 ---
@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务 (OpenVINO)</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/check", response_model=CheckResponse)
async def check_service(t: str = ""):
    # 此端点不加载模型，仅用于健康检查
    # 也不重置空闲计时器
    return {
        "result": "pass",
        "title": "MT-Photos AI 统一服务 (OpenVINO 版本)",
        "help": "https://mtmt.tech/docs/advanced/ocr_api"
    }

@app.post("/restart", response_model=RestartResponse)
async def restart_service():
    """
    手动触发模型释放，以释放内存。(任务 7)
    模型将在下次 API 调用时自动重新加载。
    """
    logging.info("收到 /restart 请求，正在手动释放模型...")
    if models_instance:
        models_instance.release_models()
    return {"result": "pass"}

@app.post("/restart_v2", response_model=RestartResponse)
async def restart_process():
    """
    触发整个 Python 进程重启。
    （注意：这在 Docker 中可能导致容器退出，除非有 supervisor）
    """
    logging.warning("收到 /restart_v2 请求，将重启整个服务进程！")

    # 立即返回响应，然后在后台重启
    def delayed_restart():
        # 延迟1秒以确保响应已发送
        import time
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    threading.Thread(target=delayed_restart).start()
    return {"result": "pass"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        # ensure_models_loaded() 会在 get_ocr_results 内部调用
        ocr_results_obj = models_instance.get_ocr_results(image)
        # 匹配参考脚本响应格式 (任务 8)
        return {"result": ocr_results_obj, "msg": "ok"}
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        # 匹配参考脚本的错误格式 (任务 8)
        return OCRResponse(result=OCRResult(texts=[], scores=[], boxes=[]), msg=str(e))

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    logging.info(f"开始处理 CLIP 图像请求: {file.filename}")
    try:
        image = await read_image_from_upload(file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # ensure_models_loaded() 会在 get_image_embedding 内部调用
        embedding = models_instance.get_image_embedding(image_pil, file.filename)

        # 格式化为字符串列表 (匹配参考脚本) (任务 8)
        result_strings = [f"{f:.16f}" for f in embedding]

        logging.info(f"成功为 '{file.filename}' 生成 embedding。向量长度: {len(embedding)}")
        return {"result": result_strings, "msg": "ok"}
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        # 匹配参考脚本错误格式 (任务 8)
        return ClipResponse(result=[], msg=str(e))

@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    try:
        # ensure_models_loaded() 会在 get_text_embedding 内部调用
        embedding = models_instance.get_text_embedding(request.text)

        # 格式化为字符串列表 (匹配参考脚本) (任务 8)
        result_strings = [f"{f:.16f}" for f in embedding]

        logging.info(f"成功为文本 '{request.text[:30]}...' 生成 embedding。向量长度: {len(embedding)}")
        return {"result": result_strings, "msg": "ok"}
    except Exception as e:
        logging.error(f"处理 CLIP 文本请求失败: '{request.text[:50]}...', 错误: {e}", exc_info=True)
        # 匹配参考脚本错误格式 (任务 8)
        return ClipResponse(result=[], msg=str(e))

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    """ InsightFace 人脸识别端点 (任务 4) """
    try:
        image = await read_image_from_upload(file)
        # ensure_models_loaded() 会在 get_face_representation 内部调用
        face_results_list = models_instance.get_face_representation(image)

        # --- 修正: face_results_list 现在是 [schemas.RepresentResult]
        # Pydantic V2 会自动处理，无需转换
        return RepresentResponse(result=face_results_list)
        # --- 结束修正 ---

    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True)
        # InsightFace 端点的错误响应保持不变
        raise HTTPException(status_code=500, detail=f"处理人脸识别失败: {str(e)}")

# Uvicorn 服务器入口
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8060))
    # 在生产环境中不应使用 reload=True
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)