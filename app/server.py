# app/server.py
import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

import models as ai_models
from schemas import (
    CheckResponse,
    OCRResult,
    TextClipRequest,
    RepresentResponse,
    RestartResponse
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt_photos_ai_extra")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key":
        return
    if api_key_header != API_AUTH_KEY:
        logging.warning(f"拒绝了无效的 API 密钥: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥",
        )

SERVER_IDLE_TIMEOUT = int(os.environ.get("SERVER_IDLE_TIMEOUT", "300"))
idle_timer: threading.Timer = None
models_instance: ai_models.AIModels = None

def idle_timeout_handler():
    global models_instance
    if models_instance and models_instance.models_loaded:
        logging.info(f"服务器已空闲 {SERVER_IDLE_TIMEOUT} 秒。正在释放模型以节省内存...")
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            models_instance.release_models()
        else:
            logging.error("AIModels 实例没有 release_models 方法 (idle_timeout_handler)。")

def reset_idle_timer():
    global idle_timer
    if idle_timer:
        idle_timer.cancel()
    if SERVER_IDLE_TIMEOUT > 0:
        idle_timer = threading.Timer(SERVER_IDLE_TIMEOUT, idle_timeout_handler)
        idle_timer.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    logging.info("应用启动... 初始化 AIModels 实例。")
    models_instance = ai_models.AIModels()
    reset_idle_timer()
    yield
    global idle_timer
    if idle_timer:
        idle_timer.cancel()
    logging.info("应用关闭。正在释放所有模型...")
    if models_instance:
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            models_instance.release_models()
        else:
            logging.warning("AIModels 实例没有 release_models 方法 (lifespan)。")

app = FastAPI(
    title="MT-Photos AI 统一服务 (OpenVINO 版本)",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务 (支持自动内存释放)。",
    version="2.1.0",
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

@app.middleware("http")
async def reset_timer_middleware(request: Request, call_next):
    if request.url.path not in ["/check", "/"]:
        reset_idle_timer()
    response = await call_next(request)
    return response

async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # 使用 cv2.imdecode 在后台线程运行以避免阻塞事件循环 (特别是对于大文件)
    img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_UNCHANGED)
    # img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) # 旧的阻塞方式

    if img is None:
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 无法被解码为图像。")

    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

    if width > 10000 or height > 10000:
        raise HTTPException(status_code=400, detail="height or width out of range")

    # 色彩空间转换通常很快，可以在主线程进行，但也可以移到后台线程
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

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
    return {
        "result": "pass",
        "title": "MT-Photos AI 统一服务 (OpenVINO 版本)",
        "help": "https://mtmt.tech/docs/advanced/ocr_api"
    }

@app.post("/restart", response_model=RestartResponse)
async def restart_service():
    logging.info("收到 /restart 请求，正在手动释放模型...")
    if models_instance:
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            # release_models 是 CPU 密集型操作 (del, gc.collect)，也可以考虑放入线程
            # 但通常很快，且不频繁调用，放在主线程问题不大
            models_instance.release_models()
        else:
            logging.error("'AIModels' object has no attribute 'release_models' during restart request!")
    return {"result": "pass"}

@app.post("/restart_v2", response_model=RestartResponse)
async def restart_process():
    logging.warning("收到 /restart_v2 请求，将重启整个服务进程！")
    def delayed_restart():
        import time
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, *sys.argv)
    # 启动新线程执行重启操作
    threading.Thread(target=delayed_restart).start()
    return {"result": "pass"}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        # --- 使用 asyncio.to_thread 运行阻塞的模型推理 ---
        ocr_results_obj = await asyncio.to_thread(models_instance.get_ocr_results, image)
        # --- 结束修改 ---
        return {"result": ocr_results_obj.dict()}
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        return {"result": OCRResult(texts=[], scores=[], boxes=[]).dict()}

@app.post("/clip/img")
async def clip_image_endpoint(file: UploadFile = File(...)):
    logging.info(f"开始处理 CLIP 图像请求: {file.filename}")
    try:
        image = await read_image_from_upload(file)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 这步很快，可以在主线程
        # image_pil = Image.fromarray(image_rgb)           # 这步也很快
        # 可以将这两步也放入 to_thread，但收益可能不大
        image_pil = await asyncio.to_thread(lambda img_arr: Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)), image)

        # --- 使用 asyncio.to_thread 运行阻塞的模型推理 ---
        embedding = await asyncio.to_thread(models_instance.get_image_embedding, image_pil, file.filename)
        # --- 结束修改 ---
        result_strings = [f"{f:.16f}" for f in embedding]
        # logging.info(f"成功为 '{file.filename}' 生成 embedding。") # 精简日志
        return {"result": result_strings}
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        return {"result": []}

@app.post("/clip/txt")
async def clip_text_endpoint(request: TextClipRequest):
    try:
        # --- 使用 asyncio.to_thread 运行阻塞的模型推理 ---
        embedding = await asyncio.to_thread(models_instance.get_text_embedding, request.text)
        # --- 结束修改 ---
        result_strings = [f"{f:.16f}" for f in embedding]
        # logging.info(f"成功为文本 '{request.text[:30]}...' 生成 embedding。") # 精简日志
        return {"result": result_strings}
    except Exception as e:
        logging.error(f"处理 CLIP 文本请求失败: '{request.text[:50]}...', 错误: {e}", exc_info=True)
        return {"result": []}

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    try:
        image = await read_image_from_upload(file)
        # --- 使用 asyncio.to_thread 运行阻塞的模型推理 ---
        face_results_list = await asyncio.to_thread(models_instance.get_face_representation, image)
        # --- 结束修改 ---
        return RepresentResponse(result=face_results_list)
    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理人脸识别失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8060))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    # 考虑增加 worker 数量以利用多核 CPU 处理 I/O 和 Python 代码
    workers = int(os.environ.get("WEB_CONCURRENCY", 1)) # 默认为 1 个 worker
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level=log_level, workers=workers)