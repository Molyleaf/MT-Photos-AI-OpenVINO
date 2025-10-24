# app/server.py
import os
from contextlib import asynccontextmanager
from typing import List
import logging
import asyncio
import threading #
import sys

import cv2 #
import numpy as np #
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request #
from fastapi.responses import HTMLResponse #
from fastapi.security import APIKeyHeader #
from PIL import Image #

import models as ai_models #

from schemas import ( #
    CheckResponse,
    OCRResponse,
    OCRResult,
    OCRBox,
    ClipResponse,
    TextClipRequest,
    RepresentResponse,
    RestartResponse
)

# 配置日志记录器 - 可以考虑将 INFO 改为 WARNING 以减少生产日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') #

API_AUTH_KEY = os.environ.get("API_AUTH_KEY", "mt_photos_ai_extra") #
API_KEY_NAME = "api-key" #
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) #

async def get_api_key(api_key_header: str = Depends(api_key_header)): #
    if not API_AUTH_KEY or API_AUTH_KEY == "no-key": #
        return #
    if api_key_header != API_AUTH_KEY: #
        logging.warning(f"拒绝了无效的 API 密钥: {api_key_header}") # 保留警告
        raise HTTPException( #
            status_code=status.HTTP_401_UNAUTHORIZED, #
            detail="无效的 API 密钥", #
        )

SERVER_IDLE_TIMEOUT = int(os.environ.get("SERVER_IDLE_TIMEOUT", "300")) #
idle_timer: threading.Timer = None #
models_instance: ai_models.AIModels = None #

def idle_timeout_handler(): #
    """当服务器空闲超时时调用此函数。"""
    global models_instance #
    if models_instance and models_instance.models_loaded: #
        logging.info(f"服务器已空闲 {SERVER_IDLE_TIMEOUT} 秒。正在释放模型以节省内存...") # 保留 Info
        # 调用 release_models() 内部会记录日志
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            models_instance.release_models()
        else:
            logging.error("AIModels 实例没有 release_models 方法 (idle_timeout_handler)。")


def reset_idle_timer(): #
    """重置空闲计时器。"""
    global idle_timer #
    if idle_timer: #
        idle_timer.cancel() #
    if SERVER_IDLE_TIMEOUT > 0: #
        idle_timer = threading.Timer(SERVER_IDLE_TIMEOUT, idle_timeout_handler) #
        idle_timer.start() #

@asynccontextmanager
async def lifespan(app: FastAPI): #
    global models_instance #
    logging.info("应用启动... 初始化 AIModels 实例。") # 保留 Info
    models_instance = ai_models.AIModels() #
    reset_idle_timer() #
    yield #
    global idle_timer #
    if idle_timer: #
        idle_timer.cancel() #
    logging.info("应用关闭。正在释放所有模型...") # 保留 Info
    if models_instance: #
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            models_instance.release_models() #
        else:
            logging.warning("AIModels 实例没有 release_models 方法 (lifespan)。")


app = FastAPI( #
    title="MT-Photos AI 统一服务 (OpenVINO 版本)", #
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务 (支持自动内存释放)。", #
    version="2.1.0", #
    dependencies=[Depends(get_api_key)], #
    lifespan=lifespan #
)

@app.middleware("http") #
async def reset_timer_middleware(request: Request, call_next): #
    if request.url.path not in ["/check", "/"]: #
        reset_idle_timer() #
    response = await call_next(request) #
    return response #

async def read_image_from_upload(file: UploadFile) -> np.ndarray: #
    """从上传的文件中读取并解码图像，处理常见的颜色通道问题。"""
    contents = await file.read() #
    nparr = np.frombuffer(contents, np.uint8) #
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) #

    if img is None: #
        raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 无法被解码为图像。") #

    height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1) #

    if width > 10000 or height > 10000: #
        raise HTTPException(status_code=400, detail="height or width out of range") #

    if channels == 1: #
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #
    elif channels == 4: #
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) #

    return img #

@app.get("/", response_class=HTMLResponse) #
async def top_info(): #
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
</html>""" #
    return HTMLResponse(content=html_content) #

@app.post("/check", response_model=CheckResponse) #
async def check_service(t: str = ""): #
    # Keep this endpoint light, no extra logs needed normally
    return { #
        "result": "pass", #
        "title": "MT-Photos AI 统一服务 (OpenVINO 版本)", #
        "help": "https://mtmt.tech/docs/advanced/ocr_api" #
    }

@app.post("/restart", response_model=RestartResponse) #
async def restart_service(): #
    """手动触发模型释放，以释放内存。"""
    logging.info("收到 /restart 请求，正在手动释放模型...") # 保留 Info
    if models_instance: #
        if hasattr(models_instance, 'release_models') and callable(models_instance.release_models):
            models_instance.release_models() # release_models() 内部会记录日志
        else:
            logging.error("'AIModels' object has no attribute 'release_models' during restart request!")
    return {"result": "pass"} #

@app.post("/restart_v2", response_model=RestartResponse) #
async def restart_process(): #
    """触发整个 Python 进程重启。"""
    logging.warning("收到 /restart_v2 请求，将重启整个服务进程！") # 保留警告

    def delayed_restart(): #
        import time #
        time.sleep(1) #
        python = sys.executable #
        os.execl(python, python, *sys.argv) #

    threading.Thread(target=delayed_restart).start() #
    return {"result": "pass"} #


@app.post("/ocr") #
async def ocr_endpoint(file: UploadFile = File(...)): #
    # 可以在这里加一个 Info 日志，但如果请求量大可能会刷屏
    # logging.info(f"开始处理 OCR 请求: {file.filename}")
    try:
        image = await read_image_from_upload(file) #
        ocr_results_obj = models_instance.get_ocr_results(image) #
        # 成功时不记录日志，减少噪音
        return {"result": ocr_results_obj.dict()} #
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True) # 保留错误
        return {"result": OCRResult(texts=[], scores=[], boxes=[]).dict()} #

@app.post("/clip/img") #
async def clip_image_endpoint(file: UploadFile = File(...)): #
    logging.info(f"开始处理 CLIP 图像请求: {file.filename}") # 保留 Info
    try:
        image = await read_image_from_upload(file) #
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #
        image_pil = Image.fromarray(image_rgb) #
        embedding = models_instance.get_image_embedding(image_pil, file.filename) #
        result_strings = [f"{f:.16f}" for f in embedding] #
        # 移除成功日志，只在失败时记录
        # logging.info(f"成功为 '{file.filename}' 生成 embedding。向量长度: {len(embedding)}")
        return {"result": result_strings}
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True) # 保留错误
        return {"result": []} #

@app.post("/clip/txt") #
async def clip_text_endpoint(request: TextClipRequest): #
    # logging.info(f"开始处理 CLIP 文本请求: {request.text[:30]}...") # 可以按需开启
    try:
        embedding = models_instance.get_text_embedding(request.text) #
        result_strings = [f"{f:.16f}" for f in embedding] #
        # 移除成功日志
        # logging.info(f"成功为文本 '{request.text[:30]}...' 生成 embedding。向量长度: {len(embedding)}")
        return {"result": result_strings}
    except Exception as e:
        logging.error(f"处理 CLIP 文本请求失败: '{request.text[:50]}...', 错误: {e}", exc_info=True) # 保留错误
        return {"result": []} #

@app.post("/represent", response_model=RepresentResponse) #
async def represent_endpoint(file: UploadFile = File(...)): #
    """ InsightFace 人脸识别端点 """
    # logging.info(f"开始处理人脸识别请求: {file.filename}") # 可以按需开启
    try:
        image = await read_image_from_upload(file) #
        face_results_list = models_instance.get_face_representation(image) #
        # 成功时不记录日志
        return RepresentResponse(result=face_results_list) #
    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True) # 保留错误
        raise HTTPException(status_code=500, detail=f"处理人脸识别失败: {str(e)}") #

if __name__ == "__main__": #
    import uvicorn #
    port = int(os.environ.get("PORT", 8060)) #
    # 获取日志级别环境变量，默认为 INFO
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level=log_level) # 通过环境变量控制日志级别