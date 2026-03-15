# app/server.py
import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional, Tuple

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_LOG_FILE = os.path.join(_PROJECT_ROOT, "server.log")
_LOG_NAMESPACE = "mt_photos_ai"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_APP_LOG_HANDLER_FLAG = "_mt_photos_ai_handler"
_KNOWN_LOGGER_NAMES = (
    _LOG_NAMESPACE,
    f"{_LOG_NAMESPACE}.server",
    f"{_LOG_NAMESPACE}.models",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "rapidocr",
    "rapidocr.utils.log",
)


def _resolve_log_level() -> tuple[str, int]:
    configured_name = str(os.environ.get("LOG_LEVEL", "WARNING")).strip().upper() or "WARNING"
    resolved_level = getattr(logging, configured_name, logging.WARNING)
    resolved_name = logging.getLevelName(resolved_level)
    if not isinstance(resolved_name, str):
        resolved_name = "WARNING"
        resolved_level = logging.WARNING
    return resolved_name, int(resolved_level)


def _synchronize_known_logger_levels(log_level: int) -> None:
    for logger_name in _KNOWN_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(log_level)


def _configure_application_logging() -> None:
    _, log_level = _resolve_log_level()
    namespace_logger = logging.getLogger(_LOG_NAMESPACE)
    namespace_logger.setLevel(log_level)
    namespace_logger.propagate = False
    _synchronize_known_logger_levels(log_level)

    configured_handlers = [
        handler
        for handler in namespace_logger.handlers
        if getattr(handler, _APP_LOG_HANDLER_FLAG, False)
    ]
    if configured_handlers:
        for handler in configured_handlers:
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        return

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    setattr(console_handler, _APP_LOG_HANDLER_FLAG, True)
    namespace_logger.addHandler(console_handler)

    if sys.platform == "win32":
        try:
            file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
        except Exception as exc:
            print(f"无法设置文件日志: {exc}")
        else:
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, _APP_LOG_HANDLER_FLAG, True)
            namespace_logger.addHandler(file_handler)


def _configure_standalone_logging() -> None:
    _configure_application_logging()


_configure_application_logging()

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from models.constants import MODEL_NAME
from models.runtime import AIModels
from models.schemas import (
    CheckResponse,
    TextClipRequest,
    RestartResponse,
)

LOGGER = logging.getLogger(f"{_LOG_NAMESPACE}.server")


API_AUTH_KEY_DEFAULT = "mt_photos_ai_extra"
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _get_api_auth_key() -> str:
    return os.environ.get("API_AUTH_KEY", API_AUTH_KEY_DEFAULT)


async def get_api_key(api_key_header: str = Depends(api_key_header)):
    api_auth_key = _get_api_auth_key()
    if not api_auth_key or api_auth_key == "no-key":
        return
    if api_key_header != api_auth_key:
        LOGGER.warning("拒绝了无效的 API 密钥。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

models_instance: Optional[AIModels] = None
MAX_IMAGE_SIDE = 10000


def _mark_non_text_request_activity() -> None:
    if models_instance is None:
        return
    models_instance.mark_non_text_request_activity()


def _device_requests_gpu(device_name: str) -> bool:
    normalized = str(device_name or "").strip().upper()
    return normalized == "AUTO" or "GPU" in normalized


def _startup_self_check_dri() -> None:
    if os.name == "nt":
        return

    inference_device = os.environ.get("INFERENCE_DEVICE", "AUTO")
    clip_device = os.environ.get("CLIP_INFERENCE_DEVICE", inference_device)
    if not (_device_requests_gpu(inference_device) or _device_requests_gpu(clip_device)):
        LOGGER.info("启动自检：未请求 GPU 设备，跳过 /dev/dri 检查。")
        return

    dri_dir = "/dev/dri"
    if not os.path.isdir(dri_dir):
        raise RuntimeError(
            "启动自检失败：已请求 GPU 推理，但容器内不存在 /dev/dri。"
            "请映射 --device /dev/dri:/dev/dri 并设置正确的 video/render 组。"
        )

    try:
        dri_nodes = [
            os.path.join(dri_dir, name)
            for name in sorted(os.listdir(dri_dir))
            if name.startswith("card") or name.startswith("renderD")
        ]
    except Exception as exc:
        raise RuntimeError(f"启动自检失败：无法读取 {dri_dir}: {exc}") from exc

    if not dri_nodes:
        raise RuntimeError(
            "启动自检失败：/dev/dri 未发现 card*/renderD* 节点，无法执行 GPU 推理。"
        )

    denied_nodes = [node for node in dri_nodes if not os.access(node, os.R_OK | os.W_OK)]
    if denied_nodes:
        raise RuntimeError(
            "启动自检失败：/dev/dri 设备权限不足，请检查容器用户组映射。"
            f" 无权限节点: {', '.join(denied_nodes)}"
        )

    LOGGER.info(
        "启动自检通过：GPU 设备节点可访问。INFERENCE_DEVICE=%s CLIP_INFERENCE_DEVICE=%s",
        inference_device,
        clip_device,
    )


def _decode_first_gif_frame(contents: bytes) -> Tuple[Optional[np.ndarray], Optional[str]]:
    buffer = np.frombuffer(contents, np.uint8)
    errors: list[str] = []

    if hasattr(cv2, "imdecodeanimation"):
        try:
            ok, animation = cv2.imdecodeanimation(buffer)
        except Exception as exc:
            errors.append(f"cv2.imdecodeanimation failed: {exc}")
        else:
            frames = getattr(animation, "frames", None) if ok else None
            if frames:
                return np.asarray(frames[0]), None

    if hasattr(cv2, "imdecodemulti"):
        try:
            ok, frames = cv2.imdecodemulti(buffer, cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            errors.append(f"cv2.imdecodemulti failed: {exc}")
        else:
            if ok and frames:
                return np.asarray(frames[0]), None

    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        if errors:
            return None, "; ".join(errors)
        return None, "GIF first-frame decode failed"
    return decoded, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    _configure_application_logging()
    _startup_self_check_dri()
    LOGGER.info("应用启动：初始化主 AIModels 实例；/clip/txt 通过独立 Text-CLIP 服务处理，非文本模型按首次请求懒加载。")
    models_instance = AIModels()

    yield
    LOGGER.info("应用关闭：正在释放所有模型。")
    if models_instance:
        await asyncio.to_thread(models_instance.release_all_models)


app = FastAPI(
    title="MT-Photos AI 统一服务",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。\n https://github.com/Molyleaf/MT-Photos-AI-OpenVINO",
    version="2.2.0",
    lifespan=lifespan
)

async def read_image_from_upload(file: UploadFile) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    读取上传的图像文件，处理 GIF、16-bit 和其他格式。
    在失败时返回 (None, "错误消息")，而不是抛出 HTTPException。
    """
    contents = await file.read()
    img = None

    try:
        is_gif = file.content_type == "image/gif" or str(file.filename).lower().endswith(".gif")
        if is_gif:
            img, gif_err = await asyncio.to_thread(_decode_first_gif_frame, contents)
            if img is None and gif_err:
                LOGGER.info("GIF 首帧解码失败，将按普通静态图继续尝试: %s", gif_err)

        if img is None:
            nparr = np.frombuffer(contents, np.uint8)
            img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            LOGGER.info("文件 '%s' 无法被解码为图像。", file.filename)
            return None, f"文件 '{file.filename}' 无法被解码为图像。"

        if img.dtype == np.uint16:
            LOGGER.info("文件 '%s' 是 16-bit 图像，正在转换为 8-bit。", file.filename)
            img = (img / 256).astype(np.uint8)

        height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            LOGGER.info("文件 '%s' 尺寸超限: %sx%s", file.filename, width, height)
            return None, "height or width out of range"

        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if len(img.shape) < 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, None

    except Exception as e:
        LOGGER.error("读取图像 '%s' 时发生意外错误: %s", file.filename, e, exc_info=True)
        return None, f"处理图像时发生意外错误: {str(e)}"


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
<p>作者：https://github.com/Molyleaf/MT-Photos-AI-OpenVINO</p>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/check", response_model=CheckResponse, dependencies=[Depends(get_api_key)])
async def check_service():
    return {
        "result": "pass",
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
    }

@app.post("/restart", response_model=RestartResponse, dependencies=[Depends(get_api_key)])
async def restart_service():
    LOGGER.info("收到 /restart 请求，正在同步释放当前非文本模型。独立 Text-CLIP 服务保持可用。")
    if models_instance:
        await asyncio.to_thread(models_instance.release_models_for_restart)
    return {"result": "pass"}

@app.post("/restart_v2", response_model=RestartResponse, dependencies=[Depends(get_api_key)])
async def restart_process():
    LOGGER.info("收到 /restart_v2 请求，将重启整个服务进程。")
    def delayed_restart():
        import time
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, *sys.argv)
    threading.Thread(target=delayed_restart).start()
    return {"result": "pass"}


@app.post("/ocr", dependencies=[Depends(get_api_key)])
async def ocr_endpoint(file: UploadFile = File(...)):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    _mark_non_text_request_activity()
    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        ocr_results_obj = await models_instance.get_ocr_results_async(image)
        return {"result": ocr_results_obj.model_dump()}
    except Exception as e:
        LOGGER.error("处理 OCR 请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        return {"result": [], "msg": str(e)}

@app.post("/clip/img", dependencies=[Depends(get_api_key)])
async def clip_image_endpoint(file: UploadFile = File(...)):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    _mark_non_text_request_activity()
    LOGGER.debug("开始处理 CLIP 图像请求: %s", file.filename)

    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        embedding = await models_instance.get_image_embedding_async(image)
        result_strings = [f"{f:.16f}" for f in embedding]
        return {"result": result_strings}
    except Exception as e:
        LOGGER.error("处理 CLIP 请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        return {"result": [], "msg": str(e)}

@app.post("/clip/txt", dependencies=[Depends(get_api_key)])
async def clip_text_endpoint(request: TextClipRequest):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    try:
        embedding = await models_instance.get_text_embedding_async(request.text)
        result_strings = [f"{f:.16f}" for f in embedding]
        return {"result": result_strings}
    except Exception as e:
        LOGGER.error("处理 CLIP 文本请求失败: '%s...', 错误: %s", request.text[:50], e, exc_info=True)
        return {"result": [], "msg": str(e)}

@app.post("/represent", dependencies=[Depends(get_api_key)])
async def represent_endpoint(file: UploadFile = File(...)):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    _mark_non_text_request_activity()
    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        face_results_list = await models_instance.get_face_representation_async(image)
        results_dict = [r.model_dump() for r in face_results_list]
        return {
            "detector_backend": "insightface",
            "recognition_model": MODEL_NAME,
            "result": results_dict
        }
    except Exception as e:
        LOGGER.error("处理人脸识别请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        if 'set enforce_detection' in str(e) or 'Face could not be detected' in str(e):
            return {"result": []}

        return {"result": [], "msg": str(e)}

if __name__ == "__main__":
    import uvicorn
    _configure_standalone_logging()
    port = int(os.environ.get("PORT", 8060))
    log_level, _ = _resolve_log_level()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level=log_level.lower(),
        access_log=False
    )
