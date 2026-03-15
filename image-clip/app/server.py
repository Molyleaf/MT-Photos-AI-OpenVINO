import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from models.runtime import ImageClipRuntime
from models.schemas import CheckResponse

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_CLIP_ROOT = os.path.dirname(_APP_DIR)
_LOG_FILE = os.path.join(_IMAGE_CLIP_ROOT, "server.log")
_LOG_NAMESPACE = "mt_photos_ai.image_clip"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_APP_LOG_HANDLER_FLAG = "_mt_photos_ai_image_clip_handler"
_KNOWN_LOGGER_NAMES = (
    _LOG_NAMESPACE,
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "transformers",
)

API_AUTH_KEY_DEFAULT = "mt_photos_ai_extra"
API_KEY_NAME = "api-key"
MAX_IMAGE_SIDE = 10000
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
runtime_instance: Optional[ImageClipRuntime] = None


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
LOGGER = logging.getLogger(_LOG_NAMESPACE)


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


async def read_image_from_upload(file: UploadFile) -> Tuple[Optional[np.ndarray], Optional[str]]:
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
    except Exception as exc:
        LOGGER.error("读取图像 '%s' 时发生意外错误: %s", file.filename, exc, exc_info=True)
        return None, f"处理图像时发生意外错误: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global runtime_instance
    _configure_application_logging()
    LOGGER.info("应用启动：初始化独立 Image-CLIP CUDA 服务。")
    runtime_instance = ImageClipRuntime()
    await asyncio.to_thread(runtime_instance.load)

    yield

    LOGGER.info("应用关闭：正在释放 Image-CLIP 模型。")
    if runtime_instance is not None:
        await asyncio.to_thread(runtime_instance.release)


app = FastAPI(
    title="MT-Photos Image-CLIP 服务",
    description="一个独立部署的 QA-CLIP Image-CLIP CUDA 服务。",
    version="2.2.0-imageclip",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def top_info():
    device_info = runtime_instance.runtime_device_label if runtime_instance is not None else "loading"
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos Image-CLIP Server</title>
    <style>p{{text-align: center;}}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos Image-CLIP 服务</p>
<p>服务状态： 运行中</p>
<p>推理设备： {device_info}</p>
<p>作者：https://github.com/Molyleaf/MT-Photos-AI-OpenVINO</p>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.post("/check", response_model=CheckResponse, dependencies=[Depends(get_api_key)])
async def check_service():
    return {
        "result": "pass",
        "title": "mt-photos-ai-image-clip",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
    }


@app.post("/clip/img", dependencies=[Depends(get_api_key)])
async def clip_image_endpoint(file: UploadFile = File(...)):
    if runtime_instance is None:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        embedding = await runtime_instance.get_image_embedding_async(image)
        result_strings = [f"{value:.16f}" for value in embedding]
        return {"result": result_strings}
    except Exception as exc:
        LOGGER.error("处理 Image-CLIP 请求失败: %s, 错误: %s", file.filename, exc, exc_info=True)
        return {"result": [], "msg": str(exc)}


if __name__ == "__main__":
    import uvicorn

    _configure_standalone_logging()
    port = int(os.environ.get("PORT", "8062"))
    log_level, _ = _resolve_log_level()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level=log_level.lower(),
        access_log=False,
    )
