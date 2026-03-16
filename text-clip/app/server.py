import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from models.runtime import TextClipRuntime
from models.schemas import CheckResponse, TextClipRequest

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_TEXT_CLIP_ROOT = os.path.dirname(_APP_DIR)
_LOG_FILE = os.path.join(_TEXT_CLIP_ROOT, "server.log")
_LOG_NAMESPACE = "mt_photos_ai.text_clip"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_APP_LOG_HANDLER_FLAG = "_mt_photos_ai_text_clip_handler"
_KNOWN_LOGGER_NAMES = (
    _LOG_NAMESPACE,
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
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

LOGGER = logging.getLogger(_LOG_NAMESPACE)
API_AUTH_KEY_DEFAULT = "mt_photos_ai_extra"
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
runtime_instance: Optional[TextClipRuntime] = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime_instance
    _configure_application_logging()
    LOGGER.info("应用启动：初始化独立 Text-CLIP 服务，固定使用 CPU。")
    runtime_instance = TextClipRuntime()
    await asyncio.to_thread(runtime_instance.load)

    yield

    LOGGER.info("应用关闭：Text-CLIP 常驻模型将随进程退出统一回收。")


app = FastAPI(
    title="MT-Photos Text-CLIP 服务",
    description="一个独立部署的 Text-CLIP OpenVINO CPU 服务。",
    version="2.2.0-textclip",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos Text-CLIP Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos Text-CLIP 服务</p>
<p>服务状态： 运行中</p>
<p>推理设备： CPU</p>
<p>作者：https://github.com/Molyleaf/MT-Photos-AI-OpenVINO</p>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.post("/check", response_model=CheckResponse, dependencies=[Depends(get_api_key)])
async def check_service():
    return {
        "result": "pass",
        "title": "mt-photos-ai-text-clip",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
    }


@app.post("/clip/txt", dependencies=[Depends(get_api_key)])
async def clip_text_endpoint(request: TextClipRequest):
    if runtime_instance is None:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    try:
        embedding = await asyncio.to_thread(runtime_instance.get_text_embedding, request.text)
        result_strings = [f"{value:.16f}" for value in embedding]
        return {"result": result_strings}
    except Exception as exc:
        LOGGER.error("处理 Text-CLIP 请求失败: '%s...', 错误: %s", request.text[:50], exc, exc_info=True)
        return {"result": [], "msg": str(exc)}


if __name__ == "__main__":
    import uvicorn

    _configure_standalone_logging()
    port = int(os.environ.get("PORT", "8061"))
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
