import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from bootstrap import (
    LOG_NAMESPACE,
    configure_application_logging,
    configure_standalone_logging,
    load_server_settings,
    startup_self_check_dri,
)
from image_io import read_image_from_upload
from models.constants import MODEL_NAME
from models.runtime import AIModels
from models.schemas import (
    CheckResponse,
    RestartResponse,
)
from text_clip_proxy import TextClipProxyClient

configure_application_logging()

LOGGER = logging.getLogger(f"{LOG_NAMESPACE}.server")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

STATUS_PAGE_HTML = """<!DOCTYPE html>
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


def _build_text_clip_proxy_client() -> TextClipProxyClient:
    return TextClipProxyClient(
        load_server_settings().text_clip,
        api_key_header_name=API_KEY_NAME,
    )


async def get_api_key(api_key_header: str = Depends(api_key_header)):
    api_auth_key = load_server_settings().api_auth_key
    if not api_auth_key or api_auth_key == "no-key":
        return
    if api_key_header != api_auth_key:
        LOGGER.warning("拒绝了无效的 API 密钥。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


models_instance: Optional[AIModels] = None


def _require_models_instance() -> AIModels:
    if models_instance is None:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")
    return models_instance


def _mark_request_activity(models: AIModels) -> None:
    models.mark_request_activity()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    configure_application_logging()
    startup_self_check_dri(LOGGER)
    settings = load_server_settings()
    LOGGER.info(
        "应用启动：初始化主 AIModels 实例；非文本模型按首次请求懒加载。Text-CLIP 代理上游=%s",
        settings.text_clip.server_url,
    )
    instance = AIModels()
    models_instance = instance
    try:
        yield
    finally:
        LOGGER.info("应用关闭：正在释放所有模型。")
        instance_to_release = models_instance
        models_instance = None
        if instance_to_release is not None:
            await asyncio.to_thread(instance_to_release.release_all_models)


app = FastAPI(
    title="MT-Photos AI 统一服务",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务。\n https://github.com/Molyleaf/MT-Photos-AI-OpenVINO",
    version="2.2.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def top_info():
    return HTMLResponse(content=STATUS_PAGE_HTML)


@app.post("/check", response_model=CheckResponse, dependencies=[Depends(get_api_key)])
async def check_service():
    return {
        "result": "pass",
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
    }


@app.post("/restart", response_model=RestartResponse, dependencies=[Depends(get_api_key)])
async def restart_service():
    LOGGER.info("收到 /restart 请求，正在同步释放当前非文本模型。")
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

    threading.Thread(target=delayed_restart, name="restart-v2", daemon=True).start()
    return {"result": "pass"}


@app.post("/clip/txt", dependencies=[Depends(get_api_key)])
async def clip_text_proxy_endpoint(request: Request):
    request_body = await request.body()
    content_type = request.headers.get("content-type")
    try:
        proxy_client = _build_text_clip_proxy_client()
        return await asyncio.to_thread(
            proxy_client.forward_request,
            request_body,
            content_type=content_type,
        )
    except Exception as exc:
        LOGGER.error(
            "转发 Text-CLIP 请求失败: bytes=%s content_type=%s 错误: %s",
            len(request_body),
            content_type or "application/json; charset=utf-8",
            exc,
            exc_info=True,
        )
        return {"result": [], "msg": str(exc)}


@app.post("/ocr", dependencies=[Depends(get_api_key)])
async def ocr_endpoint(file: UploadFile = File(...)):
    models = _require_models_instance()

    _mark_request_activity(models)
    image, error_msg = await read_image_from_upload(file, logger=LOGGER)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        ocr_results_obj = await models.get_ocr_results_async(image)
        return {"result": ocr_results_obj.model_dump()}
    except Exception as e:
        LOGGER.error("处理 OCR 请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        return {"result": [], "msg": str(e)}


@app.post("/clip/img", dependencies=[Depends(get_api_key)])
async def clip_image_endpoint(file: UploadFile = File(...)):
    models = _require_models_instance()

    _mark_request_activity(models)
    LOGGER.debug("开始处理 CLIP 图像请求: %s", file.filename)

    image, error_msg = await read_image_from_upload(file, logger=LOGGER)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        embedding = await models.get_image_embedding_async(image)
        result_strings = [f"{f:.16f}" for f in embedding]
        return {"result": result_strings}
    except Exception as e:
        LOGGER.error("处理 CLIP 请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        return {"result": [], "msg": str(e)}


@app.post("/represent", dependencies=[Depends(get_api_key)])
async def represent_endpoint(file: UploadFile = File(...)):
    models = _require_models_instance()

    _mark_request_activity(models)
    image, error_msg = await read_image_from_upload(file, logger=LOGGER)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        face_results_list = await models.get_face_representation_async(image)
        results_dict = [r.model_dump() for r in face_results_list]
        return {
            "detector_backend": "insightface",
            "recognition_model": MODEL_NAME,
            "result": results_dict,
        }
    except Exception as e:
        LOGGER.error("处理人脸识别请求失败: %s, 错误: %s", file.filename, e, exc_info=True)
        if "set enforce_detection" in str(e) or "Face could not be detected" in str(e):
            return {"result": []}

        return {"result": [], "msg": str(e)}


if __name__ == "__main__":
    import uvicorn

    configure_standalone_logging()
    settings = load_server_settings()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
        workers=1,
        log_level=settings.log_level_name.lower(),
        access_log=False,
    )
