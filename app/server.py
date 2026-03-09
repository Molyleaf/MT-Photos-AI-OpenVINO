# app/server.py
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

import models as ai_models
from schemas import (
    CheckResponse,
    TextClipRequest,
    # RepresentResponse, # 不再用于 /represent 的响应模型
    RestartResponse
)

# --- NSSM 日志优化和级别设置 ---
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_LOG_FILE = os.path.join(_PROJECT_ROOT, "server.log")

LOG_LEVEL_NAME = os.environ.get("LOG_LEVEL", "WARNING").upper() #
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.WARNING)

log_handlers = [logging.StreamHandler()] # 默认输出到控制台
if sys.platform == "win32":
    # 在 Windows (NSSM) 上，额外记录到文件
    try:
        log_handlers = [
            logging.FileHandler(_LOG_FILE, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
        print(f"服务日志将被写入: {_LOG_FILE}")
    except Exception as e:
        print(f"无法设置文件日志: {e}")

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

# --- 【修复日志：强制关闭 Uvicorn 200 访问日志】 ---
# 必须在 basicConfig 之后调用，以覆盖 uvicorn.access 的默认设置
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# --- 日志配置结束 ---


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
        logging.warning(f"拒绝了无效的 API 密钥: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

SERVER_IDLE_TIMEOUT = int(os.environ.get("SERVER_IDLE_TIMEOUT", "300"))
TEXT_MODEL_RESTORE_DELAY_MS = ai_models.TEXT_MODEL_RESTORE_DELAY_MS
idle_timer: Optional[threading.Timer] = None
models_instance: Optional[ai_models.AIModels] = None
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
MAX_IMAGE_SIDE = 10000

def idle_timeout_handler():
    global models_instance
    if models_instance:
        logging.warning(f"服务器已空闲 {SERVER_IDLE_TIMEOUT} 秒。正在释放 (按需) 模型以节省内存...")
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


def _device_requests_gpu(device_name: str) -> bool:
    normalized = str(device_name or "").strip().upper()
    return normalized == "AUTO" or "GPU" in normalized


def _startup_self_check_dri() -> None:
    if os.name == "nt":
        return

    inference_device = os.environ.get("INFERENCE_DEVICE", "AUTO")
    clip_device = os.environ.get("CLIP_INFERENCE_DEVICE", inference_device)
    if not (_device_requests_gpu(inference_device) or _device_requests_gpu(clip_device)):
        logging.warning("启动自检：未请求 GPU 设备，跳过 /dev/dri 检查。")
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

    logging.warning(
        "启动自检通过：GPU 设备节点可访问。INFERENCE_DEVICE=%s CLIP_INFERENCE_DEVICE=%s",
        inference_device,
        clip_device,
    )


def _probe_image_size(
    contents: bytes,
) -> Tuple[Optional[int], Optional[int], Optional[bool], Optional[str]]:
    probe_cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,bits_per_raw_sample",
        "-of",
        "json",
        "-",
    ]
    try:
        proc = subprocess.run(
            probe_cmd,
            input=contents,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        return None, None, None, f"ffprobe not found: {FFPROBE_BIN}"
    except subprocess.TimeoutExpired:
        return None, None, None, "ffprobe timeout"
    except Exception as exc:
        return None, None, None, f"ffprobe error: {exc}"

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        return None, None, None, f"ffprobe failed: {stderr or 'unknown error'}"

    raw_stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
    try:
        payload = json.loads(raw_stdout) if raw_stdout else {}
        streams = payload.get("streams", [])
        if not streams:
            return None, None, None, f"ffprobe output parse failed: {raw_stdout}"
        stream = streams[0] or {}
        width = int(stream.get("width"))
        height = int(stream.get("height"))
        bits_per_raw_sample = stream.get("bits_per_raw_sample")
        pix_fmt = str(stream.get("pix_fmt") or "").lower()
        bit_depth = 0
        if bits_per_raw_sample not in (None, ""):
            try:
                bit_depth = int(bits_per_raw_sample)
            except (TypeError, ValueError):
                bit_depth = 0
        if bit_depth <= 0 and pix_fmt:
            # 常见高位深像素格式：rgb48*/gray16*/yuv420p10* 等。
            if any(token in pix_fmt for token in ("10", "12", "14", "16", "32", "48", "64")):
                bit_depth = 16
        high_bit_depth = bit_depth > 8
    except Exception:
        return None, None, None, f"ffprobe output parse failed: {raw_stdout}"

    if width <= 0 or height <= 0:
        return None, None, None, f"ffprobe invalid image size: {raw_stdout}"
    return width, height, high_bit_depth, None


def _decode_image_with_ffmpeg(
    contents: bytes,
    prefer_qsv: bool,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    decode_cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if prefer_qsv:
        decode_cmd.extend(["-hwaccel", "qsv"])
    decode_cmd.extend(["-i", "pipe:0", "-frames:v", "1"])
    use_rawvideo = width is not None and height is not None and width > 0 and height > 0
    if use_rawvideo:
        decode_cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    else:
        # Width/height unknown: decode one frame to BMP bytes to avoid guessing raw frame shape.
        decode_cmd.extend(["-f", "image2pipe", "-vcodec", "bmp", "pipe:1"])

    try:
        proc = subprocess.run(
            decode_cmd,
            input=contents,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
    except FileNotFoundError:
        return None, f"ffmpeg not found: {FFMPEG_BIN}"
    except subprocess.TimeoutExpired:
        return None, "ffmpeg decode timeout"
    except Exception as exc:
        return None, f"ffmpeg decode error: {exc}"

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        return None, f"ffmpeg decode failed: {stderr or 'unknown error'}"

    if use_rawvideo:
        expected_size = int(width) * int(height) * 3
        if len(proc.stdout) != expected_size:
            return None, (
                "ffmpeg decode size mismatch: "
                f"expected={expected_size}, got={len(proc.stdout)}"
            )
        decoded = np.frombuffer(proc.stdout, dtype=np.uint8).reshape((int(height), int(width), 3))
        return decoded, None

    frame_buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    decoded = cv2.imdecode(frame_buf, cv2.IMREAD_COLOR)
    if decoded is None:
        return None, "ffmpeg image2pipe decode failed"
    return decoded, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_instance
    _startup_self_check_dri()
    logging.warning("应用启动... 初始化 AIModels 实例。")
    models_instance = ai_models.AIModels()
    try:
        logging.warning("应用启动... 正在预热加载 Text CLIP 模型。")
        await models_instance.ensure_clip_text_model_loaded_async()
    except Exception as e:
        logging.critical(f"应用启动时基础模型加载失败: {e}", exc_info=True)

    reset_idle_timer()
    yield
    global idle_timer
    if idle_timer:
        idle_timer.cancel()
    logging.warning("应用关闭。正在释放所有模型...")
    if models_instance:
        if hasattr(models_instance, 'release_all_models') and callable(models_instance.release_all_models):
            models_instance.release_all_models()
        else:
            logging.warning("AIModels 实例没有 release_all_models 方法 (lifespan)。")

app = FastAPI(
    title="MT-Photos AI 统一服务",
    description="一个基于 OpenVINO 加速的、用于照片分析的高性能统一AI服务 (支持自动内存释放)。\n https://github.com/Molyleaf/MT-Photos-AI-OpenVINO",
    version="2.2.0",
    lifespan=lifespan
)

@app.middleware("http")
async def reset_timer_middleware(request: Request, call_next):
    if request.url.path not in ["/check", "/", "/clip/txt"]:
        reset_idle_timer()
    response = await call_next(request)
    return response

# --- 【修复：重写 read_image_from_upload 以匹配示例行为】 ---
async def read_image_from_upload(file: UploadFile) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    读取上传的图像文件，处理 GIF、16-bit 和其他格式。
    在失败时返回 (None, "错误消息")，而不是抛出 HTTPException。
    """
    contents = await file.read()
    img = None

    try:
        # 尝试使用 PIL 打开（主要为了处理 GIF）
        is_gif = file.content_type == "image/gif" or str(file.filename).lower().endswith(".gif")
        if is_gif:
            with Image.open(BytesIO(contents)) as pil_img:
                if pil_img.is_animated:
                    pil_img.seek(0)  # 移动到第一帧
                frame = pil_img.convert("RGB")
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR) # 转为 CV2 BGR

        # 非 GIF 优先尝试 ffmpeg(QSV) 解码，再显式降级 ffmpeg(CPU) 和 OpenCV。
        if img is None:
            width_probe, height_probe, high_bit_depth, probe_err = await asyncio.to_thread(
                _probe_image_size, contents
            )
            if probe_err:
                logging.warning(
                    "ffprobe 探测失败，将继续尝试 ffmpeg(QSV/CPU) 再回退 cv2.imdecode: %s",
                    probe_err,
                )
            elif width_probe is None or height_probe is None:
                logging.warning("ffprobe 未返回有效尺寸，将继续尝试 ffmpeg(QSV/CPU)。")
            elif high_bit_depth:
                logging.warning("检测到高位深图像，改用 cv2.imdecode 保持 16-bit->8-bit 语义。")
            elif width_probe is not None and height_probe is not None and (
                width_probe > MAX_IMAGE_SIDE or height_probe > MAX_IMAGE_SIDE
            ):
                logging.warning(f"文件 '{file.filename}' 尺寸超限（ffmpeg probe）")
                return None, "height or width out of range"

            if not high_bit_depth:
                ffmpeg_qsv_img, ffmpeg_qsv_err = await asyncio.to_thread(
                    _decode_image_with_ffmpeg, contents, True, width_probe, height_probe
                )
                if ffmpeg_qsv_img is not None:
                    img = ffmpeg_qsv_img
                else:
                    logging.warning("ffmpeg(QSV) 解码失败，将尝试 ffmpeg(CPU): %s", ffmpeg_qsv_err)
                    ffmpeg_cpu_img, ffmpeg_cpu_err = await asyncio.to_thread(
                        _decode_image_with_ffmpeg, contents, False, width_probe, height_probe
                    )
                    if ffmpeg_cpu_img is not None:
                        img = ffmpeg_cpu_img
                    else:
                        logging.warning(
                            "ffmpeg(CPU) 解码失败，将尝试 cv2.imdecode: %s",
                            ffmpeg_cpu_err,
                        )

        # ffmpeg 路径失败后，回退到 cv2.imdecode。
        if img is None:
            nparr = np.frombuffer(contents, np.uint8)
            img = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_UNCHANGED)

        # 检查解码是否成功
        if img is None:
            logging.warning(f"文件 '{file.filename}' 无法被解码为图像。")
            return None, f"文件 '{file.filename}' 无法被解码为图像。"

        # 检查 16-bit 图像并转换
        if img.dtype == np.uint16:
            logging.warning(f"文件 '{file.filename}' 是 16-bit 图像，正在转换为 8-bit。")
            img = (img / 256).astype(np.uint8)

        # 检查尺寸
        height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
        if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
            logging.warning(f"文件 '{file.filename}' 尺寸超限: {width}x{height}")
            return None, "height or width out of range"

        # 转换通道
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 最后的健壮性检查
        if len(img.shape) < 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, None

    except Exception as e:
        logging.error(f"读取图像 '{file.filename}' 时发生意外错误: {e}", exc_info=True)
        return None, f"处理图像时发生意外错误: {str(e)}"
# --- 修复结束 ---

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

# --- 【修复：合并 /check 响应】 ---
@app.post("/check", response_model=CheckResponse, dependencies=[Depends(get_api_key)])
async def check_service(t: str = ""):
    return {
        "result": "pass",
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
    }
# --- 修复结束 ---

@app.post("/restart", response_model=RestartResponse, dependencies=[Depends(get_api_key)])
async def restart_service():
    logging.warning(
        "收到 /restart 请求，正在释放模型。Text-CLIP 将在无其它任务 %sms 后恢复。",
        TEXT_MODEL_RESTORE_DELAY_MS,
    )
    if models_instance:
        if (
            hasattr(models_instance, "release_models_for_restart")
            and callable(models_instance.release_models_for_restart)
        ):
            models_instance.release_models_for_restart()
        elif hasattr(models_instance, "release_models") and callable(models_instance.release_models):
            models_instance.release_models()
        else:
            logging.error("'AIModels' object has no attribute 'release_models' during restart request!")
    return {"result": "pass"}

@app.post("/restart_v2", response_model=RestartResponse, dependencies=[Depends(get_api_key)])
async def restart_process():
    logging.warning("收到 /restart_v2 请求，将重启整个服务进程！")
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

    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        ocr_results_obj = await models_instance.get_ocr_results_async(image)
        # --- 【修复检查点1：成功时不返回 msg】 ---
        return {"result": ocr_results_obj.model_dump()}
    except Exception as e:
        logging.error(f"处理 OCR 请求失败: {file.filename}, 错误: {e}", exc_info=True)
        return {"result": [], "msg": str(e)}

@app.post("/clip/img", dependencies=[Depends(get_api_key)])
async def clip_image_endpoint(file: UploadFile = File(...)):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    logging.debug(f"开始处理 CLIP 图像请求: {file.filename}")

    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        embedding = await models_instance.get_image_embedding_async(image, file.filename)
        result_strings = [f"{f:.16f}" for f in embedding]
        # --- 【修复检查点1：成功时不返回 msg】 ---
        return {"result": result_strings}
    except Exception as e:
        logging.error(f"处理 CLIP 请求失败: {file.filename}, 错误: {e}", exc_info=True)
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
        logging.error(f"处理 CLIP 文本请求失败: '{request.text[:50]}...', 错误: {e}", exc_info=True)
        return {"result": [], "msg": str(e)}

@app.post("/represent", dependencies=[Depends(get_api_key)])
async def represent_endpoint(file: UploadFile = File(...)):
    if not models_instance:
        raise HTTPException(status_code=503, detail="模型实例尚未初始化")

    image, error_msg = await read_image_from_upload(file)
    if image is None:
        return {"result": [], "msg": error_msg}

    try:
        face_results_list = await models_instance.get_face_representation_async(image)
        results_dict = [r.model_dump() for r in face_results_list]
        return {
            "detector_backend": "insightface",
            "recognition_model": ai_models.MODEL_NAME,
            "result": results_dict
        }
    except Exception as e:
        logging.error(f"处理人脸识别请求失败: {file.filename}, 错误: {e}", exc_info=True)
        if 'set enforce_detection' in str(e) or 'Face could not be detected' in str(e):
            return {"result": []}

        return {"result": [], "msg": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8060))
    log_level = os.environ.get("LOG_LEVEL", "warning").lower()
    workers = int(os.environ.get("WEB_CONCURRENCY", 2))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=log_level,
        workers=workers,
        access_log=False
    )
