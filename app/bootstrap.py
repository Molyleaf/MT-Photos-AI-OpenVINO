import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
LOG_FILE = PROJECT_ROOT / "server.log"
LOG_NAMESPACE = "mt_photos_ai"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
APP_LOG_HANDLER_FLAG = "_mt_photos_ai_handler"
API_AUTH_KEY_DEFAULT = "mt_photos_ai_extra"
TEXT_CLIP_SERVER_URL_DEFAULT = "http://127.0.0.1:8061"
TEXT_CLIP_REQUEST_TIMEOUT_DEFAULT_SECONDS = 30.0
DEFAULT_SERVER_PORT = 8060
KNOWN_LOGGER_NAMES = (
    LOG_NAMESPACE,
    f"{LOG_NAMESPACE}.server",
    f"{LOG_NAMESPACE}.models",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "rapidocr",
    "rapidocr.utils.log",
)


@dataclass(frozen=True, slots=True)
class TextClipProxySettings:
    server_url: str
    api_key: str
    request_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AppServerSettings:
    api_auth_key: str
    log_level_name: str
    log_level_value: int
    port: int
    text_clip: TextClipProxySettings


def resolve_log_level() -> tuple[str, int]:
    configured_name = str(os.environ.get("LOG_LEVEL", "WARNING")).strip().upper() or "WARNING"
    resolved_level = getattr(logging, configured_name, logging.WARNING)
    resolved_name = logging.getLevelName(resolved_level)
    if not isinstance(resolved_name, str):
        resolved_name = "WARNING"
        resolved_level = logging.WARNING
    return resolved_name, int(resolved_level)


def load_server_settings() -> AppServerSettings:
    log_level_name, log_level_value = resolve_log_level()
    api_auth_key = os.environ.get("API_AUTH_KEY", API_AUTH_KEY_DEFAULT)

    text_clip_server_url = str(
        os.environ.get("TEXT_CLIP_SERVER_URL", TEXT_CLIP_SERVER_URL_DEFAULT)
    ).strip() or TEXT_CLIP_SERVER_URL_DEFAULT
    configured_text_clip_api_key = os.environ.get("TEXT_CLIP_API_KEY")
    if configured_text_clip_api_key is None:
        text_clip_api_key = api_auth_key
    else:
        normalized_text_clip_api_key = str(configured_text_clip_api_key).strip()
        text_clip_api_key = normalized_text_clip_api_key or api_auth_key

    raw_timeout = os.environ.get(
        "TEXT_CLIP_REQUEST_TIMEOUT",
        str(TEXT_CLIP_REQUEST_TIMEOUT_DEFAULT_SECONDS),
    )
    try:
        text_clip_timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        text_clip_timeout_seconds = TEXT_CLIP_REQUEST_TIMEOUT_DEFAULT_SECONDS

    raw_port = os.environ.get("PORT", str(DEFAULT_SERVER_PORT))
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = DEFAULT_SERVER_PORT

    return AppServerSettings(
        api_auth_key=api_auth_key,
        log_level_name=log_level_name,
        log_level_value=log_level_value,
        port=port,
        text_clip=TextClipProxySettings(
            server_url=text_clip_server_url,
            api_key=text_clip_api_key,
            request_timeout_seconds=max(1.0, text_clip_timeout_seconds),
        ),
    )


def _synchronize_known_logger_levels(log_level: int) -> None:
    for logger_name in KNOWN_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(log_level)


def configure_application_logging() -> None:
    settings = load_server_settings()
    namespace_logger = logging.getLogger(LOG_NAMESPACE)
    namespace_logger.setLevel(settings.log_level_value)
    namespace_logger.propagate = False
    _synchronize_known_logger_levels(settings.log_level_value)

    configured_handlers = [
        handler
        for handler in namespace_logger.handlers
        if getattr(handler, APP_LOG_HANDLER_FLAG, False)
    ]
    if configured_handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        for handler in configured_handlers:
            handler.setLevel(settings.log_level_value)
            handler.setFormatter(formatter)
        return

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(settings.log_level_value)
    console_handler.setFormatter(formatter)
    setattr(console_handler, APP_LOG_HANDLER_FLAG, True)
    namespace_logger.addHandler(console_handler)

    if sys.platform == "win32":
        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
        except Exception as exc:
            print(f"无法设置文件日志: {exc}")
        else:
            file_handler.setLevel(settings.log_level_value)
            file_handler.setFormatter(formatter)
            setattr(file_handler, APP_LOG_HANDLER_FLAG, True)
            namespace_logger.addHandler(file_handler)


def configure_standalone_logging() -> None:
    configure_application_logging()


def device_requests_gpu(device_name: str) -> bool:
    normalized = str(device_name or "").strip().upper()
    return normalized == "AUTO" or "GPU" in normalized


def startup_self_check_dri(logger: logging.Logger) -> None:
    if os.name == "nt":
        return

    inference_device = os.environ.get("INFERENCE_DEVICE", "AUTO")
    clip_device = os.environ.get("CLIP_INFERENCE_DEVICE", inference_device)
    if not (device_requests_gpu(inference_device) or device_requests_gpu(clip_device)):
        logger.info("启动自检：未请求 GPU 设备，跳过 /dev/dri 检查。")
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

    logger.info(
        "启动自检通过：GPU 设备节点可访问。INFERENCE_DEVICE=%s CLIP_INFERENCE_DEVICE=%s",
        inference_device,
        clip_device,
    )
