import asyncio
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

MAX_IMAGE_SIDE = 10000


def decode_first_gif_frame(contents: bytes) -> tuple[Optional[np.ndarray], Optional[str]]:
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


def normalize_uploaded_image(
    image: np.ndarray,
    *,
    filename: str,
    logger: logging.Logger,
    max_image_side: int = MAX_IMAGE_SIDE,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if image.dtype == np.uint16:
        logger.info("文件 '%s' 是 16-bit 图像，正在转换为 8-bit。", filename)
        image = (image / 256).astype(np.uint8)

    if len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    if width > max_image_side or height > max_image_side:
        logger.info("文件 '%s' 尺寸超限: %sx%s", filename, width, height)
        return None, "height or width out of range"

    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if len(image.shape) < 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image, None


async def read_image_from_upload(
    file: UploadFile,
    *,
    logger: logging.Logger,
    max_image_side: int = MAX_IMAGE_SIDE,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    读取上传图像，按既有契约返回 (image, error_msg)。
    """
    contents = await file.read()
    image = None

    try:
        is_gif = file.content_type == "image/gif" or str(file.filename).lower().endswith(".gif")
        if is_gif:
            image, gif_err = await asyncio.to_thread(decode_first_gif_frame, contents)
            if image is None and gif_err:
                logger.info("GIF 首帧解码失败，将按普通静态图继续尝试: %s", gif_err)

        if image is None:
            buffer = np.frombuffer(contents, np.uint8)
            image = await asyncio.to_thread(cv2.imdecode, buffer, cv2.IMREAD_UNCHANGED)

        if image is None:
            logger.info("文件 '%s' 无法被解码为图像。", file.filename)
            return None, f"文件 '{file.filename}' 无法被解码为图像。"

        return normalize_uploaded_image(
            image,
            filename=str(file.filename),
            logger=logger,
            max_image_side=max_image_side,
        )
    except Exception as exc:
        logger.error("读取图像 '%s' 时发生意外错误: %s", file.filename, exc, exc_info=True)
        return None, f"处理图像时发生意外错误: {str(exc)}"
