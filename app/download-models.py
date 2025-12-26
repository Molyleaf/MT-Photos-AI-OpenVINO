# scripts/prepare_models.py
import os
import requests
import logging
import shutil
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 路径配置 ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_ROOT = PROJECT_ROOT / "models"

# --- 1. InsightFace 配置 (AntelopeV2) ---
# InsightFace 默认会自动下载，但为了离线部署，我们手动触发并移动
INSIGHTFACE_ROOT = MODELS_ROOT / "insightface"

# --- 2. QA-CLIP 配置 (ViT-L-14) ---
CLIP_ROOT = MODELS_ROOT / "qa-clip"
CLIP_MODEL_URL = "https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-large.pt"
CLIP_FILENAME = "QA-CLIP-large.pt"

# --- 3. RapidOCR 配置 (默认模型) ---
OCR_ROOT = MODELS_ROOT / "rapidocr"
OCR_URLS = {
    "ch_PP-OCRv4_det_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_ppocr_mobile_v2.0_cls_infer.onnx"
}

def download_file(url, dest_path):
    if dest_path.exists():
        logging.info(f"文件已存在，跳过: {dest_path.name}")
        return

    logging.info(f"正在下载: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def prepare_insightface():
    """下载 AntelopeV2 模型"""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        logging.error("请先安装 insightface: pip install insightface onnxruntime")
        return

    logging.info("--- 准备 InsightFace (antelopev2) ---")

    # 临时设置 HOME 目录，欺骗 insightface 下载到我们将要移动的地方
    # InsightFace 默认下载到 ~/.insightface/models/antelopev2
    # 我们这里通过实例化触发下载
    try:
        logging.info("正在触发 InsightFace 自动下载 (这可能需要一些时间)...")
        # 注意: providers 设为 CPU 只是为了下载，不需要 GPU 环境
        app = FaceAnalysis(name='antelopev2', root=str(INSIGHTFACE_ROOT), providers=['CPUExecutionProvider'])
        # 这一步会检查模型，如果不存在则下载
        # 模型会被下载到 INSIGHTFACE_ROOT/models/antelopev2
        logging.info(f"InsightFace 模型已准备在: {INSIGHTFACE_ROOT}")
    except Exception as e:
        logging.error(f"InsightFace 下载失败: {e}")

def prepare_clip():
    """下载 QA-CLIP ViT-L-14"""
    logging.info("--- 准备 QA-CLIP (ViT-L-14) ---")
    dest_path = CLIP_ROOT / CLIP_FILENAME
    download_file(CLIP_MODEL_URL, dest_path)

def prepare_rapidocr():
    """下载 RapidOCR 模型"""
    logging.info("--- 准备 RapidOCR ---")
    for filename, url in OCR_URLS.items():
        dest_path = OCR_ROOT / filename
        download_file(url, dest_path)

    # 创建 config.yaml 指向本地模型 (可选，如果 RapidOCR 库能自动发现则不需要)
    # 通常 RapidOCR 指定 path 即可

if __name__ == "__main__":
    prepare_clip()
    prepare_rapidocr()
    prepare_insightface()

    logging.info("\n✅ 所有模型准备完成！")
    logging.info(f"请将 '{MODELS_ROOT}' 文件夹完整复制到目标机器的对应位置。")
    logging.info("并在 .env 文件中设置:")
    logging.info("MODEL_NAME=antelopev2")