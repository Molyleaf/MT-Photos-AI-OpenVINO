# scripts/prepare_models.py
import os
import requests
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 路径配置 ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_ROOT = PROJECT_ROOT / "models"

# --- 1. InsightFace 配置 ---
INSIGHTFACE_ROOT = MODELS_ROOT / "insightface"

# --- 2. QA-CLIP 配置 (ViT-L-14) ---
CLIP_ROOT = MODELS_ROOT / "qa-clip"
CLIP_MODEL_URL = "https://huggingface.co/TencentARC/QA-CLIP/resolve/main/QA-CLIP-large.pt"
CLIP_FILENAME_PT = "QA-CLIP-large.pt"
CLIP_FILENAME_SAFE = "QA-CLIP-large.safetensors"

# --- 3. RapidOCR 配置 ---
OCR_ROOT = MODELS_ROOT / "rapidocr"
OCR_URLS = {
    "ch_PP-OCRv4_det_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx": "https://github.com/RapidAI/RapidOCR/releases/download/v1.3.0/ch_ppocr_mobile_v2.0_cls_infer.onnx"
}

def download_file(url, dest_path):
    if dest_path.exists():
        logging.info(f"文件已存在，跳过: {dest_path.name}")
        return True

    logging.info(f"正在下载: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
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
        return True
    except Exception as e:
        logging.error(f"下载失败: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def convert_pt_to_safetensors(pt_path: Path, safe_path: Path):
    """将 PyTorch .pt 模型转换为 .safetensors"""
    if safe_path.exists():
        logging.info(f"Safetensors 模型已存在，跳过转换: {safe_path.name}")
        return

    if not pt_path.exists():
        logging.error(f"找不到源文件用于转换: {pt_path}")
        return

    logging.info(f"正在将 .pt 转换为 .safetensors以提升加载速度...")
    try:
        # 1. 加载 PyTorch 权重
        # 【修改点】添加 weights_only=False 以解决 PyTorch 新版本的安全报错
        # 因为我们信任这个模型源 (TencentARC/QA-CLIP)
        model_state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        # 处理可能的嵌套
        if "state_dict" in model_state_dict:
            model_state_dict = model_state_dict["state_dict"]

        # 2. 保存为 Safetensors
        save_file(model_state_dict, safe_path)
        logging.info(f"转换成功: {safe_path}")

        # 3. 删除旧文件节省空间
        pt_path.unlink()
        logging.info("已删除原始 .pt 文件")

    except Exception as e:
        logging.error(f"转换失败: {e}", exc_info=True) # 打印详细堆栈以便排查

def prepare_insightface():
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        logging.error("请先安装 insightface: pip install insightface onnxruntime")
        return

    logging.info("--- 准备 InsightFace (antelopev2) ---")
    try:
        logging.info("正在触发 InsightFace 自动下载...")
        # providers 设为 CPU 只是为了下载
        app = FaceAnalysis(name='antelopev2', root=str(INSIGHTFACE_ROOT), providers=['CPUExecutionProvider'])
        logging.info(f"InsightFace 模型已准备在: {INSIGHTFACE_ROOT}")
    except Exception as e:
        logging.error(f"InsightFace 下载失败: {e}")

def prepare_clip():
    """下载并转换 QA-CLIP"""
    logging.info("--- 准备 QA-CLIP (Safetensors) ---")
    pt_path = CLIP_ROOT / CLIP_FILENAME_PT
    safe_path = CLIP_ROOT / CLIP_FILENAME_SAFE

    # 1. 如果 safetensors 已经存在，直接结束
    if safe_path.exists():
        logging.info(f"QA-CLIP Safetensors 已就绪: {safe_path}")
        return

    # 2. 下载 .pt (会检查是否已下载)
    if download_file(CLIP_MODEL_URL, pt_path):
        # 3. 转换
        convert_pt_to_safetensors(pt_path, safe_path)

def prepare_rapidocr():
    logging.info("--- 准备 RapidOCR ---")
    for filename, url in OCR_URLS.items():
        dest_path = OCR_ROOT / filename
        download_file(url, dest_path)

if __name__ == "__main__":
    prepare_clip()
    prepare_rapidocr()
    prepare_insightface()

    logging.info("\n✅ 所有模型准备完成！")