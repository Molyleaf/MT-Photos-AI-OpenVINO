# /app/convert.py
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# 导入 openvino，确保已安装
try:
    import openvino as ov
except ImportError:
    logging.error("OpenVINO 库未找到。请运行: pip install openvino openvino-dev")
    sys.exit(1)

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 对应 'chinese-clip-vit-large-patch14'
MODEL_ARCH = "ViT-L-14"
# ViT-L-14 的原生维度就是 768
NATIVE_DIMS = 768

# --- 动态路径定义 ---
# 获取此脚本所在的目录 (e.g., .../mt-photos-ai-openvino/scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取项目根目录 (e.g., .../mt-photos-ai-openvino)
PROJECT_ROOT = SCRIPT_DIR.parent

# --- 修正: 路径指向您所描述的 cn_clip 内部路径 ---
# (e.g., .../mt-photos-ai-openvino/scripts/cn_clip/deploy/pytorch_to_onnx.py)
ONNX_SCRIPT_PATH = SCRIPT_DIR / "cn_clip" / "deploy" / "pytorch_to_onnx.py"
# --- 结束修正 ---

# --- 新增: 定义 cn_clip 文件夹的路径 ---
CN_CLIP_DIR = SCRIPT_DIR / "cn_clip"
# --- 结束新增 ---

def run_command(cmd: list, env: dict = None):
    """
    辅助函数：运行 shell 命令并记录输出。
    --- 修正: 增加 env 参数 ---
    """
    # 将所有 Path 对象转换为字符串，以便 subprocess 可以处理
    str_cmd = [str(item) for item in cmd]
    logging.info(f"正在运行命令: {' '.join(str_cmd)}")

    # 合并当前环境变量和传入的自定义环境变量
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
        logging.info(f"设置环境变量: {env}")

    try:
        process = subprocess.run(
            str_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=merged_env # --- 修正: 传递环境变量 ---
        )
        logging.info(f"命令输出:\n{process.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"命令执行失败! 返回码: {e.returncode}")
        logging.error(f"错误输出:\n{e.stdout}")
        raise

def convert_models():
    """
    执行 Pytorch -> ONNX -> OpenVINO IR (FP16) 的完整转换流程。
    """

    # --- 修改: 根据您的需求设置路径 ---
    # 基础路径: ../models/chinese-clip/ (相对于 scripts 目录)
    models_base_dir = PROJECT_ROOT / "models" / "chinese-clip"

    ov_save_path = models_base_dir / "openvino"
    onnx_temp_dir = models_base_dir / "onnx"

    # 修正: ONNX 保存前缀必须是完整路径，否则会保存到当前工作目录
    onnx_save_prefix = onnx_temp_dir / "vit-l-14"

    # 在运行前创建所需目录
    ov_save_path.mkdir(parents=True, exist_ok=True)
    onnx_temp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"ONNX 模型将保存到: {onnx_temp_dir}")
    logging.info(f"OpenVINO 模型将保存到: {ov_save_path}")
    # --- 结束修改 ---

    logging.info(f"--- 步骤 1: 转换 Pytorch -> ONNX (FP16) ---")
    logging.info(f"模型架构: {MODEL_ARCH}")
    logging.info(f"ONNX 临时保存路径前缀: {onnx_save_prefix}")

    # --- 修改: 使用动态路径并检查脚本是否存在 ---
    if not ONNX_SCRIPT_PATH.exists():
        logging.error(f"找不到 ONNX 转换脚本: {ONNX_SCRIPT_PATH}")
        logging.error("请确保 'pytorch_to_onnx.py' 位于 'scripts/cn_clip/deploy/' 目录下。")
        sys.exit(1)

    if not CN_CLIP_DIR.exists():
        logging.error(f"找不到 'cn_clip' 目录: {CN_CLIP_DIR}")
        logging.error("请确保 'cn_clip' 文件夹完整位于 'scripts' 目录下。")
        sys.exit(1)

    cmd_onnx = [
        "python",
        ONNX_SCRIPT_PATH, # 使用动态路径
        "--model-arch", MODEL_ARCH,
        "--save-onnx-path", onnx_save_prefix, # 使用包含路径的前缀
        "--convert-text",
        "--convert-vision",
        "--download-root", PROJECT_ROOT / "cache" # 指定下载缓存位置
    ]

    # --- 新增: 设置 PYTHONPATH ---
    # 告诉子脚本去 'scripts/cn_clip' 目录寻找 'clip' 模块
    # 同时添加 'scripts' 目录，以便 'cn_clip.clip' 能被找到
    python_path = f"{str(CN_CLIP_DIR)}{os.pathsep}{str(SCRIPT_DIR)}"

    # 将现有的 PYTHONPATH 也包含进来
    existing_python_path = os.environ.get('PYTHONPATH', '')
    if existing_python_path:
        python_path = f"{python_path}{os.pathsep}{existing_python_path}"

    custom_env = {"PYTHONPATH": python_path}
    # --- 结束新增 ---

    try:
        # --- 修正: 传递自定义环境变量 ---
        run_command(cmd_onnx, env=custom_env)
        # --- 结束修正 ---
        logging.info("ONNX 模型转换成功。")
    except Exception as e:
        logging.error(f"ONNX 转换失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 步骤 2: 转换 ONNX -> OpenVINO IR (FP16) ---
    logging.info("--- 步骤 2: 转换 ONNX -> OpenVINO IR (FP16) ---")

    # --- 修正：使用 FP32 ONNX 模型作为输入 ---
    # 官方脚本输出的 FP32 ONNX 文件路径
    text_onnx_path = onnx_temp_dir / f"vit-l-14.txt.fp32.onnx"
    vision_onnx_path = onnx_temp_dir / f"vit-l-14.img.fp32.onnx"
    # --- 结束修正 ---

    # 最终 OpenVINO IR 的输出路径
    ov_text_path = ov_save_path / "openvino_text_fp16.xml"
    ov_vision_path = ov_save_path / "openvino_image_fp16.xml"

    if not text_onnx_path.exists() or not vision_onnx_path.exists():
        logging.error(f"未找到预期的 ONNX 文件: {text_onnx_path} / {vision_onnx_path}")
        logging.error("请检查步骤1的日志。确保 ONNX 转换成功且路径正确。")
        sys.exit(1)

    core = ov.Core()

    try:
        # 转换文本模型
        logging.info(f"正在转换文本模型: {text_onnx_path} -> {ov_text_path}")
        ov_text_model = ov.convert_model(text_onnx_path)
        # 我们在这里（保存时）进行 FP16 转换
        ov.save_model(ov_text_model, ov_text_path, compress_to_fp16=True)

        # 转换图像模型
        logging.info(f"正在转换图像模型: {vision_onnx_path} -> {ov_vision_path}")
        ov_vision_model = ov.convert_model(vision_onnx_path)
        # 我们在这里（保存时）进行 FP16 转换
        ov.save_model(ov_vision_model, ov_vision_path, compress_to_fp16=True)

        logging.info("OpenVINO IR 模型已成功保存。")

    except Exception as e:
        logging.error(f"OpenVINO 转换失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 步骤 3: 验证转换后的 OpenVINO 模型 ---
    logging.info("--- 步骤 3: 验证 OpenVINO IR 模型 ---")
    try:
        vision_model = core.read_model(ov_vision_path)
        vision_output = vision_model.output(0)
        vision_dims = vision_output.get_partial_shape()[1].get_length()
        if vision_dims == NATIVE_DIMS:
            logging.info(f"✅ 视觉模型维度验证成功: {vision_dims}d")
        else:
            raise RuntimeError(f"视觉模型维度错误! 预期: {NATIVE_DIMS}, 得到: {vision_dims}")

        text_model = core.read_model(ov_text_path)
        text_output = text_model.output(0)
        text_dims = text_output.get_partial_shape()[1].get_length()
        text_inputs_count = len(text_model.inputs)

        # 官方 Chinese-CLIP ONNX 文本模型只有 1 个输入 (input_ids)
        if text_inputs_count != 1:
            raise RuntimeError(f"文本模型输入数量错误! 预期: 1, 得到: {text_inputs_count}")
        logging.info(f"✅ 文本模型输入数量验证成功: {text_inputs_count}")

        if text_dims == NATIVE_DIMS:
            logging.info(f"✅ 文本模型维度验证成功: {text_dims}d")
        else:
            raise RuntimeError(f"文本模型维度错误! 预期: {NATIVE_DIMS}, 得到: {text_dims}")

    except Exception as e:
        logging.error(f"模型验证失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 步骤 4: 清理 ---
    logging.info("--- 步骤 4: 清理临时 ONNX 文件 ---")
    try:
        for f in onnx_temp_dir.glob("vit-l-14*"):
            f.unlink()
            logging.info(f"已删除: {f}")
        onnx_temp_dir.rmdir()
        logging.info("清理完成。")
    except Exception as e:
        logging.warning(f"清理临时文件时出错: {e}", exc_info=True)

    logging.info(f"🎉 全部转换和验证成功完成。模型保存在: {ov_save_path}")

if __name__ == "__main__":
    # --- 修改: 移除了 argparse ---
    # 脚本现在使用相对于自身的固定路径结构，不再需要外部参数
    convert_models()
    # --- 结束修改 ---