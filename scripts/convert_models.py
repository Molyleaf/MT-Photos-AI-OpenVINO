# scripts/convert_models.py

import argparse
from pathlib import Path

import openvino as ov
import torch
from transformers import AltCLIPModel, AltCLIPProcessor


def convert_alt_clip_to_openvino(output_dir: Path):
    """
    下载 Alt-CLIP 模型，并将其转换为 ONNX，最终转换为 OpenVINO IR 格式。
    """
    model_name = "BAAI/AltCLIP-m18"
    print(f"正在加载模型: {model_name}")
    processor = AltCLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = AltCLIPModel.from_pretrained(model_name)
    model.eval() # 设置为评估模式

    # --- 目录设置 ---
    onnx_dir = output_dir / "onnx"
    openvino_dir = output_dir / "openvino"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    openvino_dir.mkdir(parents=True, exist_ok=True)

    # --- 视觉模型转换 ---
    print("正在转换视觉模型...")
    vision_onnx_path = onnx_dir / "clip_vision.onnx"
    vision_ov_path = openvino_dir / "clip_vision.xml"

    # 视觉模型的伪输入数据
    dummy_vision_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model.vision_model,
        dummy_vision_input,
        str(vision_onnx_path),
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
    )
    print(f"视觉模型已保存为 ONNX: {vision_onnx_path}")

    # 将视觉模型的 ONNX 格式转换为 OpenVINO IR 格式
    ov_vision_model = ov.convert_model(str(vision_onnx_path))
    ov.save_model(ov_vision_model, vision_ov_path, compress_to_fp16=True)
    print(f"视觉模型已保存为 OpenVINO IR: {vision_ov_path}")


    # --- 文本模型转换 ---
    print("正在转换文本模型...")
    text_onnx_path = onnx_dir / "clip_text.onnx"
    text_ov_path = openvino_dir / "clip_text.xml"

    # 文本模型的伪输入数据
    dummy_text_input_ids = torch.randint(0, 1000, (1, 77))
    dummy_text_attention_mask = torch.ones(1, 77, dtype=torch.long)

    torch.onnx.export(
        model.text_model,
        (dummy_text_input_ids, dummy_text_attention_mask),
        str(text_onnx_path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "text_embeds": {0: "batch_size"},
        },
    )
    print(f"文本模型已保存为 ONNX: {text_onnx_path}")

    # 将文本模型的 ONNX 格式转换为 OpenVINO IR 格式
    ov_text_model = ov.convert_model(str(text_onnx_path))
    ov.save_model(ov_text_model, text_ov_path, compress_to_fp16=True)
    print(f"文本模型已保存为 OpenVINO IR: {text_ov_path}")

    # --- 保存 processor 配置文件 ---
    processor.save_pretrained(openvino_dir)
    print(f"Processor 配置文件已保存至: {openvino_dir}")

    print("\n模型转换完成。")


if __name__ == "__main__":
    # --- 路径修改开始 ---
    # 获取项目根目录 (即 scripts 文件夹的上一级目录)
    # Path(__file__) -> 当前脚本的路径
    # .parent -> scripts 目录
    # .parent -> 项目根目录
    project_root = Path(__file__).parent.parent
    default_output_path = project_root / "models" / "alt-clip"
    # --- 路径修改结束 ---

    parser = argparse.ArgumentParser(description="将 Alt-CLIP 模型转换为 OpenVINO 格式。")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_path), # <-- 使用新的默认路径
        help="用于保存转换后模型的目录。",
    )
    args = parser.parse_args()

    # 确保路径是 Path 对象
    convert_alt_clip_to_openvino(Path(args.output_dir))