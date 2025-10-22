# scripts/convert_models_fixed.py

import argparse
from pathlib import Path

import openvino as ov
import torch
import torch.nn as nn
from transformers import AltCLIPModel, AltCLIPProcessor


class VisionModelWithProjection(nn.Module):
    """
    一个封装了视觉骨干网络 (vision_model) 和视觉投影层 (visual_projection) 的模块。
    这确保了导出的模型是端到端的，可以输出最终的 768 维特征向量。
    """
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        # 从视觉骨干网络获取输出
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # 获取池化后的输出（通常是 [CLS] 标记的表示）
        # 这是 1280 维的内部向量
        pooled_output = vision_outputs.pooler_output
        # 应用最终的投影层，将其转换为 768 维
        image_embeds = self.visual_projection(pooled_output)
        return image_embeds


def convert_alt_clip_to_openvino(output_dir: Path):
    """
    下载 Alt-CLIP 模型，并将其转换为 ONNX，最终转换为 OpenVINO IR 格式。
    此版本修复了视觉模型转换不完整的问题。
    """
    model_name = "BAAI/AltCLIP-m18"
    print(f"正在加载模型: {model_name}")
    processor = AltCLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = AltCLIPModel.from_pretrained(model_name)
    model.eval()  # 设置为评估模式

    # --- 目录设置 ---
    onnx_dir = output_dir / "onnx"
    openvino_dir = output_dir / "openvino"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    openvino_dir.mkdir(parents=True, exist_ok=True)

    # --- 视觉模型转换 (已修复) ---
    print("正在转换视觉模型...")
    vision_onnx_path = onnx_dir / "clip_vision.onnx"
    vision_ov_path = openvino_dir / "clip_vision.xml"

    # 创建包含投影层的完整视觉模型实例
    full_vision_model = VisionModelWithProjection(model.vision_model, model.visual_projection)
    full_vision_model.eval()

    # 视觉模型的伪输入数据
    dummy_vision_input = torch.randn(1, 3, 224, 224)

    # 将封装后的完整模型导出为 ONNX
    torch.onnx.export(
        full_vision_model,
        dummy_vision_input,
        str(vision_onnx_path),
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["image_embeds"], # 输出现在是最终的 embedding
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

    # --- 文本模型转换 (无变化) ---
    print("\n正在转换文本模型...")
    text_onnx_path = onnx_dir / "clip_text.onnx"
    text_ov_path = openvino_dir / "clip_text.xml"

    # 文本模型的伪输入数据
    dummy_text_input_ids = torch.randint(0, 1000, (1, 77), dtype=torch.long)
    dummy_text_attention_mask = torch.ones(1, 77, dtype=torch.long)

    # 将文本模型导出为 ONNX
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
    print(f"\nProcessor 配置文件已保存至: {openvino_dir}")

    print("\n模型转换完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Alt-CLIP 模型转换为 OpenVINO 格式。")
    # 让脚本自动确定输出目录，而不是依赖于硬编码的相对路径
    # 这使得脚本在任何位置运行都更加健壮
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "models" / "alt-clip"

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output,
        help=f"存放转换后模型的目录。默认为: {default_output}"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)

    convert_alt_clip_to_openvino(args.output_dir)

