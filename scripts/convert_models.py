import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import openvino as ov
from pathlib import Path
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Alt-CLIP-Converter")

MODEL_NAME = "BAAI/alt-clip-m18"
DEFAULT_OUTPUT_DIR = Path("./models/alt-clip/openvino")

def convert_alt_clip_to_openvino(output_dir: Path):
    """
    Downloads the Alt-CLIP model, converts its vision and text encoders
    to ONNX, and then to OpenVINO IR format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Ensuring output directory exists: {output_dir}")

    # 1. 加载 PyTorch 模型、Tokenizer 和 Processor
    log.info(f"Loading PyTorch model '{MODEL_NAME}' from Hugging Face...")
    pytorch_model = CLIPModel.from_pretrained(MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    log.info("Model, Tokenizer, and Processor loaded successfully.")

    # 定义 ONNX 和 OpenVINO 模型路径
    onnx_vision_path = output_dir / "vision.onnx"
    onnx_text_path = output_dir / "text.onnx"
    ir_vision_path = output_dir / "clip_vision.xml"
    ir_text_path = output_dir / "clip_text.xml"

    # 2. 导出视觉模型 (Vision Encoder)
    log.info("Converting Vision Encoder to ONNX...")
    dummy_pixel_values = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        pytorch_model.vision_model,
        dummy_pixel_values,
        onnx_vision_path,
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
    )
    log.info(f"Vision model saved to {onnx_vision_path}")

    # 3. 导出文本模型 (Text Encoder)
    log.info("Converting Text Encoder to ONNX...")
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 77))
    torch.onnx.export(
        pytorch_model.text_model,
        dummy_input_ids,
        onnx_text_path,
        opset_version=14,
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
        },
    )
    log.info(f"Text model saved to {onnx_text_path}")

    # 4. 将 ONNX 模型转换为 OpenVINO IR
    log.info("Converting ONNX models to OpenVINO IR format...")

    # 转换视觉模型
    ov_vision_model = ov.convert_model(onnx_vision_path)
    ov.save_model(ov_vision_model, ir_vision_path)
    log.info(f"OpenVINO Vision model saved to {ir_vision_path}")

    # 转换文本模型
    ov_text_model = ov.convert_model(onnx_text_path)
    ov.save_model(ov_text_model, ir_text_path)
    log.info(f"OpenVINO Text model saved to {ir_text_path}")

    # 5. 保存 Processor 和 Tokenizer 配置
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Processor and Tokenizer configs saved to {output_dir}")

    # 6. 清理临时的 ONNX 文件
    onnx_vision_path.unlink()
    onnx_text_path.unlink()
    log.info("Temporary ONNX files have been removed.")
    log.info("Conversion process completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Alt-CLIP model to OpenVINO IR format.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"The directory to save the converted models. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()
    convert_alt_clip_to_openvino(args.output_dir)