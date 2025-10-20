import torch
import openvino as ov
from transformers import AltCLIPModel, AltCLIPProcessor
import os
import argparse
from pathlib import Path

def convert_alt_clip_to_openvino(output_dir: Path):
    """
    Downloads the Alt-CLIP model, converts it to ONNX, and then to OpenVINO IR.
    """
    model_name = "BAAI/AltCLIP-m18"
    print(f"Loading model: {model_name}")
    processor = AltCLIPProcessor.from_pretrained(model_name)
    model = AltCLIPModel.from_pretrained(model_name)
    model.eval()

    # --- Directory Setup ---
    onnx_dir = output_dir / "onnx"
    openvino_dir = output_dir / "openvino"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    openvino_dir.mkdir(parents=True, exist_ok=True)

    # --- Vision Model Conversion ---
    print("Converting Vision model...")
    vision_onnx_path = onnx_dir / "clip_vision.onnx"
    vision_ov_path = openvino_dir / "clip_vision.xml"

    # Fake input for a vision model
    dummy_vision_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model.vision_model,
        dummy_vision_input,
        str(vision_onnx_path),
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
    )
    print(f"Vision model saved to ONNX: {vision_onnx_path}")

    # ONNX to OpenVINO IR for Vision Model
    ov_vision_model = ov.convert_model(str(vision_onnx_path))
    ov.save_model(ov_vision_model, vision_ov_path, compress_to_fp16=True)
    print(f"Vision model saved to OpenVINO IR: {vision_ov_path}")


    # --- Text Model Conversion ---
    print("Converting Text model...")
    text_onnx_path = onnx_dir / "clip_text.onnx"
    text_ov_path = openvino_dir / "clip_text.xml"

    # Dummy input for text model
    dummy_text_input_ids = torch.randint(0, 1000, (1, 77))
    dummy_text_attention_mask = torch.ones(1, 77, dtype=torch.long)

    torch.onnx.export(
        model.text_model,
        (dummy_text_input_ids, dummy_text_attention_mask),
        str(text_onnx_path),
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "text_embeds": {0: "batch_size"},
        },
    )
    print(f"Text model saved to ONNX: {text_onnx_path}")

    # ONNX to OpenVINO IR for Text Model
    ov_text_model = ov.convert_model(str(text_onnx_path))
    ov.save_model(ov_text_model, text_ov_path, compress_to_fp16=True)
    print(f"Text model saved to OpenVINO IR: {text_ov_path}")

    # --- Save processor files ---
    processor.save_pretrained(openvino_dir)
    print(f"Processor files saved to: {openvino_dir}")

    print("\nConversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Alt-CLIP model to OpenVINO format.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/alt-clip",
        help="The directory to save the converted models.",
    )
    args = parser.parse_args()

    convert_alt_clip_to_openvino(Path(args.output_dir))
