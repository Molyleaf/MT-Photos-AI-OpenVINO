import argparse
from pathlib import Path
import logging
import sys
import torch
import openvino as ov
from transformers import AutoProcessor, AutoModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 修改：包装类现在包含 编码器 + 投影层 ---
class VisionModelWrapper(torch.nn.Module):
    """
    包装视觉模型及其投影层，
    使其 forward 时返回 [batch_size, 1024] 的最终投影特征。
    """
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        # 1. 编码器输出 (得到 [?, 1280] 维的 pooler_output)
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        # 2. 投影层 (将 [?, 1280] 转换为 [?, 1024])
        projected_output = self.visual_projection(pooled_output)
        return projected_output

class TextModelWrapper(torch.nn.Module):
    """
    包装文本模型及其投影层，
    使其 forward 时返回 [batch_size, 1024] 的最终投影特征。
    """
    def __init__(self, text_model, text_projection):
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection

    def forward(self, input_ids):
        # 1. 编码器输出 (得到 [?, 1024] 维的 pooler_output)
        outputs = self.text_model(input_ids=input_ids)
        pooled_output = outputs.pooler_output
        # 2. 投影层 (将 [?, 1024] 转换为 [?, 1024])
        projected_output = self.text_projection(pooled_output)
        return projected_output
# --- 结束修改 ---


def convert_model_manual(output_dir_str: str):
    model_name = "BAAI/AltCLIP-m18"
    output_dir = Path(output_dir_str)

    logging.info(f"开始从 '{model_name}' 手动转换模型...")
    logging.info(f"模型将被保存到: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- 1. 加载并保存处理器 ---
        logging.info("加载 Processor...")
        # trust_remote_code=True 是必需的
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.save_pretrained(output_dir)
        logging.info(f"Processor 文件已保存到 {output_dir}")

        # --- 2. 加载 PyTorch 模型 ---
        logging.info("加载 PyTorch 模型 (trust_remote_code=True)...")
        pt_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        pt_model.eval() # 设置为评估模式

        # --- 修改：实例化包装模型 (传入编码器和投影层) ---
        logging.info("创建用于导出的包装模型 (包含投影层)...")
        vision_model = VisionModelWrapper(
            pt_model.vision_model,
            pt_model.visual_projection
        )
        text_model = TextModelWrapper(
            pt_model.text_model,
            pt_model.text_projection
        )
        # --- 结束修改 ---

        # --- 3. 准备 ONNX 导出的虚拟输入 ---

        # 视觉模型
        # (batch_size, num_channels, height, width)
        dummy_pixel_values = torch.randn(1, 3, 224, 224)

        # 文本模型
        # (batch_size, sequence_length)
        seq_len = processor.tokenizer.model_max_length
        vocab_size = processor.tokenizer.vocab_size
        dummy_input_ids = torch.randint(0, vocab_size, (1, seq_len))

        vision_onnx_path = output_dir / "vision_model.onnx"
        text_onnx_path = output_dir / "text_model.onnx"

        # --- 4. 导出视觉模型到 ONNX ---
        logging.info("导出视觉模型到 ONNX...")
        torch.onnx.export(
            vision_model,            # <--- MODIFIED: 使用新的包装器
            dummy_pixel_values,
            vision_onnx_path,
            input_names=["pixel_values"],
            output_names=["pooler_output"], # 现在这个名称对应正确的 [?, 1024] 张量
            dynamic_axes={"pixel_values": {0: "batch_size"}},
            opset_version=18  # 保持 opset 18
        )

        # --- 5. 导出文本模型到 ONNX ---
        logging.info("导出文本模型到 ONNX...")
        torch.onnx.export(
            text_model,              # <--- MODIFIED: 使用新的包装器
            dummy_input_ids,
            text_onnx_path,
            input_names=["input_ids"],
            output_names=["pooler_output"], # 现在这个名称对应正确的 [?, 1024] 张量
            dynamic_axes={"input_ids": {0: "batch_size"}},
            opset_version=18  # 保持 opset 18
        )

        # --- 6. 将 ONNX 转换为 OpenVINO ---
        logging.info("使用 OpenVINO 转换 ONNX 模型...")
        core = ov.Core()

        ov_vision_model = ov.convert_model(vision_onnx_path)
        ov_text_model = ov.convert_model(text_onnx_path)

        # --- 7. 保存 OpenVINO 模型 ---
        # 保存为你的验证脚本期望的 .xml/.bin 文件
        vision_model_path = output_dir / "openvino_vision_model.xml"
        text_model_path = output_dir / "openvino_text_model.xml"

        ov.save_model(ov_vision_model, vision_model_path)
        ov.save_model(ov_text_model, text_model_path)

        logging.info(f"OpenVINO 视觉模型已保存到: {vision_model_path}")
        logging.info(f"OpenVINO 文本模型已保存到: {text_model_path}")

        # --- 8. 清理临时的 ONNX 文件 ---
        vision_onnx_path.unlink()
        text_onnx_path.unlink()
        logging.info("临时的 ONNX 文件已删除。")

    except Exception as e:
        logging.error(f"手动模型转换时发生严重错误: {e}", exc_info=True)
        sys.exit(1)

    # --- 你的验证逻辑 (无需更改，现在应该会通过) ---
    logging.info("--- 开始验证转换后的模型 ---")
    try:
        core = ov.Core()
        # 验证处理器是否已正确保存
        AutoProcessor.from_pretrained(output_dir)
        logging.info("✅ 验证成功: Processor 加载正常。")

        vision_model_path = output_dir / "openvino_vision_model.xml"
        text_model_path = output_dir / "openvino_text_model.xml"

        if not vision_model_path.exists() or not text_model_path.exists():
            raise FileNotFoundError("错误: 手动转换后未找到预期的模型文件。")

        vision_model = core.read_model(vision_model_path)
        # 重命名验证日志中的 "V视觉模型" 为 "视觉模型"
        vision_output_shape = vision_model.output("pooler_output").get_partial_shape()
        logging.info(f"已加载的视觉模型 'pooler_output' 维度: {vision_output_shape}")
        if vision_output_shape.rank.get_length() != 2 or vision_output_shape[1].get_length() != 1024:
            logging.error(f"验证失败: 视觉模型维度不是 1024！")
        else:
            logging.info("✅ 验证成功: 视觉模型维度正确 (1024)。")

        text_model = core.read_model(text_model_path)
        text_output_shape = text_model.output("pooler_output").get_partial_shape()
        logging.info(f"已加载的文本模型 'pooler_output' 维度: {text_output_shape}")
        if text_output_shape.rank.get_length() != 2 or text_output_shape[1].get_length() != 1024:
            logging.error(f"验证失败: 文本模型维度不是 1024！")
        else:
            logging.info("✅ 验证成功: 文本模型维度正确 (1024)。")

        logging.info("🎉 全部转换和验证成功完成。")

    except Exception as e:
        logging.error(f"验证转换后的模型时出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手动将 Alt-CLIP 模型转换为 OpenVINO 格式。")
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "models" / "alt-clip" / "openvino"
    parser.add_argument("--output_dir", type=str, default=str(default_output), help="转换后模型的保存目录。")
    args = parser.parse_args()

    # 调用新的手动转换函数
    convert_model_manual(args.output_dir)