# scripts/convert_qa_clip_openvino.py
import logging
import sys
from pathlib import Path

# 确保已安装所需库
try:
    import openvino as ov
    import torch
    import torch.nn as nn
    # 核心：使用 transformers 加载
    from transformers import AutoProcessor, AutoModel
except ImportError:
    logging.error("必需库未找到。请运行: pip install openvino openvino-dev torch transformers")
    sys.exit(1)

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 目标模型 (Hugging Face)
MODEL_ID = "TencentARC/QA-CLIP-ViT-L-14"
# ViT-L-14 的原生维度
NATIVE_DIMS = 768
# 从模型配置中获取的标准输入
INPUT_RESOLUTION = 224
# QA-CLIP 使用的上下文长度 (与 OpenAI CLIP 一致)
CONTEXT_LENGTH = 77

# --- 动态路径定义 ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# 输出到 models/qa-clip/openvino 目录
OV_SAVE_PATH = PROJECT_ROOT / "models" / "qa-clip" / "openvino"
CACHE_PATH = PROJECT_ROOT / "cache"


# --- 定义模型包装器 (适配 Transformers/QA-CLIP 结构) ---

class VisionModelWrapper(nn.Module):
    """
    包装器，访问基础 ViT (.vision_model) 和投影层 (.visual_projection)。
    适配 transformers 的 QA-CLIP (ChineseCLIPVisionModel) 结构。
    """
    def __init__(self, model):
        super().__init__()
        # 1. 获取基础 ViT 模型 (在 .vision_model 内部)
        self.vision_model_base = model.vision_model
        # 2. 获取顶层的投影层
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        # 运行基础 ViT 模型。
        # 它返回 (last_hidden_state, pooler_output, ...)
        vision_outputs = self.vision_model_base(pixel_values=pixel_values)

        # 获取 [CLS] 标记的池化输出 (pooler_output, 索引 1)
        pooled_output = vision_outputs[1]

        # 运行最终的投影层
        image_embeds = self.visual_projection(pooled_output)
        return image_embeds

# ... VisionModelWrapper 保持不变 ...

class TextModelWrapper(nn.Module):
    """
    包装器, 访问基础 BERT (.text_model) 和 *外部* 的投影层 (.text_projection)。
    适配 transformers 的 ChineseCLIPTextModel 和 ChineseCLIPModel 结构。
    """
    def __init__(self, model):
        super().__init__()

        # --- 修正 ---
        # ChineseCLIPTextModel 没有 .transformer 属性
        # 它本身就是我们要调用的基础模型。
        self.text_model_base = model.text_model
        # --- 结束修正 ---

        # 2. 获取位于顶层 ChineseCLIPModel 的 text_projection 层
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask):
        # 运行基础 BERT 模型 (现在是 self.text_model_base = model.text_model)
        # 它返回 (last_hidden_state, pooler_output, ...)
        text_outputs = self.text_model_base(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 我们的上一个修复（手动池化）在这里仍然是正确的。
        # 1. 从索引 [0] 获取 last_hidden_state
        last_hidden_state = text_outputs[0]

        # 2. 手动执行池化：提取 [CLS] 标记的输出 (在序列索引 0 处)
        pooled_output = last_hidden_state[:, 0, :]

        # 运行最终的投影层
        text_embeds = self.text_projection(pooled_output)
        return text_embeds

# --- 结束定义包装器 ---


def convert_models():
    """
    执行 Pytorch -> OpenVINO IR (FP16) 的完整转换流程。
    使用 transformers 加载，通过 Wrapper 分离分支，直接转换为 OpenVINO。
    """

    OV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    logging.info(f"OpenVINO 模型将保存到: {OV_SAVE_PATH}")
    logging.info(f"Hugging Face 模型缓存将位于: {CACHE_PATH}")

    core = ov.Core()

    try:
        # --- 步骤 1: 从 Hugging Face 加载模型 ---
        logging.info(f"--- 步骤 1: 正在从 Hugging Face 下载和加载模型: {MODEL_ID} ---")

        # 1.1 加载 Processor (用于生成伪输入)
        # 注意：QA-CLIP 的 processor (tokenizer) 与 cn_clip 不同
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_PATH,
            use_fast=True
        )

        # 1.2 加载完整的 AutoModel (ChineseCLIPModel)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_PATH
        )
        model.eval() # 设置为评估模式

        # --- 步骤 2: 转换 Vision (图像) 模型 ---
        logging.info("--- 步骤 2: 转换 Vision 模型 (FP16) ---")
        ov_vision_path = OV_SAVE_PATH / "openvino_image_fp16.xml"

        # 实例化 Vision 包装器
        vision_wrapper = VisionModelWrapper(model)

        # 创建伪图像输入 (Tensor, 匹配 VisionWrapper.forward)
        dummy_image_input = torch.randn(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)

        logging.info(f"正在转换 Vision 包装器 -> {ov_vision_path}")
        ov_vision_model = ov.convert_model(vision_wrapper, example_input=dummy_image_input)
        ov.save_model(ov_vision_model, ov_vision_path, compress_to_fp16=True)
        logging.info("Vision 模型转换成功。")

        # --- 步骤 3: 转换 Text (文本) 模型 ---
        logging.info("--- 步骤 3: 转换 Text 模型 (FP16) ---")
        ov_text_path = OV_SAVE_PATH / "openvino_text_fp16.xml"

        # 实例化 Text 包装器
        text_wrapper = TextModelWrapper(model)

        # 创建伪文本输入 (Dict, 匹配 TextWrapper.forward)
        # 注意 context_length 使用 QA-CLIP 的 77
        dummy_text_input = {
            "input_ids": torch.randint(0, processor.tokenizer.vocab_size, (1, CONTEXT_LENGTH)),
            "attention_mask": torch.ones(1, CONTEXT_LENGTH, dtype=torch.long)
        }

        logging.info(f"正在转换 Text 包装器 -> {ov_text_path}")
        # 关键：指定 input_ids 和 attention_mask 的动态维度
        ov_text_model = ov.convert_model(text_wrapper, example_input=dummy_text_input)
        ov.save_model(ov_text_model, ov_text_path, compress_to_fp16=True)
        logging.info("Text 模型转换成功。")

    except Exception as e:
        logging.error(f"模型转换失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 步骤 4: 验证转换后的 OpenVINO 模型 ---
    logging.info("--- 步骤 4: 验证 OpenVINO IR 模型 ---")
    try:
        # 验证视觉模型
        vision_model_ov = core.read_model(ov_vision_path)
        vision_output = vision_model_ov.output(0)
        vision_dims = vision_output.get_partial_shape()[1].get_length()

        if vision_dims == NATIVE_DIMS:
            logging.info(f"✅ 视觉模型维度验证成功: {vision_dims}d")
        else:
            raise RuntimeError(f"视觉模型维度错误! 预期: {NATIVE_DIMS}, 得到: {vision_dims}")

        vision_inputs_count = len(vision_model_ov.inputs)
        if vision_inputs_count != 1:
            raise RuntimeError(f"视觉模型输入数量错误! 预期: 1, 得到: {vision_inputs_count}")
        logging.info(f"✅ 视觉模型输入数量验证成功: {vision_inputs_count}")

        # 验证文本模型
        text_model_ov = core.read_model(ov_text_path)
        text_output = text_model_ov.output(0)
        text_dims = text_output.get_partial_shape()[1].get_length()

        if text_dims == NATIVE_DIMS:
            logging.info(f"✅ 文本模型维度验证成功: {text_dims}d")
        else:
            raise RuntimeError(f"文本模型维度错误! 预期: {NATIVE_DIMS}, 得到: {text_dims}")

        text_inputs_count = len(text_model_ov.inputs)
        # TextWrapper 需要 input_ids 和 attention_mask
        if text_inputs_count != 2:
            raise RuntimeError(f"文本模型输入数量错误! 预期: 2, 得到: {text_inputs_count}")
        logging.info(f"✅ 文本模型输入数量验证成功: {text_inputs_count}")

    except Exception as e:
        logging.error(f"模型验证失败: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"🎉 全部转换和验证成功完成。模型保存在: {OV_SAVE_PATH}")

if __name__ == "__main__":
    convert_models()