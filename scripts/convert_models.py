import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_model_with_optimum(output_dir_str: str):
    """
    使用 Hugging Face Optimum 库将 BAAI/AltCLIP-m18 模型转换为 OpenVINO 格式。
    这是目前最可靠、最推荐的转换方法。
    """
    try:
        # 导入必要的库
        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoProcessor
        import openvino as ov
    except ImportError:
        logging.error("错误: 缺少必要的库。请运行 'pip install optimum[openvino]' 来安装。")
        return

    model_name = "BAAI/AltCLIP-m18"
    output_dir = Path(output_dir_str)

    logging.info(f"开始使用 Optimum 从 '{model_name}' 转换模型...")
    logging.info(f"模型将被保存到: {output_dir}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 这一步会自动处理下载、转换和保存所有必要的模型文件
        # export=True 表示如果本地缓存中没有转换好的模型，则执行转换
        ov_model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True, compile=False)

        # 将转换后的模型（包括视觉和文本部分）以及 processor 配置文件保存到指定目录
        ov_model.save_pretrained(output_dir)
        logging.info("Optimum 模型转换成功。")

    except Exception as e:
        logging.error(f"使用 Optimum 进行模型转换时发生严重错误: {e}", exc_info=True)
        return

    # --- 转换后验证 ---
    logging.info("开始验证转换后的模型...")
    try:
        core = ov.Core()
        processor = AutoProcessor.from_pretrained(output_dir)

        # Optimum 会将模型保存为 openvino_vision_model.xml 和 openvino_text_model.xml
        vision_model_path = output_dir / "openvino_vision_model.xml"
        text_model_path = output_dir / "openvino_text_model.xml"

        if not vision_model_path.exists() or not text_model_path.exists():
            raise FileNotFoundError("错误: Optimum 转换后未找到预期的模型文件。")

        # 验证视觉模型
        vision_model = core.read_model(vision_model_path)
        vision_output_shape = vision_model.outputs[0].get_partial_shape()
        logging.info(f"已加载的视觉模型输出维度: {vision_output_shape}")
        if vision_output_shape.rank.get_length() != 2 or vision_output_shape[1].get_length() != 768:
            logging.error(f"验证失败: 视觉模型维度不是 768！")
        else:
            logging.info("验证成功: 视觉模型维度正确 (768)。")

        # 验证文本模型
        text_model = core.read_model(text_model_path)
        text_output_shape = text_model.outputs[0].get_partial_shape()
        logging.info(f"已加载的文本模型输出维度: {text_output_shape}")
        if text_output_shape.rank.get_length() != 2 or text_output_shape[1].get_length() != 768:
            logging.error(f"验证失败: 文本模型维度不是 768！")
        else:
            logging.info("验证成功: 文本模型维度正确 (768)。")

    except Exception as e:
        logging.error(f"验证转换后的模型时出错: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Optimum 将 Alt-CLIP 模型转换为 OpenVINO 格式。")

    # 获取项目根目录 (即脚本文件所在目录的上一级)
    project_root = Path(__file__).parent.parent
    default_output = project_root / "models" / "alt-clip" / "openvino"

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output),
        help="转换后模型的保存目录。"
    )
    args = parser.parse_args()

    convert_model_with_optimum(args.output_dir)
