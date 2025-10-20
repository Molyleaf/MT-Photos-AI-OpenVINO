# ---- 阶段 1: 模型转换构建器 ----
# 使用包含所有转换所需工具的开发镜像
FROM openvino/ubuntu24_dev:2025.3.0 AS builder

WORKDIR /builder

# 优先复制依赖和脚本文件，以便利用 Docker 的层缓存机制
COPY requirements.txt .
COPY scripts/convert_models.py ./scripts/

# 安装模型转换过程需要的所有依赖，并使用国内镜像源加速
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/web/simple -r requirements.txt

# 运行模型转换脚本
# 该脚本会从 Hugging Face 下载 BAAI/AltCLIP-m18 模型，
# 先将其转换为 ONNX 格式，然后转换为 OpenVINO IR FP16 格式。
# 转换后的模型将输出到 /builder/models/alt-clip/openvino 目录
RUN python scripts/convert_models.py --output_dir /builder/models/alt-clip

# ---- 阶段 2: 最终运行时镜像 ----
# 使用轻量级的运行时镜像作为最终的应用镜像
FROM openvino/ubuntu24_runtime:2025.3.0

WORKDIR /app

# 复制运行时环境所需的依赖文件
COPY requirements.txt .

# 仅安装运行时必要的依赖，以保持镜像的轻量
# 这里排除了 torch, onnx 等只在构建阶段需要的库
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/web/simple -r requirements.txt

# 从构建器阶段复制已经转换好的 OpenVINO IR 模型
COPY --from=builder /builder/models/alt-clip/openvino /models/alt-clip/openvino

# 复制预先下载好的 InsightFace 模型
# 在项目构建前，需要将这些模型文件放置在项目根目录的 models/insightface/buffalo_l 目录下
COPY models/insightface/buffalo_l /models/insightface/buffalo_l

# 复制应用程序的源代码
COPY app/server_openvino.py .
COPY app/common/ /app/common/

# 暴露服务运行的端口
EXPOSE 8060

# 设置运行应用所需的默认环境变量
# 使用环境变量来设置 API 密钥，增强安全性
ENV API_AUTH_KEY=""
# 使用环境变量来控制推理设备 (例如: "CPU", "GPU", "AUTO")
ENV INFERENCE_DEVICE="AUTO"

# 设置容器启动时执行的默认命令
CMD ["uvicorn", "server_openvino.py", "--host", "0.0.0.0", "--port", "8060"]

