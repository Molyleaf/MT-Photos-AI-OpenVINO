# 使用轻量级的运行时镜像作为最终的应用镜像
FROM openvino/ubuntu24_runtime:2025.3.0

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制运行时环境所需的依赖文件
COPY requirements.txt .

USER root

# 复制您在本地提前转换好的 Alt-CLIP OpenVINO IR 模型
# 请确保在运行 `docker build` 之前，这些模型文件存在于您项目的 `./models/alt-clip/openvino` 目录下
COPY models/alt-clip/openvino /models/alt-clip/openvino

# 复制预先下载好的 InsightFace 模型
# 在项目构建前，需要将这些模型文件放置在项目根目录的 models/insightface/buffalo_l 目录下
# 将 insightface 模型复制到库所期望的 "models" 子目录中
COPY models/insightface /models/insightface/models

# 更换 APT 源
RUN rm -f /etc/apt/sources.list \
    rm -rf /etc/apt/sources.list.d/

COPY sources.list /etc/apt/sources.list

RUN apt update

# 系统依赖
RUN apt update && apt install -y --no-install-recommends \
    python3-dev \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.pku.edu.cn/pypi/simple/

RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序的源代码
COPY app/server_openvino.py .
COPY app/common/ /app/common/

# 暴露服务运行的端口
EXPOSE 8060

# 使用环境变量来控制推理设备 (例如: "CPU", "GPU", "AUTO")
ENV INFERENCE_DEVICE="AUTO"

# 设置容器启动时执行的默认命令
CMD ["uvicorn", "server_openvino:app", "--host", "0.0.0.0", "--port", "8060"]

