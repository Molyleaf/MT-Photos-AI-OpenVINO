# 使用轻量级的运行时镜像作为最终的应用镜像
FROM openvino/ubuntu24_runtime:2025.3.0

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制运行时环境所需的依赖文件
COPY requirements.txt .

USER root [cite: 4]

# 更换 APT 源
RUN rm -f /etc/apt/sources.list \
    rm -rf /etc/apt/sources.list.d/

COPY sources.list /etc/apt/sources.list

RUN apt update

# 系统依赖
RUN apt install -y \
    python3-dev g++ && \
    rm -rf /var/lib/apt/lists/* \

RUN pip config set global.index-url https://mirrors.pku.edu.cn/pypi/simple/

# 仅安装运行时必要的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制您在本地提前转换好的 Alt-CLIP OpenVINO IR 模型
# 请确保在运行 `docker build` 之前，这些模型文件存在于您项目的 `./models/alt-clip/openvino` 目录下
COPY models/alt-clip/openvino /models/alt-clip/openvino

# 复制预先下载好的 InsightFace 模型
# 在项目构建前，需要将这些模型文件放置在项目根目录的 models/insightface/buffalo_l 目录下
COPY models/insightface /models/insightface

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

