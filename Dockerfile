FROM openvino/ubuntu24_runtime:2025.3.0

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER root

COPY requirements-docker.txt .

COPY models/chinese-clip/openvino /models/chinese-clip/openvino

COPY models/insightface/models /models/insightface/models

# 更换 APT 源
RUN rm -f /etc/apt/sources.list \
    rm -rf /etc/apt/sources.list.d/

COPY sources.list /etc/apt/sources.list

# 系统依赖
RUN apt update && apt install -y --no-install-recommends \
    python3-dev \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && apt remove g++ -y \
    && apt autoremove -y \
    && apt autoclean -y

COPY app /app

RUN groupadd -g 991 render || true

RUN usermod -a -G render root

# 暴露服务运行的端口
EXPOSE 8060

# 设置容器启动时执行的默认命令
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8060"]

