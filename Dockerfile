FROM python:3.12-slim-trixie

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_DEVICE=GPU \
    WEB_CONCURRENCY=2 \
    OV_CACHE_DIR=/models/cache/openvino \
    RAPIDOCR_OPENVINO_CONFIG_PATH=/example/cfg_openvino_cpu.yaml \
    RAPIDOCR_MODEL_DIR=/models/rapidocr

USER root

# Optional APT mirror override.
RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/*

COPY sources.list /etc/apt/sources.list

RUN apt update \
    && apt dist-upgrade -y \
    && apt install -y --no-install-recommends \
        python3-dev \
        g++ \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ocl-icd-libopencl1 \
        mesa-opencl-icd \
        libva2 \
        mesa-va-drivers \
        clinfo \
        vainfo \
        mesa-vulkan-drivers \
        libclang-rt-19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.pku.edu.cn/pypi/simple/ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt remove g++ -y \
    && apt autoremove -y \
    && apt autoclean -y

RUN mkdir -p /models/qa-clip/openvino /models/insightface/models /models/rapidocr /models/cache/openvino

COPY models/qa-clip/openvino /models/qa-clip/openvino
COPY models/insightface/models /models/insightface/models
COPY app/config/cfg_openvino_cpu.yaml /example/cfg_openvino_cpu.yaml
COPY app /app

EXPOSE 8060

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8060 --workers ${WEB_CONCURRENCY:-2}"]
