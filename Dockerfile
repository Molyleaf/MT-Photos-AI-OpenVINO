FROM python:3.12-slim-trixie

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=1000
ARG APP_GID=1000
ARG PIP_INDEX_URL=https://mirrors.zju.edu.cn/pypi/web/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    INFERENCE_DEVICE=GPU \
    WEB_CONCURRENCY=2 \
    OV_CACHE_DIR=/models/cache/openvino \
    RAPIDOCR_OPENVINO_CONFIG_PATH=/app/config/cfg_openvino_cpu.yaml \
    RAPIDOCR_MODEL_DIR=/models/rapidocr \
    LIBVA_DRIVER_NAME=iHD

COPY sources.list /etc/apt/sources.list

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        clinfo \
        vainfo \
        libdrm2 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libva2 \
        libva-drm2 \
        intel-media-va-driver-non-free \
        libvpl2 \
        libmfx-gen1.2 \
        libze1 \
        ocl-icd-libopencl1 \
        mesa-opencl-icd; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN set -eux; \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    rm -f /tmp/requirements.txt

RUN set -eux; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin appuser; \
    mkdir -p /models/qa-clip/openvino /models/insightface/models /models/rapidocr /models/cache/openvino; \
    chown -R appuser:appgroup /app /models

COPY --chown=appuser:appgroup app /app
COPY --chown=appuser:appgroup models/qa-clip/openvino /models/qa-clip/openvino
COPY --chown=appuser:appgroup models/insightface/models /models/insightface/models

USER appuser

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8060/', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8060 --workers ${WEB_CONCURRENCY:-2}"]
