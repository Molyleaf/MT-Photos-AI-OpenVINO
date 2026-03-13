# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-trixie

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn \
    INFERENCE_DEVICE=AUTO \
    WEB_CONCURRENCY=1 \
    RAPIDOCR_OPENVINO_CONFIG_PATH=/app/config/cfg_openvino_cpu.yaml \
    RAPIDOCR_MODEL_DIR=/models/rapidocr

RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/*

COPY sources.list /etc/apt/sources.list
COPY sources.sid.list /etc/apt/sources.list.d/sid.list
COPY intel-gpu-runtime.pref /etc/apt/preferences.d/intel-gpu-runtime

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libdrm2 \
        libglib2.0-0 \
        libgomp1 \
        libze1 \
        ocl-icd-libopencl1; \
    apt-get install -y --no-install-recommends -t sid \
        intel-opencl-icd \
        libze-intel-gpu1

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    pip install --cache-dir /root/.cache/pip --prefer-binary -r /tmp/requirements.txt; \
    if pip show opencv-python >/dev/null 2>&1; then pip uninstall -y opencv-python; fi; \
    if pip show opencv-contrib-python >/dev/null 2>&1; then pip uninstall -y opencv-contrib-python; fi; \
    pip install --cache-dir /root/.cache/pip --prefer-binary opencv-python-headless; \
    rm -f /tmp/requirements.txt

RUN set -eux; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin appuser; \
    mkdir -p /cache /models/qa-clip/openvino /models/insightface/models/antelopev2 /models/rapidocr /models/cache/openvino; \
    chmod 777 /cache; \
    chown -R appuser:appgroup /app /models

COPY --chown=appuser:appgroup models/qa-clip/openvino /models/qa-clip/openvino
COPY --chown=appuser:appgroup models/insightface/models/antelopev2 /models/insightface/models/antelopev2
COPY --chown=appuser:appgroup models/rapidocr /models/rapidocr

RUN set -eux; \
    test -f /models/qa-clip/openvino/openvino_image_fp16.xml; \
    test -f /models/qa-clip/openvino/openvino_image_fp16.bin; \
    test -f /models/qa-clip/openvino/openvino_text_fp16.xml; \
    test -f /models/qa-clip/openvino/openvino_text_fp16.bin; \
    test -f /models/insightface/models/antelopev2/glintr100.onnx; \
    test -f /models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx; \
    test -f /models/rapidocr/ch_PP-OCRv5_mobile_det.onnx; \
    test -f /models/rapidocr/ch_PP-OCRv5_rec_mobile_infer.onnx; \
    test -f /models/rapidocr/ppocrv5_dict.txt; \
    test -f /models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx

COPY --chown=appuser:appgroup app /app

USER appuser

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8060/', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8060 --workers ${WEB_CONCURRENCY:-1}"]
