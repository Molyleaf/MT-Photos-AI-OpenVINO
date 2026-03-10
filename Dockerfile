FROM python:3.12-slim-trixie

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=1024
ARG APP_GID=100
ARG PIP_INDEX_URL=https://mirrors.zju.edu.cn/pypi/web/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    INFERENCE_DEVICE=AUTO \
    WEB_CONCURRENCY=1 \
    OV_CACHE_DIR=/models/cache/openvino \
    RAPIDOCR_OPENVINO_CONFIG_PATH=/app/config/cfg_openvino_cpu.yaml \
    RAPIDOCR_MODEL_DIR=/models/rapidocr

RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/
COPY sources.list /etc/apt/sources.list
COPY requirements.txt /tmp/requirements.txt

RUN set -eux; \
    ov_opencl_runtime_deps="\
        libdrm2 \
        libze1 \
        ocl-icd-libopencl1 \
        mesa-opencl-icd"; \
    intel_gpu_runtime_deps="\
        intel-opencl-icd \
        libze-intel-gpu1"; \
    python_runtime_deps="\
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1"; \
    printf '%s\n' \
        'deb https://mirrors.zju.edu.cn/debian sid main contrib non-free non-free-firmware' \
        > /etc/apt/sources.list.d/sid.list; \
    printf '%s\n' \
        'Package: *' \
        'Pin: release n=sid' \
        'Pin-Priority: 100' \
        '' \
        'Package: intel-opencl-icd libze-intel-gpu1' \
        'Pin: release n=sid' \
        'Pin-Priority: 990' \
        > /etc/apt/preferences.d/intel-gpu-runtime; \
    apt-get update; \
    apt-get install -y --no-install-recommends $ov_opencl_runtime_deps $python_runtime_deps; \
    apt-get install -y --no-install-recommends -t sid $intel_gpu_runtime_deps; \
    pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt; \
    rm -f /etc/apt/sources.list.d/sid.list /etc/apt/preferences.d/intel-gpu-runtime; \
    rm -rf /var/lib/apt/lists/* /tmp/requirements.txt

RUN set -eux; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin appuser; \
    mkdir -p /models/qa-clip/openvino /models/insightface/models/antelopev2 /models/rapidocr /models/cache/openvino; \
    chown -R appuser:appgroup /app /models

COPY --chown=appuser:appgroup app /app
COPY --chown=appuser:appgroup models/qa-clip/openvino /models/qa-clip/openvino
COPY --chown=appuser:appgroup models/insightface/models/antelopev2 /models/insightface/models/antelopev2
COPY --chown=appuser:appgroup models/rapidocr /models/rapidocr

RUN set -eux; \
    test -f /models/qa-clip/openvino/openvino_image_fp16.xml; \
    test -f /models/qa-clip/openvino/openvino_image_fp16.bin; \
    test -f /models/qa-clip/openvino/openvino_text_fp16.xml; \
    test -f /models/qa-clip/openvino/openvino_text_fp16.bin; \
    test -f /models/insightface/models/antelopev2/1k3d68.onnx; \
    test -f /models/insightface/models/antelopev2/2d106det.onnx; \
    test -f /models/insightface/models/antelopev2/genderage.onnx; \
    test -f /models/insightface/models/antelopev2/glintr100.onnx; \
    test -f /models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx; \
    test -f /models/rapidocr/ch_PP-OCRv5_mobile_det.onnx; \
    test -f /models/rapidocr/ch_PP-OCRv5_rec_mobile_infer.onnx; \
    test -f /models/rapidocr/ppocrv5_dict.txt; \
    test -f /models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx

USER appuser

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8060/', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8060 --workers ${WEB_CONCURRENCY:-2}"]
