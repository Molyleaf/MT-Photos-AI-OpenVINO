# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-trixie AS wheels-builder

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn

RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/*

COPY sources.list /etc/apt/sources.list

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    # requirements.txt includes the transitions runtime used by the non-text family state machine. \
    pip wheel --cache-dir /root/.cache/pip --wheel-dir /tmp/wheels --prefer-binary -r /tmp/requirements.txt; \
    pip wheel --cache-dir /root/.cache/pip --wheel-dir /tmp/wheels --prefer-binary --no-deps opencv-python-headless; \
    rm -f /tmp/requirements.txt

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
    MODEL_PATH=/models

RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/*

COPY sources.list /etc/apt/sources.list
COPY sources.sid.list /etc/apt/sources.list.d/sid.list
COPY intel-gpu-runtime.pref /etc/apt/preferences.d/intel-gpu-runtime
COPY requirements.txt /tmp/requirements.txt
COPY --from=wheels-builder /tmp/wheels /tmp/wheels

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
            libze-intel-gpu1 \
    rm -f /etc/apt/sources.list.d/sid.list /etc/apt/preferences.d/intel-gpu-runtime

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    pip install --no-index --find-links=/tmp/wheels -r /tmp/requirements.txt; \
    if pip show opencv-python >/dev/null 2>&1; then pip uninstall -y opencv-python; fi; \
    if pip show opencv-contrib-python >/dev/null 2>&1; then pip uninstall -y opencv-contrib-python; fi; \
    pip install --no-index --find-links=/tmp/wheels --force-reinstall --no-deps opencv-python-headless; \
    rm -rf /tmp/requirements.txt /tmp/wheels

RUN set -eux; \
    RAPIDOCR_MODEL_ROOT="$(python -c "import pathlib, rapidocr; print(pathlib.Path(rapidocr.__file__).resolve().parent / 'models')")"; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin appuser; \
    mkdir -p /cache /models/qa-clip/openvino /models/insightface/models/antelopev2 /models/cache/openvino "${RAPIDOCR_MODEL_ROOT}"; \
    chmod 777 /cache; \
    chown -R appuser:appgroup /app /models "${RAPIDOCR_MODEL_ROOT}"

COPY --chown=appuser:appgroup models/qa-clip/openvino/openvino_image.xml /models/qa-clip/openvino/openvino_image.xml
COPY --chown=appuser:appgroup models/qa-clip/openvino/openvino_image.bin /models/qa-clip/openvino/openvino_image.bin
COPY --chown=appuser:appgroup models/insightface/models/antelopev2 /models/insightface/models/antelopev2

RUN set -eux; \
    test -f /models/qa-clip/openvino/openvino_image.xml; \
    test -f /models/qa-clip/openvino/openvino_image.bin; \
    test -f /models/insightface/models/antelopev2/glintr100.onnx; \
    test -f /models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx

COPY --chown=appuser:appgroup app /app
COPY --chown=appuser:appgroup scripts /app/scripts

USER appuser

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD sh -c "python -c \"import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8060') + '/', timeout=3)\" || exit 1"

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8060} --log-level $(printf '%s' \"${LOG_LEVEL:-warning}\" | tr '[:upper:]' '[:lower:]')"]
