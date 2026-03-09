# MT-Photos AI (OpenVINO)

统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）的 FastAPI 服务。本文档仅保留部署相关内容：Windows 本机部署与 Debian/Linux Docker 部署。

## 部署前准备

- Python **3.12**
- 已准备模型目录（至少包含）：
  - `models/qa-clip/openvino`
  - `models/insightface/models`
  - `models/rapidocr`（必须预置 `PP-OCRv5 mobile det/rec/dict + cls` 本地文件）
- 服务入口：`app/server.py`

## Debian 容器目标与当前实践

当前仓库已支持 Debian 容器部署，基线为 `python:3.12-slim-trixie`（Debian 13）。

当前 Docker 实践状态（已落地）：

- 使用 Debian 13 基础镜像（`trixie`）
- APT 镜像固定为 `https://mirrors.zju.edu.cn/debian/`
- PyPI 镜像固定为 `https://mirrors.zju.edu.cn/pypi/web/simple`
- 使用 `apt-get install --no-install-recommends`，并清理 apt 索引
- `pip install` 完成后卸载 InsightFace 构建依赖（`build-essential/gcc/g++/libpq-dev`）
- 以非 root 用户运行服务（可通过 `APP_UID` / `APP_GID` 对齐宿主机权限）
- 提供容器健康检查（`GET /`，不依赖 API Key）
- 提供 `.dockerignore`，降低构建上下文体积
- 内置 OpenVINO + Xe 核显图片转码所需驱动（`intel-media-va-driver-non-free`、`libvpl2`、`libmfx-gen1.2`、`libze1`、`ocl-icd-libopencl1`、`mesa-opencl-icd`、`libva2`、`libva-drm2`、`ffmpeg`）
- 镜像内仅打包 InsightFace `antelopev2` 模型，移除 `buffalo_l` 分支
- `docker-compose` 默认不挂载 `/models`，模型随镜像静态打包
- 启动阶段增加 `/dev/dri` 自检：请求 GPU 推理时，`/dev/dri` 不可用将直接报错并终止启动

仍需按宿主机确认项：

- Debian 宿主需正确映射 `/dev/dri` 才能使用 Intel iGPU
- `VIDEO_GID` / `RENDER_GID` 需与宿主机设备组一致（默认 `44/109`）

### Debian 13 Xe 依赖（干净 apt install 清单）

以下列表可直接用于 Debian 13 宿主或容器内安装（与当前 Dockerfile 一致）：

```bash
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  ffmpeg \
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
  mesa-opencl-icd
```

## 环境变量（运行时，全量）

| 环境变量 | 可选值 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | 任意字符串；`no-key` 或空字符串表示关闭鉴权 | `mt_photos_ai_extra` |
| `INFERENCE_DEVICE` | OpenVINO 设备字符串，如 `GPU` / `CPU` / `AUTO` / `AUTO:GPU,CPU` | `AUTO` |
| `CLIP_INFERENCE_DEVICE` | 同 `INFERENCE_DEVICE`；仅覆盖 QA-CLIP 设备。值为 `AUTO` 或显式包含 `GPU` 时都要求成功初始化 GPU Remote Context；失败直接报错（无 silent fallback） | 跟随 `INFERENCE_DEVICE` |
| `MODEL_PATH` | 模型根目录路径 | `<repo>/models` |
| `WEB_CONCURRENCY` | 整数，建议 `>=1` | `2` |
| `INFERENCE_QUEUE_MAX_SIZE` | 整数；运行时会被约束为 `>=1` | `64` |
| `TEXT_CLIP_BATCH_SIZE` | 整数；运行时会被约束为 `>=1` | `8` |
| `INFERENCE_TASK_TIMEOUT` | 整数秒；运行时会被约束为 `>=1` | `120` |
| `SERVER_IDLE_TIMEOUT` | 整数秒；`<=0` 表示禁用空闲释放计时器 | `300` |
| `TEXT_MODEL_RESTORE_DELAY_MS` | 整数毫秒；运行时会被约束为 `>=0` | `2000` |
| `RESTART_TEXT_RESTORE_DELAY_MS` | 旧变量，仅在未设置 `TEXT_MODEL_RESTORE_DELAY_MS` 时生效 | `2000` |
| `OV_CACHE_DIR` | OpenVINO 编译缓存目录路径 | `<repo>/cache/openvino` |
| `RAPIDOCR_OPENVINO_CONFIG_PATH` | RapidOCR 配置文件路径（YAML） | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR` | RapidOCR 模型目录（必须包含 `PP-OCRv5 mobile det/rec/dict` 与 `ch_ppocr_mobile_v2.0_cls_infer.onnx`，缺失即失败） | `<repo>/models/rapidocr` |
| `RAPIDOCR_FONT_PATH` | 字体文件路径；空表示不指定 | 空 |
| `RAPIDOCR_INFERENCE_NUM_THREADS` | 整数，建议 `1~CPU核心数` | `-1`（跟随 `cfg_openvino_cpu.yaml`） |
| `RAPIDOCR_PERFORMANCE_HINT` | OpenVINO 性能提示，常用：`LATENCY` / `THROUGHPUT` | `LATENCY` |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | 整数；`-1` 表示自动 | `-1` |
| `RAPIDOCR_ENABLE_CPU_PINNING` | 布尔：`1/true/yes/on` 或 `0/false/no/off`（大小写不敏感） | `true` |
| `RAPIDOCR_NUM_STREAMS` | 整数；`-1` 表示自动 | `1`（跟随 `cfg_openvino_cpu.yaml`） |
| `RAPIDOCR_ENABLE_HYPER_THREADING` | 布尔：`1/true/yes/on` 或 `0/false/no/off` | `true` |
| `RAPIDOCR_SCHEDULING_CORE_TYPE` | OpenVINO 核调度类型，常用：`ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY` | `ANY_CORE` |
| `INSIGHTFACE_OV_DEVICE` | 透传给 ORT OpenVINO EP 的 `device_type`；常见：`CPU_FP32` / `CPU_FP16` / `GPU_FP32` / `GPU_FP16` | `CPU_FP32` |
| `PORT` | 端口号（整数） | `8060` |
| `LOG_LEVEL` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` | `WARNING` |
| `FFMPEG_BIN` | 图片解码 ffmpeg 可执行文件名/路径（服务内非 GIF 上传默认优先走 QSV+ffmpeg） | `ffmpeg` |
| `FFPROBE_BIN` | 图片解码尺寸探测 ffprobe 可执行文件名/路径 | `ffprobe` |

## 环境变量（`app/convert.py`）

| 环境变量 | 可选值 | 默认值 |
|---|---|---|
| `PROJECT_ROOT` | 项目根目录路径 | 自动探测 |
| `MODEL_PATH` | 模型根目录路径（导出会写入 `qa-clip/openvino`） | `<PROJECT_ROOT>/models` |
| `HF_CACHE_DIR` | Hugging Face 缓存目录路径 | `<PROJECT_ROOT>/cache/huggingface` |
| `QA_CLIP_ENABLE_NNCF_WEIGHT_COMPRESSION` | `0`（关闭）或 `1`（开启） | `0` |
| `QA_CLIP_NNCF_WEIGHT_MODE` | NNCF 压缩模式名（常见：`INT8_ASYM` / `INT8_SYM` / `NF4` / `E2M1`） | `INT8_ASYM` |

`convert.py` 还会在未预设时自动设置以下变量：`HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`、`HF_HUB_DISABLE_SYMLINKS_WARNING`。

## Windows 本机部署

1. 进入仓库根目录，确认 Python 版本：

```powershell
python -V
```

2. （可选）创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. 安装依赖：

```powershell
pip install -r requirements.txt
```

4. （可选）设置环境变量（示例）：

```powershell
$env:API_AUTH_KEY="your_secret_key"
$env:INFERENCE_DEVICE="GPU"
$env:WEB_CONCURRENCY="2"
```

5. 启动服务：

```powershell
cd app
uvicorn server:app --host 0.0.0.0 --port 8060 --workers 2
```

6. 冒烟检查（示例）：

```powershell
Invoke-WebRequest http://127.0.0.1:8060/ -UseBasicParsing
```

业务端点（如 `/check`、`/clip/txt`、`/ocr`、`/represent`）请在请求头携带 `api-key`（当 `API_AUTH_KEY` 不是 `no-key` 时）。

## Debian/Linux Docker 部署

推荐宿主环境：

- Debian 13（或兼容发行版）
- Docker Engine 24+ 与 Docker Compose v2
- Intel iGPU 场景下，宿主机可见 `/dev/dri`（可用 `ls -l /dev/dri` 验证）

### 方式一：docker compose（推荐）

1. 准备配置文件：

```bash
cp docker-compose.example.yml docker-compose.yml
```

2. 根据宿主机设置用户与 GPU 组（Debian 示例）：

```bash
export APP_UID=$(id -u)
export APP_GID=$(id -g)
export VIDEO_GID=$(getent group video | cut -d: -f3)
export RENDER_GID=$(getent group render | cut -d: -f3)
```

3. 编辑 `docker-compose.yml`：
- 默认 `API_AUTH_KEY=mt_photos_ai_extra`，生产环境建议通过环境变量覆盖
- 根据机器情况设置 `INFERENCE_DEVICE`（有 Intel iGPU 且已映射 `/dev/dri` 时可用 `GPU`）
- 如需替换 PyPI 镜像，可覆盖 `PIP_INDEX_URL`（默认已是浙大镜像）
- 按需调整并发与缓存目录（`/models` 默认由镜像静态提供，不再挂载宿主机目录）

4. 启动：

```bash
docker compose up -d --build
```

5. 查看状态：

```bash
docker compose ps
docker compose logs -f mt-photos-ai-openvino
```

### 方式二：docker run

```bash
docker build \
  --build-arg APP_UID=$(id -u) \
  --build-arg APP_GID=$(id -g) \
  --build-arg PIP_INDEX_URL=https://mirrors.zju.edu.cn/pypi/web/simple \
  -t mt-photos-ai-openvino .
docker run -d \
  --name mt-photos-ai-openvino \
  --init \
  -p 8060:8060 \
  --device /dev/dri:/dev/dri \
  --group-add $(getent group video | cut -d: -f3) \
  --group-add $(getent group render | cut -d: -f3) \
  -e API_AUTH_KEY=mt_photos_ai_extra \
  -e INFERENCE_DEVICE=GPU \
  -e WEB_CONCURRENCY=2 \
  -e OV_CACHE_DIR=/models/cache/openvino \
  -e LIBVA_DRIVER_NAME=iHD \
  mt-photos-ai-openvino
```

### OpenVINO 驱动检查（容器内）

首次部署建议执行以下检查，确认 OpenVINO GPU 运行环境完整：

```bash
docker exec -it mt-photos-ai-openvino ls -l /dev/dri
docker exec -it mt-photos-ai-openvino ffmpeg -hide_banner -hwaccels
```

若服务启动时请求了 GPU 推理（`INFERENCE_DEVICE/CLIP_INFERENCE_DEVICE` 含 `GPU` 或 `AUTO`），应用会在启动阶段强制执行 `/dev/dri` 自检；自检失败会直接报错并终止启动（无静默回退）。

服务内 `read_image_from_upload` 对非 GIF 上传默认采用“`ffmpeg(QSV)` -> `ffmpeg(CPU)` -> `cv2.imdecode`”顺序解码，优先使用 Intel 核显链路并显式记录每一级失败原因；`ffprobe` 失败/未返回尺寸时仍会先尝试 `ffmpeg(QSV/CPU)`，再回退 `cv2.imdecode`；检测到高位深图像时会直接使用 `cv2.imdecode` 以保持 16-bit 转 8-bit 语义。
`/clip/img` 端点已去除 `BGR -> RGB -> PIL` 额外拷贝链，模型层直接消费 `numpy BGR` 并优先走 OpenVINO host tensor。

## Windows Server GPU-PV（`/dev/dxg`）能力边界

在 Windows Server + Linux 容器 + GPU-PV（映射 `/dev/dxg`）场景下，当前服务可启用的加速项：

- QA-CLIP OpenVINO GPU 推理（`CLIP_INFERENCE_DEVICE=AUTO/GPU`）；
- QA-CLIP GPU Remote Context + host tensor 互操作（减少 Host<->Device 拷贝）；
- InsightFace 的 OpenVINO EP（当 `INSIGHTFACE_OV_DEVICE` 配置为 GPU 类型且运行时可用）。

需要明确的限制：

- RapidOCR 固定 OpenVINO CPU 后端，不走 GPU。
- `ffmpeg QSV` 在 Linux 容器里通常依赖 `/dev/dri`（VAAPI/oneVPL）链路；仅有 `/dev/dxg` 时不保证可用。若 QSV 不可用，服务会记录原因并按既定顺序显式降级到 `ffmpeg(CPU)` 或 `cv2.imdecode`（无静默 fallback）。

## RapidOCR 模型预置（强制）

服务已禁用线上下载回退，必须在镜像/部署目录预置以下 4 个本地文件（缺失即失败）：

当前 `Dockerfile` 也会在构建阶段强校验这 4 个文件，缺失将直接 `docker build` 失败（避免部署后才暴露问题）。

- `ch_PP-OCRv5_mobile_det.onnx`
- `ch_PP-OCRv5_rec_mobile_infer.onnx`
- `ppocrv5_dict.txt`
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`（即使 `Global.use_cls=false` 仍需预置，避免初始化触发下载路径）

Linux/macOS 下载示例：

```bash
mkdir -p models/rapidocr
curl -L -o models/rapidocr/ch_PP-OCRv5_mobile_det.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"
curl -L -o models/rapidocr/ch_PP-OCRv5_rec_mobile_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx"
curl -L -o models/rapidocr/ppocrv5_dict.txt "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt"
curl -L -o models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
```

Windows PowerShell 下载示例：

```powershell
$dir = "models/rapidocr"
New-Item -ItemType Directory -Path $dir -Force | Out-Null

Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx" -OutFile "$dir/ch_PP-OCRv5_mobile_det.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx" -OutFile "$dir/ch_PP-OCRv5_rec_mobile_infer.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt" -OutFile "$dir/ppocrv5_dict.txt"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx" -OutFile "$dir/ch_ppocr_mobile_v2.0_cls_infer.onnx"
```

下载后请在 `app/config/cfg_openvino_cpu.yaml` 中指定绝对路径（推荐）：

```yaml
Det:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_PP-OCRv5_mobile_det.onnx"
Rec:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_PP-OCRv5_rec_mobile_infer.onnx"
  rec_keys_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ppocrv5_dict.txt"
Cls:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"
```

说明：
- 当前默认配置 `Global.use_cls=false`，即不启用方向分类器（仅使用 `PP-OCRv5 mobile det+rec`）。
- 即使 `Global.use_cls=false`，服务仍会强校验 `ch_ppocr_mobile_v2.0_cls_infer.onnx` 是否存在。

## RapidOCR OpenVINO CPU 配置

示例文件：`app/config/cfg_openvino_cpu.yaml`

当前默认值针对 Intel i7-11800H + OpenVINO CPU 的低延迟场景：

- `performance_hint: LATENCY`
- `inference_num_threads: -1`（交给 OpenVINO 自动决策线程）
- `num_streams: 1`（避免和应用层并发叠加造成过度订阅）
- `enable_cpu_pinning: true`

可用关键参数：

- `inference_num_threads`
- `performance_hint`
- `performance_num_requests`
- `enable_cpu_pinning`
- `num_streams`
- `enable_hyper_threading`
- `scheduling_core_type`

## 许可证

MIT License
