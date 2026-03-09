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

### 最近稳定性修复（2026-03）

- 修复 `/clip/img` 在 OpenVINO 动态输入模型上的 PPP 构建失败：视觉预处理 `resize` 显式固定为 `224x224`，保持输出向量维度 `768` 与接口语义不变。
- 修复 RapidOCR 设备覆盖优先级：显式环境变量（如 `RAPIDOCR_DEVICE=CPU/GPU/AUTO`）优先级高于 `cfg_openvino_cpu.yaml`，避免 YAML 旧值覆盖运行时配置。
- 修复 InsightFace 旧版本兼容：当 `FaceAnalysis.__init__` 不支持 `providers` 参数时，运行时显式强制 `OpenVINOExecutionProvider`，并兼容旧路由器仅加载检测+识别必需模型文件。

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
| `RAPIDOCR_OPENVINO_CONFIG_PATH` | RapidOCR 配置文件路径（YAML）；仅作为基线配置，显式 `RAPIDOCR_*` 环境变量会覆盖对应字段 | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR` | RapidOCR 模型目录（必须包含 `PP-OCRv5 mobile det/rec/dict` 与 `ch_ppocr_mobile_v2.0_cls_infer.onnx`，缺失即失败） | `<repo>/models/rapidocr` |
| `RAPIDOCR_FONT_PATH` | 字体文件路径；空表示不指定 | 空 |
| `RAPIDOCR_DEVICE` | RapidOCR OpenVINO 设备字符串（如 `AUTO` / `GPU` / `AUTO:GPU,CPU`）；`AUTO` 或包含 `GPU` 且无 GPU 设备时会直接报错（无 silent fallback）；显式设置时优先级高于 YAML | `AUTO` |
| `RAPIDOCR_INFERENCE_NUM_THREADS` | 整数，建议 `1~CPU核心数` | `-1`（跟随 `cfg_openvino_cpu.yaml`） |
| `RAPIDOCR_PERFORMANCE_HINT` | OpenVINO 性能提示，常用：`LATENCY` / `THROUGHPUT` | `LATENCY` |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | 整数；`-1` 表示自动 | `-1` |
| `RAPIDOCR_ENABLE_CPU_PINNING` | 布尔：`1/true/yes/on` 或 `0/false/no/off`（大小写不敏感） | `true` |
| `RAPIDOCR_NUM_STREAMS` | 整数；`-1` 表示自动 | `-1`（跟随 `cfg_openvino_cpu.yaml`） |
| `RAPIDOCR_ENABLE_HYPER_THREADING` | 布尔：`1/true/yes/on` 或 `0/false/no/off` | `true` |
| `RAPIDOCR_SCHEDULING_CORE_TYPE` | OpenVINO 核调度类型，常用：`ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY` | `ANY_CORE` |
| `RAPIDOCR_USE_CLS` | 布尔：是否启用方向分类器（`true` 表示开启） | `true` |
| `RAPIDOCR_MAX_SIDE_LEN` | OCR 全图预处理最大边限制（建议 960/1280） | `960` |
| `RAPIDOCR_DET_LIMIT_SIDE_LEN` | 检测模型输入边长限制 | `960` |
| `RAPIDOCR_REC_BATCH_NUM` | 识别批大小（建议 6~8） | `6` |
| `RAPIDOCR_CLS_BATCH_NUM` | 方向分类批大小 | `6` |
| `INSIGHTFACE_OV_DEVICE` | 透传给 ORT OpenVINO EP 的 `device_type`；常见：`CPU_FP32` / `CPU_FP16` / `GPU_FP32` / `GPU_FP16` | `CPU_FP32` |
| `OPENCV_OPENCL_DEVICE` | OpenCV OpenCL 设备选择（可选，如 `Intel:GPU:0`），用于 InsightFace 对齐阶段 OpenCL 路径 | OpenCV 默认设备 |
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
`/clip/img` 端点已去除 `BGR -> RGB -> PIL` 额外拷贝链，模型层直接消费 `numpy BGR`，并使用 OpenVINO PPP 执行 `resize + BGR->RGB + 归一化 + layout` 预处理。
`/represent`（InsightFace）链路使用 OpenCV + OpenCL（Intel 驱动）执行对齐 `warpAffine`；检测/识别输入的归一化与通道转换由 OpenVINO PPP 执行，替代 `cv2.dnn.blobFromImage(s)`。OpenCL 不可用或 OpenVINO EP 不生效会直接报错（无 silent fallback）。
`/ocr` 链路采用 OpenCV `numpy BGR` 零拷贝优先输入（连续 `uint8` 直接透传 RapidOCR，必要时才补齐连续内存），默认开启方向分类器（`use_cls=true`），并使用 `max_side_len=960`、`rec_batch_num=6` 的 11800H 核显基线配置。

## `/dev/dri` 条件专项验收清单

以下清单用于 Debian/Linux Docker 在 Intel iGPU 场景下的上线前验收。

### A. `docker-compose.yml` 必备参数

```yaml
services:
  mt-photos-ai-openvino:
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - "${VIDEO_GID:-44}"
      - "${RENDER_GID:-109}"
    environment:
      - API_AUTH_KEY=mt_photos_ai_extra
      - INFERENCE_DEVICE=GPU
      - CLIP_INFERENCE_DEVICE=GPU
      - RAPIDOCR_DEVICE=GPU
      - INSIGHTFACE_OV_DEVICE=GPU_FP16
      - OPENCV_OPENCL_DEVICE=Intel:GPU:0
      - OV_CACHE_DIR=/models/cache/openvino
      - WEB_CONCURRENCY=2
      - INFERENCE_QUEUE_MAX_SIZE=64
      - TEXT_CLIP_BATCH_SIZE=8
      - TEXT_MODEL_RESTORE_DELAY_MS=2000
```

说明：
- 若需要自动设备选择，可将 `CLIP_INFERENCE_DEVICE/RAPIDOCR_DEVICE` 改为 `AUTO`；但验收时仍要求 GPU 实际可用。
- `VIDEO_GID/RENDER_GID` 必须与宿主机 `video/render` 组一致。

### B. 启动与设备可见性判定

通过标准：
- `docker compose ps` 显示容器 `Up`。
- `docker inspect <container> --format '{{.State.Health.Status}}'` 返回 `healthy`。
- 启动日志包含“启动自检通过：GPU 设备节点可访问”。
- 容器内 `ls -l /dev/dri` 可见 `card*` 与 `renderD*` 节点。
- 容器用户对 `/dev/dri` 节点具备读写权限（无 permission denied）。

失败判定（任一命中即失败）：
- 日志出现“已请求 GPU 推理，但容器内不存在 /dev/dri”。
- 日志出现“/dev/dri 未发现 card*/renderD* 节点”。
- 日志出现“/dev/dri 设备权限不足”。

### C. 端点语义与后端判定

通过标准：
- `POST /check` 返回 `{"result":"pass", ...}`。
- `POST /clip/txt` 成功返回 `768` 维字符串数组（成功无 `msg`）。
- `POST /clip/img` 成功返回 `768` 维字符串数组（成功无 `msg`）。
- `POST /ocr` 成功返回 `{"result":{"texts","scores","boxes"}}`（成功无 `msg`）。
- `POST /represent` 返回 `{"detector_backend":"insightface","recognition_model":"antelopev2","result":[...]}` 或在人脸不存在时 `result=[]`。

失败判定（任一命中即失败）：
- 日志出现 `No silent fallback is allowed`（排除你主动构造失败用例）。
- 日志出现 `OpenCL is unavailable` 或非 Intel OpenCL 设备错误。
- 日志出现 `provider validation failed` 或 `OpenVINOExecutionProvider` 非首位。

### D. 回归脚本建议（最小）

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=200 mt-photos-ai-openvino

curl -s http://127.0.0.1:8060/
curl -s -X POST http://127.0.0.1:8060/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8060/clip/txt -H "api-key: mt_photos_ai_extra" -H "Content-Type: application/json" -d '{"text":"smoke"}'
```

## Windows Server GPU-PV（`/dev/dxg`）能力边界

在 Windows Server + Linux 容器 + GPU-PV（映射 `/dev/dxg`）场景下，当前服务可启用的加速项：

- QA-CLIP OpenVINO GPU 推理（`CLIP_INFERENCE_DEVICE=AUTO/GPU`）；
- QA-CLIP GPU Remote Context + host tensor 互操作（减少 Host<->Device 拷贝）；
- InsightFace 的 OpenVINO EP（当 `INSIGHTFACE_OV_DEVICE` 配置为 GPU 类型且运行时可用），并配合 OpenCV(OpenCL) 对齐 + OpenVINO PPP 预处理链路。

需要明确的限制：

- RapidOCR 默认 OpenVINO `AUTO`（可显式 `GPU`）；若请求 `AUTO/GPU` 但运行时无 GPU 设备，会直接报错而不是回退到纯 CPU。
- `ffmpeg QSV` 在 Linux 容器里通常依赖 `/dev/dri`（VAAPI/oneVPL）链路；仅有 `/dev/dxg` 时不保证可用。若 QSV 不可用，服务会记录原因并按既定顺序显式降级到 `ffmpeg(CPU)` 或 `cv2.imdecode`（无静默 fallback）。

## RapidOCR 模型预置（强制）

当前服务依赖 `rapidocr==3.7.0`。

服务已禁用线上下载回退，必须在镜像/部署目录预置以下 4 个本地文件（缺失即失败）：

当前 `Dockerfile` 也会在构建阶段强校验这 4 个文件，缺失将直接 `docker build` 失败（避免部署后才暴露问题）。

- `ch_PP-OCRv5_mobile_det.onnx`
- `ch_PP-OCRv5_rec_mobile_infer.onnx`
- `ppocrv5_dict.txt`
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`（即使 `Global.use_cls=false` 仍需预置，避免初始化触发下载路径）

Linux/macOS 下载示例：

```bash
mkdir -p models/rapidocr
curl -L -o models/rapidocr/ch_PP-OCRv5_mobile_det.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"
curl -L -o models/rapidocr/ch_PP-OCRv5_rec_mobile_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx"
curl -L -o models/rapidocr/ppocrv5_dict.txt "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt"
curl -L -o models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
```

Windows PowerShell 下载示例：

```powershell
$dir = "models/rapidocr"
New-Item -ItemType Directory -Path $dir -Force | Out-Null

Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx" -OutFile "$dir/ch_PP-OCRv5_mobile_det.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx" -OutFile "$dir/ch_PP-OCRv5_rec_mobile_infer.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt" -OutFile "$dir/ppocrv5_dict.txt"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx" -OutFile "$dir/ch_ppocr_mobile_v2.0_cls_infer.onnx"
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
- 当前默认配置 `Global.use_cls=true`，启用方向分类器以处理拍照文本旋转场景。
- 即使你手动关闭 `Global.use_cls`，服务仍会强校验 `ch_ppocr_mobile_v2.0_cls_infer.onnx` 是否存在（避免初始化路径触发下载）。

## RapidOCR OpenVINO(AUTO/GPU) 配置

示例文件：`app/config/cfg_openvino_cpu.yaml`

当前默认值针对 Intel i7-11800H + 核显场景：

- `device_name: AUTO`
- `performance_hint: LATENCY`
- `inference_num_threads: -1`（交给 OpenVINO 自动决策线程）
- `num_streams: -1`（由 OpenVINO 自动调度）
- `max_side_len: 960`
- `det_limit_side_len: 960`
- `rec_batch_num: 6`
- `use_cls: true`

配置优先级（运行时）：
- 显式环境变量（`RAPIDOCR_*`） > `cfg_openvino_cpu.yaml` > 代码默认值。
- 仅当环境变量被显式设置时才覆盖 YAML（避免“空值覆盖”）。

可用关键参数：

- `device_name`
- `inference_num_threads`
- `performance_hint`
- `performance_num_requests`
- `enable_cpu_pinning`
- `num_streams`
- `enable_hyper_threading`
- `scheduling_core_type`

## 许可证

MIT License
