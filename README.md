# MT-Photos AI (OpenVINO)

统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）的 FastAPI 服务。本文档仅保留最终用户部署、运行和配置说明。

## 部署前准备

- Python **3.12**
- 已准备模型目录（至少包含）：
  - `models/qa-clip/openvino`
  - `models/insightface/models`
  - `models/rapidocr`（需预置 PP-OCRv5 mobile det/rec/dict + cls 本地文件）
- 服务入口：`app/server.py`

## 运行时环境变量

| 环境变量                                | 说明                                                        | 默认值                                |
|-------------------------------------|-----------------------------------------------------------|------------------------------------|
| `API_AUTH_KEY`                      | API Key；`no-key` 或空字符串表示关闭鉴权                              | `mt_photos_ai_extra`               |
| `INFERENCE_DEVICE`                  | OpenVINO 设备字符串，如 `GPU` / `CPU` / `AUTO` / `AUTO:GPU,CPU`  | `AUTO`                             |
| `CLIP_INFERENCE_DEVICE`             | 仅覆盖 QA-CLIP 设备；请求 `AUTO/GPU` 时需保证 GPU 可用                  | 跟随 `INFERENCE_DEVICE`              |
| `MODEL_PATH`                        | 模型根目录路径                                                   | `<repo>/models`                    |
| `WEB_CONCURRENCY`                   | Uvicorn worker 数                                          | `2`                                |
| `INFERENCE_QUEUE_MAX_SIZE`          | 推理队列长度                                                    | `64`                               |
| `TEXT_CLIP_BATCH_SIZE`              | 文本 CLIP 批大小                                               | `8`                                |
| `INFERENCE_TASK_TIMEOUT`            | 单任务超时时间（秒）                                                | `120`                              |
| `SERVER_IDLE_TIMEOUT`               | 空闲释放时间（秒）；`<=0` 表示禁用                                      | `300`                              |
| `TEXT_MODEL_RESTORE_DELAY_MS`       | 非文本任务结束后的 Text-CLIP 恢复延迟（毫秒）                              | `2000`                             |
| `RESTART_TEXT_RESTORE_DELAY_MS`     | 兼容旧变量；仅在未设置 `TEXT_MODEL_RESTORE_DELAY_MS` 时生效             | `2000`                             |
| `OV_CACHE_DIR`                      | OpenVINO 编译缓存目录                                           | `<repo>/cache/openvino`            |
| `RAPIDOCR_OPENVINO_CONFIG_PATH`     | RapidOCR YAML 配置文件路径                                      | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR`                | RapidOCR 模型目录                                             | `<repo>/models/rapidocr`           |
| `RAPIDOCR_FONT_PATH`                | RapidOCR 字体文件路径；空表示不指定                                    | 空                                  |
| `RAPIDOCR_DEVICE`                   | RapidOCR OpenVINO 设备字符串；请求 `AUTO/GPU` 时需保证 GPU 可用         | `AUTO`                             |
| `RAPIDOCR_INFERENCE_NUM_THREADS`    | RapidOCR 推理线程数                                            | `-1`                               |
| `RAPIDOCR_PERFORMANCE_HINT`         | OpenVINO 性能提示，如 `LATENCY` / `THROUGHPUT`                  | `LATENCY`                          |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | OpenVINO 请求数；`-1` 表示自动                                    | `-1`                               |
| `RAPIDOCR_ENABLE_CPU_PINNING`       | 是否启用 CPU 绑核                                               | `true`                             |
| `RAPIDOCR_NUM_STREAMS`              | OpenVINO stream 数；`-1` 表示自动                               | `-1`                               |
| `RAPIDOCR_ENABLE_HYPER_THREADING`   | 是否启用超线程                                                   | `true`                             |
| `RAPIDOCR_SCHEDULING_CORE_TYPE`     | OpenVINO 核调度类型，如 `ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY` | `ANY_CORE`                         |
| `RAPIDOCR_USE_CLS`                  | 是否启用方向分类器                                                 | `true`                             |
| `RAPIDOCR_MAX_SIDE_LEN`             | OCR 全图最大边限制                                               | `960`                              |
| `RAPIDOCR_DET_LIMIT_SIDE_LEN`       | 检测模型输入边长限制                                                | `960`                              |
| `RAPIDOCR_REC_BATCH_NUM`            | 识别批大小                                                     | `6`                                |
| `RAPIDOCR_CLS_BATCH_NUM`            | 方向分类批大小                                                   | `6`                                |
| `INSIGHTFACE_OV_DEVICE`             | ORT OpenVINO EP `device_type`，如 `CPU_FP32` / `GPU_FP16`   | `CPU_FP32`                         |
| `OPENCV_OPENCL_DEVICE`              | OpenCV OpenCL 设备选择，如 `Intel:GPU:0`                        | OpenCV 默认设备                        |
| `PORT`                              | 服务端口                                                      | `8060`                             |
| `LOG_LEVEL`                         | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`  | `WARNING`                          |

补充说明：

- 当 `CLIP_INFERENCE_DEVICE` 请求 `GPU` 或 `AUTO` 时，服务会强制初始化 OpenVINO GPU Remote Context。
- Remote Context 初始化会依次尝试默认 `GPU`、具体 `GPU.*` 设备，以及 `create_context("GPU", {})` 兼容路径；全部失败时直接终止启动，不允许 silent fallback。

## Windows 本机部署

1. 确认 Python 版本：

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

4. （可选）设置环境变量：

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

## Debian/Linux Docker 部署

推荐宿主环境：

- Debian 13（或兼容发行版）
- Docker Engine 24+ 与 Docker Compose v2
- Intel iGPU 场景下，宿主机可见 `/dev/dri`

如需在 Debian 13 宿主或自定义镜像中补齐 Intel Xe 依赖，可参考：

```bash
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  libdrm2 \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libze1 \
  ocl-icd-libopencl1 \
  mesa-opencl-icd
```

说明：

- 参考 OpenVINO 与 Intel GPU 官方文档，容器内 OpenVINO GPU 运行时需要 `intel-opencl-icd` + Level Zero 运行库（Debian 包名 `libze-intel-gpu1`），以及 `libze1`/`ocl-icd-libopencl1`。
- 当前镜像构建阶段会临时启用 sid 源，仅安装 `intel-opencl-icd` 与 `libze-intel-gpu1`，随后清理 sid 源和 pin 文件。
- 服务上传读图链已统一改为 OpenCV 原生解码，镜像不再包含 `ffmpeg/ffprobe`、VAAPI/oneVPL/QSV 媒体栈，也不预装 `clinfo` 这类诊断工具。
- 若服务日志出现 `available_devices=['CPU']`，即使 `/dev/dri` 可见，也通常意味着容器里缺少可用的 GPU OpenCL runtime，或 `/dev/dri` 并非真实的 Intel DRM render node。
- 参考文档：
  - OpenVINO GPU 设备配置与依赖：<https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>
  - Intel Linux GPU Driver（OpenCL/Level Zero 运行时包）：<https://dgpu-docs.intel.com/driver/installation.html>

### 方式一：docker compose

1. 准备配置文件，并确认 `docker-compose.yml` 里的 `image:` 已指向你实际要使用的镜像标签：

```bash
cp docker-compose.example.yml docker-compose.yml
```

2. 按宿主机实际情况修改 `group_add` 中的 `video` / `render` GID（Debian 默认通常是 `44` / `109`）。

3. 按需调整 `docker-compose.yml`：

- 生产环境建议覆盖 `API_AUTH_KEY`
- 有 Intel iGPU 且已映射 `/dev/dri` 时，建议使用 `INFERENCE_DEVICE=AUTO`、`CLIP_INFERENCE_DEVICE=AUTO`、`RAPIDOCR_DEVICE=AUTO`（若 GPU 不可用会按规则硬失败）
- 如需挂载自定义模型、RapidOCR 配置或缓存目录，可再调整 `MODEL_PATH`、`RAPIDOCR_MODEL_DIR`、`RAPIDOCR_OPENVINO_CONFIG_PATH`、`OV_CACHE_DIR`

4. 启动服务：

```bash
docker compose up -d
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
  -e INFERENCE_DEVICE=AUTO \
  -e CLIP_INFERENCE_DEVICE=AUTO \
  -e RAPIDOCR_DEVICE=AUTO \
  -e WEB_CONCURRENCY=2 \
  -e OV_CACHE_DIR=/models/cache/openvino \
  mt-photos-ai-openvino
```

### 容器内设备检查

首次部署建议执行以下检查：

```bash
docker exec -it mt-photos-ai-openvino ls -l /dev/dri
docker exec -it mt-photos-ai-openvino python -c "import openvino as ov; print(ov.Core().available_devices)"
docker exec -it mt-photos-ai-openvino python -c 'import cv2; cv2.ocl.setUseOpenCL(True); d=cv2.ocl.Device_getDefault(); print(cv2.ocl.haveOpenCL(), cv2.ocl.useOpenCL(), d.vendorName(), d.name())'
```

若请求了 GPU 推理但容器内 GPU 设备不可用，服务会直接报错并终止启动。

## Windows Server GPU-PV（`/dev/dxg`）说明

在 Windows Server + Linux 容器 + GPU-PV 场景下，QA-CLIP 与 InsightFace 的 OpenVINO GPU 推理仍以当前容器运行时能力为准。

需要注意：

- 当前服务已不依赖 `ffmpeg/QSV` 做上传解码
- 当前 Debian/Linux 容器启动自检与验收仍以 `/dev/dri` 为准；仅有 `/dev/dxg` 不视为满足当前镜像的 GPU 启动条件
- 若 `get_default_context("GPU")` 因插件上下文初始化时机失败，服务会自动重试具体 `GPU.*` 设备与 `create_context("GPU", {})`；若仍失败，则保持硬失败并退出启动

## RapidOCR 模型预置

部署前请在 `models/rapidocr` 预置以下 4 个文件：

- `ch_PP-OCRv5_mobile_det.onnx`
- `ch_PP-OCRv5_rec_mobile_infer.onnx`
- `ppocrv5_dict.txt`
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`

缺失任一文件都会导致构建或启动失败。

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

## 冒烟检查

服务启动后，可先检查：

```bash
curl -s http://127.0.0.1:8060/
curl -s -X POST http://127.0.0.1:8060/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8060/clip/txt -H "api-key: mt_photos_ai_extra" -H "Content-Type: application/json" -d '{"text":"smoke"}'
```

除 `GET /` 外，业务端点在启用鉴权时都需要携带 `api-key` 请求头。

## 许可证

MIT License
