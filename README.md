# MT-Photos AI (OpenVINO)

统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）的 FastAPI 服务。本文档仅保留部署相关内容：Windows 本机部署与 Linux Docker 部署。

## 部署前准备

- Python **3.12**
- 已准备模型目录（至少包含）：
  - `models/qa-clip/openvino`
  - `models/insightface/models`
  - `models/rapidocr`（建议提前下载，避免首次请求在线拉取）
- 服务入口：`app/server.py`

## 环境变量（运行时，全量）

| 环境变量 | 可选值 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | 任意字符串；`no-key` 或空字符串表示关闭鉴权 | `mt_photos_ai_extra` |
| `INFERENCE_DEVICE` | OpenVINO 设备字符串，如 `GPU` / `CPU` / `AUTO` / `AUTO:GPU,CPU` | `AUTO` |
| `CLIP_INFERENCE_DEVICE` | 同 `INFERENCE_DEVICE`；仅覆盖 QA-CLIP 设备 | 跟随 `INFERENCE_DEVICE` |
| `MODEL_PATH` | 模型根目录路径 | `<repo>/models` |
| `MODEL_NAME` | InsightFace 模型目录名，如 `antelopv2` / `buffalo_l` | `antelopv2` |
| `WEB_CONCURRENCY` | 整数，建议 `>=1` | `2` |
| `INFERENCE_QUEUE_MAX_SIZE` | 整数；运行时会被约束为 `>=1` | `64` |
| `TEXT_CLIP_BATCH_SIZE` | 整数；运行时会被约束为 `>=1` | `8` |
| `INFERENCE_TASK_TIMEOUT` | 整数秒；运行时会被约束为 `>=1` | `120` |
| `SERVER_IDLE_TIMEOUT` | 整数秒；`<=0` 表示禁用空闲释放计时器 | `300` |
| `TEXT_MODEL_RESTORE_DELAY_MS` | 整数毫秒；运行时会被约束为 `>=0` | `2000` |
| `RESTART_TEXT_RESTORE_DELAY_MS` | 旧变量，仅在未设置 `TEXT_MODEL_RESTORE_DELAY_MS` 时生效 | `2000` |
| `OV_CACHE_DIR` | OpenVINO 编译缓存目录路径 | `<repo>/cache/openvino` |
| `RAPIDOCR_OPENVINO_CONFIG_PATH` | RapidOCR 配置文件路径（YAML） | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR` | RapidOCR 模型目录（用于兼容不同 RapidOCR 构造参数） | `<repo>/models/rapidocr` |
| `RAPIDOCR_FONT_PATH` | 字体文件路径；空表示不指定 | 空 |
| `RAPIDOCR_INFERENCE_NUM_THREADS` | 整数，建议 `1~CPU核心数` | `8` |
| `RAPIDOCR_PERFORMANCE_HINT` | OpenVINO 性能提示，常用：`LATENCY` / `THROUGHPUT` | `LATENCY` |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | 整数；`-1` 表示自动 | `-1` |
| `RAPIDOCR_ENABLE_CPU_PINNING` | 布尔：`1/true/yes/on` 或 `0/false/no/off`（大小写不敏感） | `true` |
| `RAPIDOCR_NUM_STREAMS` | 整数；`-1` 表示自动 | `-1` |
| `RAPIDOCR_ENABLE_HYPER_THREADING` | 布尔：`1/true/yes/on` 或 `0/false/no/off` | `true` |
| `RAPIDOCR_SCHEDULING_CORE_TYPE` | OpenVINO 核调度类型，常用：`ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY` | `ANY_CORE` |
| `INSIGHTFACE_OV_DEVICE` | 透传给 ORT OpenVINO EP 的 `device_type`；常见：`CPU_FP32` / `CPU_FP16` / `GPU_FP32` / `GPU_FP16` | `CPU_FP32` |
| `PORT` | 端口号（整数） | `8060` |
| `LOG_LEVEL` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` | `WARNING` |

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

## Linux Docker 部署

### 方式一：docker compose（推荐）

1. 准备配置文件：

```bash
cp docker-compose.example.yml docker-compose.yml
```

2. 编辑 `docker-compose.yml`：
- 设置 `API_AUTH_KEY`
- 根据机器情况设置 `INFERENCE_DEVICE`（有 Intel iGPU 且已映射 `/dev/dri` 时可用 `GPU`）
- 按需调整并发与缓存目录

3. 启动：

```bash
docker compose up -d --build
```

4. 查看状态：

```bash
docker compose ps
docker compose logs -f mt-photos-ai-openvino
```

### 方式二：docker run

```bash
docker build -t mt-photos-ai-openvino .
docker run -d \
  --name mt-photos-ai-openvino \
  -p 8060:8060 \
  --device /dev/dri:/dev/dri \
  -v $(pwd)/models:/models \
  -v $(pwd)/app/config/cfg_openvino_cpu.yaml:/app/config/cfg_openvino_cpu.yaml:ro \
  -e API_AUTH_KEY=your_secret_key \
  -e INFERENCE_DEVICE=GPU \
  -e WEB_CONCURRENCY=2 \
  -e OV_CACHE_DIR=/models/cache/openvino \
  -e RAPIDOCR_OPENVINO_CONFIG_PATH=/app/config/cfg_openvino_cpu.yaml \
  -e RAPIDOCR_MODEL_DIR=/models/rapidocr \
  mt-photos-ai-openvino
```

## RapidOCR 模型下载

### 方式一：自动下载（最省事）

保持 `app/config/cfg_openvino_cpu.yaml` 里的 `Det/Cls/Rec.model_path` 为 `null`，首次调用 `/ocr` 时会自动下载模型到 RapidOCR 默认目录（通常是 Python 站点包目录下的 `rapidocr/models`）。

### 方式二：离线预下载（推荐用于生产/镜像）

本仓库当前配置使用 `PP-OCRv4 + server`，至少需要以下 4 个文件：

- `ch_PP-OCRv4_det_server_infer.onnx`
- `ch_ppocr_mobile_v2.0_cls_infer.onnx`
- `ch_PP-OCRv4_rec_server_infer.onnx`
- `ppocr_keys_v1.txt`

Linux/macOS 下载示例：

```bash
mkdir -p models/rapidocr
curl -L -o models/rapidocr/ch_PP-OCRv4_det_server_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_server_infer.onnx"
curl -L -o models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
curl -L -o models/rapidocr/ch_PP-OCRv4_rec_server_infer.onnx "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_server_infer.onnx"
curl -L -o models/rapidocr/ppocr_keys_v1.txt "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"
```

Windows PowerShell 下载示例：

```powershell
$dir = "models/rapidocr"
New-Item -ItemType Directory -Path $dir -Force | Out-Null

Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_server_infer.onnx" -OutFile "$dir/ch_PP-OCRv4_det_server_infer.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx" -OutFile "$dir/ch_ppocr_mobile_v2.0_cls_infer.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_server_infer.onnx" -OutFile "$dir/ch_PP-OCRv4_rec_server_infer.onnx"
Invoke-WebRequest "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.6.0/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt" -OutFile "$dir/ppocr_keys_v1.txt"
```

下载后请在 `app/config/cfg_openvino_cpu.yaml` 中指定绝对路径（推荐）：

```yaml
Det:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_PP-OCRv4_det_server_infer.onnx"
Cls:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"
Rec:
  model_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ch_PP-OCRv4_rec_server_infer.onnx"
  rec_keys_path: "D:/path/to/mt-photos-ai-openvino/models/rapidocr/ppocr_keys_v1.txt"
```

## RapidOCR OpenVINO CPU 配置

示例文件：`app/config/cfg_openvino_cpu.yaml`

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
