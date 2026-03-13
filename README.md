# MT-Photos AI (OpenVINO)

统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）的 FastAPI 服务。本文档仅保留最终用户部署、运行和配置说明。

## 部署前准备

- Python **3.12**
- 已准备模型目录（至少包含）：
  - `models/qa-clip/openvino`
  - `models/insightface/models/antelopev2`（至少保留 `scrfd_10g_bnkps.onnx` 与 `glintr100.onnx`）
  - `models/rapidocr`（需预置 PP-OCRv5 mobile det/rec/dict + cls 本地文件）
- 服务入口：`app/server.py`
- QA-CLIP 离线转换脚本：`convert/convert.py`
- 源码运行时需保留 `app/models/QA-CLIP/clip`，服务当前仍使用其中的 tokenizer 与词表资源

## 运行时环境变量

| 环境变量                                | 说明                                                        | 默认值                                |
|-------------------------------------|-----------------------------------------------------------|------------------------------------|
| `API_AUTH_KEY`                      | API Key；`no-key` 或空字符串表示关闭鉴权                              | `mt_photos_ai_extra`               |
| `INFERENCE_DEVICE`                  | OpenVINO 设备字符串，如 `GPU` / `CPU` / `AUTO`                 | `AUTO`                             |
| `CLIP_INFERENCE_DEVICE`             | 仅覆盖 QA-CLIP 设备；请求 `AUTO/GPU` 时需保证 GPU 可用，且会强制初始化 GPU Remote Context | 跟随 `INFERENCE_DEVICE`              |
| `MODEL_PATH`                        | 模型根目录路径                                                   | `<repo>/models`                    |
| `INFERENCE_QUEUE_MAX_SIZE`          | 跨 `/clip/img`、`/ocr`、`/represent` 共享的图片请求总名额（排队+执行）；运行时硬上限 `10`，超出会立即拒绝 | `10`                               |
| `INFERENCE_TASK_TIMEOUT`            | 兼容旧配置的总超时基线；未显式设置新变量时用作排队超时默认值，并为执行超时提供下限 | `10`                               |
| `INFERENCE_QUEUE_TIMEOUT`           | 非文本任务排队超时（秒）                                               | 跟随 `INFERENCE_TASK_TIMEOUT`       |
| `INFERENCE_EXEC_TIMEOUT`            | 非文本任务执行超时（秒）；默认至少 `30` 秒                              | `max(30, INFERENCE_TASK_TIMEOUT)`  |
| `OCR_EXEC_TIMEOUT`                  | 仅覆盖 OCR 推理执行超时（秒）；模型加载不计入该窗口，未设置时默认至少 `30` 秒       | `max(30, INFERENCE_EXEC_TIMEOUT)`  |
| `OV_CACHE_DIR`                      | 可选：自定义 OpenVINO 编译缓存目录；未设置时默认使用 `<repo>/cache/openvino`     | `<repo>/cache/openvino`             |
| `CLIP_IMAGE_BATCH`                  | `/clip/img` 在标准预处理完成后的批大小上限                                   | `8`                                 |
| `CLIP_IMAGE_BATCH_WAIT_MS`          | `/clip/img` 微批等待窗口（毫秒）                                          | `5`                                 |
| `RAPIDOCR_OPENVINO_CONFIG_PATH`     | RapidOCR YAML 配置文件路径；服务会将该文件直接作为 `config_path` 传给 `RapidOCR` | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR`                | RapidOCR 模型目录                                             | `<repo>/models/rapidocr`           |
| `RAPIDOCR_FONT_PATH`                | RapidOCR 字体文件路径；空表示不指定                                    | 空                                  |
| `RAPIDOCR_DEVICE`                   | RapidOCR OpenVINO 后端请求设备；当前实现会固定收敛到库内原生 `CPU` 路径，非 `CPU` 值仅记录告警 | `CPU`                              |
| `RAPIDOCR_DET_DEVICE`               | 保留兼容环境变量；当前库内原生 CPU 路径不会使用该 stage 覆盖                | 未使用                               |
| `RAPIDOCR_CLS_DEVICE`               | 保留兼容环境变量；当前库内原生 CPU 路径不会使用该 stage 覆盖                | 未使用                               |
| `RAPIDOCR_REC_DEVICE`               | 保留兼容环境变量；当前库内原生 CPU 路径不会使用该 stage 覆盖                | 未使用                               |
| `RAPIDOCR_INFERENCE_NUM_THREADS`    | RapidOCR 推理线程数                                            | `-1`                               |
| `RAPIDOCR_PERFORMANCE_HINT`         | OpenVINO 性能提示，如 `LATENCY` / `THROUGHPUT`                  | `THROUGHPUT`                       |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | RapidOCR OpenVINO CPU 配置项；传给库内 OpenVINO 配置，当前不再驱动自定义 stage executor | `2`                                |
| `RAPIDOCR_ENABLE_CPU_PINNING`       | 是否启用 CPU 绑核                                               | `true`                             |
| `RAPIDOCR_NUM_STREAMS`              | OpenVINO stream 数                                             | `2`                                |
| `RAPIDOCR_ENABLE_HYPER_THREADING`   | 是否启用超线程                                                   | `true`                             |
| `RAPIDOCR_SCHEDULING_CORE_TYPE`     | OpenVINO 核调度类型，如 `ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY` | `ANY_CORE`                         |
| `RAPIDOCR_USE_CLS`                  | 是否启用方向分类器                                                 | `true`                             |
| `RAPIDOCR_MAX_SIDE_LEN`             | OCR 全图最大边限制                                               | `960`                              |
| `RAPIDOCR_DET_LIMIT_SIDE_LEN`       | 检测模型输入边长限制                                                | `960`                              |
| `RAPIDOCR_DET_LIMIT_TYPE`           | 检测缩放策略；默认 `max`，避免把小图放大到阈值导致时延异常                | `max`                              |
| `RAPIDOCR_REC_BATCH_NUM`            | 识别批大小                                                     | `8`                                |
| `RAPIDOCR_CLS_BATCH_NUM`            | 方向分类批大小                                                   | `8`                                |
| `OCR_MAX_CONCURRENT_REQUESTS`       | OCR 应用层最大并发请求数；用于限制积压且不会超过共享图片总名额 | `min(INFERENCE_QUEUE_MAX_SIZE, max(2, RAPIDOCR_PERFORMANCE_NUM_REQUESTS*2))` |
| `OCR_PREWARM_ENABLED`               | 是否启用一次性后台 RapidOCR 预热；预热完成后会立即释放 OCR 模型                 | `false`                             |
| `OCR_PREWARM_DELAY_SECONDS`         | RapidOCR 一次性后台预热延迟（秒）；仅在 `OCR_PREWARM_ENABLED=true` 时生效      | `1.0`                               |
| `NON_TEXT_IDLE_RELEASE_SECONDS`     | 60 秒未收到业务请求时自动释放 Vision-CLIP / OCR / InsightFace；`<=0` 表示关闭 | `60`                                |
| `INSIGHTFACE_OV_DEVICE`             | ORT OpenVINO EP `device_type`；`AUTO` 在 GPU 可见时会优先解析为 `GPU`     | `AUTO`                             |
| `INSIGHTFACE_OV_ENABLE_OPENCL_THROTTLING` | 是否启用 OpenVINO EP 的 OpenCL 节流；吞吐优先场景建议关闭                  | `false`                            |
| `INSIGHTFACE_OV_NUM_THREADS`        | InsightFace OpenVINO EP CPU 线程数；`-1` 表示使用运行时默认值               | `-1`                               |
| `INSIGHTFACE_MAX_WORKERS`           | `/represent` 应用层 worker 数；默认允许有限并发，避免单 worker 头阻塞        | `2`                                |
| `INSIGHTFACE_MAX_CONCURRENT_REQUESTS` | `/represent` 应用层最大并发请求数；用于限制 executor 积压，且不会超过共享图片总名额 | `min(INFERENCE_QUEUE_MAX_SIZE, max(2, INSIGHTFACE_MAX_WORKERS*2))` |
| `OPENCV_OPENCL_DEVICE`              | OpenCV OpenCL 设备选择，如 `Intel:GPU:0`                        | OpenCV 默认设备                        |
| `PORT`                              | 服务端口                                                      | `8060`                             |
| `LOG_LEVEL`                         | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`  | `WARNING`                          |

补充说明：

- 开发机本地验证时，建议把所有后端统一设为 `CPU`：`INFERENCE_DEVICE=CPU`、`CLIP_INFERENCE_DEVICE=CPU`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=CPU`。`RAPIDOCR_DET_DEVICE/RAPIDOCR_CLS_DEVICE/RAPIDOCR_REC_DEVICE` 当前仅保留兼容占位，不参与运行时选路。
- 当 `CLIP_INFERENCE_DEVICE` 请求 `GPU` 或 `AUTO` 时，服务会强制初始化 OpenVINO GPU Remote Context。
- Remote Context 初始化会依次尝试默认 `GPU`、具体 `GPU.*` 设备，以及 `create_context("GPU", {})` 兼容路径；全部失败时直接终止启动，不允许 silent fallback。
- Text-CLIP 常驻在单线程后台服务中，始终复用单个模型实例。
- Vision-CLIP / OCR / InsightFace 采用“单活非文本模型族”切换策略：切换到新模型族前，会先等待当前已受理任务退场并同步释放旧族模型，避免三个大模型长期同时驻留。
- `/clip/img` 会先执行标准预处理（缩放、中心裁剪、PPP 归一化），再按 `CLIP_IMAGE_BATCH` 聚合成批并通过 `np.stack` 一次送入动态 batch 视觉模型。
- 为避免 MT-Photos 客户端在积压时主动取消，服务会对 `/clip/img`、`/ocr`、`/represent` 共享一个图片请求名额池；默认值和运行时硬上限都是 `10`，第 `11` 张会立即失败而不是继续挂起等待。
- RapidOCR 现已回到库内原生 `RapidOCR.__call__` CPU 执行链：服务只负责懒加载、应用层准入和超时控制，不再替换 session、也不再维护自定义 stage 级预处理/批调度。
- 非文本超时仍拆分为“排队超时”和“执行超时”；`/clip/img` 使用有界批队列，OCR/InsightFace 使用各自独立执行器；OCR/Face 的异步路径会先完成模型加载，再进入执行超时窗口。
- RapidOCR 会直接加载 `RAPIDOCR_OPENVINO_CONFIG_PATH` 指向的 YAML，并额外校验 `Det/Cls/Rec.engine_type=openvino`，避免回落到默认 ORT 配置。
- RapidOCR 当前基线为 `OpenVINO + CPU + PP-OCRv5 mobile + use_cls=true`；即使显式传入 `RAPIDOCR_DEVICE=AUTO/GPU` 或 stage 级设备变量，运行时也会告警并强制回到 CPU。
- RapidOCR 默认基线使用 `Det.limit_type=max`；若配置成 `min`，小图会被放大到 `limit_side_len`，通常会明显拉高检测时延。
- RapidOCR 慢请求日志现输出总耗时和当前 CPU 配置，便于区分是 OCR 本体慢还是排队/模型切换导致的尾延迟。
- 服务现在默认启用本地 OpenVINO 编译缓存目录 `<repo>/cache/openvino`；如果需要自定义路径，可显式设置 `OV_CACHE_DIR`。
- 默认不会在启动后把 OCR 拉入内存；OCR 会在首次 `/ocr` 请求时懒加载。若显式开启 `OCR_PREWARM_ENABLED=true`，服务仅做一次性预热并立即释放 OCR，不会让 OCR 常驻。
- 默认还会在连续 `60s` 未收到业务请求时自动释放 Vision-CLIP / OCR / InsightFace，只保留常驻 Text-CLIP；如需调整可设置 `NON_TEXT_IDLE_RELEASE_SECONDS`。
- `POST /restart` 会同步等待当前非文本任务退场并释放 Vision-CLIP / OCR / InsightFace；返回 `{"result":"pass"}` 时本轮释放已经完成。
- InsightFace 在 GPU 可见时会把 `INSIGHTFACE_OV_DEVICE=AUTO` 收敛为 `GPU`，并额外把 PPP 预处理编译到同一运行时设备；日志会输出 `configured_device`、`runtime_device`、`provider_runtime` 和 `ppp_execution_devices` 便于确认没有落回 CPU。
- 服务会兼容旧版 `insightface` 对 `providers` / `allowed_modules` 构造参数的不同行为：若构造阶段不接受这些参数，会自动改为兼容实例化并在 session 级强制设置 `OpenVINOExecutionProvider`，不会因为包版本差异静默回退到 CPUExecutionProvider。
- `/represent` 默认不再固定单 worker；服务会以有限 worker 并发执行人脸检测/对齐/识别，并在应用层限制最大并发请求数，减少头阻塞和后台队列堆积。

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

3. （可选）设置环境变量：

```powershell
$env:API_AUTH_KEY="your_secret_key"
$env:INFERENCE_DEVICE="CPU"
$env:CLIP_INFERENCE_DEVICE="CPU"
$env:CLIP_IMAGE_BATCH="8"
$env:RAPIDOCR_DEVICE="CPU"
$env:INSIGHTFACE_OV_DEVICE="CPU"
```

4. 启动服务：

```powershell
cd app
uvicorn server:app --host 0.0.0.0 --port 8060
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
  libglib2.0-0 \
  libgomp1 \
  libze1 \
  ocl-icd-libopencl1 \
  mesa-opencl-icd
```

说明：

- 参考 OpenVINO 与 Intel GPU 官方文档，容器内 OpenVINO GPU 运行时需要 `intel-opencl-icd` + Level Zero 运行库（Debian 包名 `libze-intel-gpu1`），以及 `libze1`/`ocl-icd-libopencl1`。
- 当前 Docker 构建会在 `pip install -r requirements.txt` 后移除传递安装的 `opencv-python`/`opencv-contrib-python`，再补装 `opencv-python-headless`；当前服务只使用 OpenCV 的图像解码、色彩转换和 OpenCL/`warpAffine` 路径，不需要 X11/OpenGL 运行时包。
- 因此当前运行时镜像不再包含 `libgl1`、`libsm6`、`libxext6`、`libxrender1`，也不再打包 `mesa-vulkan-drivers`、`intel-media-va-driver-non-free`。
- Intel GPU 固件属于宿主机职责；若宿主 Debian 13 需要补齐固件，请在宿主机安装 `firmware-misc-nonfree`（或兼容包名 `firmware-misc-non-free`），而不是放进应用容器。
- Dockerfile 已固化为清华 APT + 清华 PyPI 镜像；APT 基础列表、sid pin 文件和安装包名单都直接写在仓库文件中，便于回溯与审计。
- Dockerfile 使用 BuildKit cache mount 复用 `apt`/`pip` 下载缓存；在 Docker Engine 24+ / Compose v2 下，重复构建通常可直接命中这两类缓存。
- 当前镜像构建阶段会临时启用 sid 源，仅安装 `intel-opencl-icd` 与 `libze-intel-gpu1`。
- 容器内不安装 `xserver-xorg-video-intel`（该包用于 Xorg 显示栈，不是本服务的无头推理运行前提）。
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
- 有 Intel iGPU 且已映射 `/dev/dri` 时，建议使用 `INFERENCE_DEVICE=AUTO`、`CLIP_INFERENCE_DEVICE=AUTO`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=AUTO`；RapidOCR 当前固定走 CPU，如 OCR 首次请求存在冷加载编译开销，可显式补 `OCR_EXEC_TIMEOUT=30`
- `INFERENCE_QUEUE_MAX_SIZE` 建议保持 `10`；即使显式配得更大，服务也会按 `10` 截断，避免 MT-Photos 客户端在图片请求积压时超时取消
- 如需挂载自定义模型、RapidOCR 配置或自定义 OpenVINO cache 目录，可再调整 `MODEL_PATH`、`RAPIDOCR_MODEL_DIR`、`RAPIDOCR_OPENVINO_CONFIG_PATH`、`OV_CACHE_DIR`
- 若 `/clip/img` 仍未跑满 GPU，可结合业务流量逐步调大 `CLIP_IMAGE_BATCH`，并保持 `CLIP_IMAGE_BATCH_WAIT_MS` 在个位数毫秒级，避免明显放大单请求尾延迟

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
  -e CLIP_IMAGE_BATCH=8 \
  -e RAPIDOCR_DEVICE=CPU \
  -e INSIGHTFACE_OV_DEVICE=AUTO \
  -e OCR_EXEC_TIMEOUT=30 \
  -e NON_TEXT_IDLE_RELEASE_SECONDS=60 \
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
