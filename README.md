# MT-Photos AI (OpenVINO)

主服务提供 OCR、图像向量（QA-CLIP）和人脸向量（InsightFace）；`Text-CLIP` 现已拆为独立 CPU 服务，单独对外提供 `/clip/txt`。仓库另外补充了一个仅面向本地 Windows 开发机的并行 `Image-CLIP` CUDA 子项目，目录为 `image-clip/`，不影响主服务 OpenVINO 实现。本文档仅保留最终用户部署、运行和配置说明。

## 部署前准备

- Python **3.12**
- 已准备模型目录（至少包含）：
  - `models/qa-clip/openvino`
  - `models/insightface/models/antelopev2`（至少保留 `scrfd_10g_bnkps.onnx` 与 `glintr100.onnx`）
  - `models/rapidocr`（需预置 PP-OCRv5 mobile det/rec/dict + cls 本地文件）
- 主服务入口：`app/server.py`
- Text-CLIP 服务入口：`text-clip/app/server.py`
- Windows 本地 CUDA Image-CLIP 子项目命令行入口：`image-clip/starter.py`
- Windows 本地 CUDA Image-CLIP 子项目服务实现入口：`image-clip/app/server.py`
- QA-CLIP 离线转换脚本：`scripts/convert.py`
- 独立 Text-CLIP 服务已自带 tokenizer 与词表资源，不再依赖主服务 `app/` 目录
- `requirements.txt` 当前固定 `insightface==0.7.3`，并显式包含 `onnx`，用于 InsightFace 首次懒加载时在受控 runtime copy 中修正 `glintr100.onnx` 的识别输出 batch 元数据；`scrfd_10g_bnkps.onnx` 保持原生 detector 路径

## 运行时环境变量

以下仅保留当前仍支持且建议用户配置的环境变量。兼容别名、已固定为内部基线的变量，以及 `docker compose` 插值变量不再写入 README。

### 主容器

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | API Key；为空或 `no-key` 时关闭鉴权 | `mt_photos_ai_extra` |
| `CLIP_IMAGE_BATCH` | `/clip/img` 标准预处理后的微批上限 | `8` |
| `CLIP_IMAGE_BATCH_WAIT_MS` | `/clip/img` 微批等待窗口，单位毫秒 | `5` |
| `CLIP_INFERENCE_DEVICE` | `/clip/img` 使用的 OpenVINO 设备；请求 `AUTO/GPU` 时需要可用 GPU | 跟随 `INFERENCE_DEVICE` |
| `INFERENCE_DEVICE` | 主容器默认 OpenVINO 设备 | `AUTO` |
| `INFERENCE_EXEC_TIMEOUT` | 非文本任务执行超时，单位秒 | `max(30, INFERENCE_TASK_TIMEOUT)` |
| `INFERENCE_QUEUE_MAX_SIZE` | `/clip/img`、`/ocr`、`/represent` 共享图片名额上限；运行时硬上限仍为 `10` | `10` |
| `INFERENCE_QUEUE_TIMEOUT` | 非文本任务排队超时，单位秒 | 跟随 `INFERENCE_TASK_TIMEOUT` |
| `INFERENCE_TASK_TIMEOUT` | 兼容旧配置的总超时基线；未单独设置时会派生排队/执行超时 | `10` |
| `INSIGHTFACE_BATCH_WAIT_MS` | `/represent` recognition 聚合窗口，单位毫秒 | `5` |
| `INSIGHTFACE_OV_DEVICE` | InsightFace OpenVINO EP 推理设备；支持 `AUTO` / `CPU` / `GPU` | `AUTO` |
| `INSIGHTFACE_OV_ENABLE_OPENCL_THROTTLING` | 是否启用 OpenCL throttling | `false` |
| `INSIGHTFACE_OV_NUM_THREADS` | InsightFace OpenVINO EP 线程数；`-1` 为运行时默认 | `-1` |
| `LOG_LEVEL` | 日志级别；同时作用于 `mt_photos_ai.*` 与 `uvicorn.*` | `WARNING` |
| `MODEL_PATH` | 模型根目录 | `<repo>/models` |
| `NON_TEXT_IDLE_RELEASE_SECONDS` | 主容器非文本模型空闲释放窗口；`<=0` 表示关闭 | `60` |
| `OCR_EXEC_TIMEOUT` | OCR 执行超时，单位秒 | `max(30, INFERENCE_EXEC_TIMEOUT)` |
| `OCR_MAX_CONCURRENT_REQUESTS` | OCR 应用层最大并发请求数；不会超过共享图片名额 | `min(INFERENCE_QUEUE_MAX_SIZE, max(2, RAPIDOCR_PERFORMANCE_NUM_REQUESTS*2))` |
| `OCR_PREWARM_DELAY_SECONDS` | RapidOCR 一次性后台预热延迟，单位秒 | `1.0` |
| `OCR_PREWARM_ENABLED` | 是否启用一次性后台 RapidOCR 预热 | `false` |
| `OV_CACHE_DIR` | OpenVINO 编译缓存目录 | `<repo>/cache/openvino` |
| `PORT` | 服务端口；同时影响镜像入口与健康检查 | `8060` |
| `RAPIDOCR_CLS_BATCH_NUM` | RapidOCR 方向分类批大小 | `8` |
| `RAPIDOCR_DET_LIMIT_SIDE_LEN` | RapidOCR 检测输入边长限制 | `960` |
| `RAPIDOCR_DET_LIMIT_TYPE` | RapidOCR 检测缩放策略 | `max` |
| `RAPIDOCR_DEVICE` | RapidOCR 请求设备；当前实现会固定收敛到 `CPU` | `CPU` |
| `RAPIDOCR_ENABLE_CPU_PINNING` | 是否启用 CPU 绑核 | `true` |
| `RAPIDOCR_ENABLE_HYPER_THREADING` | 是否启用超线程 | `true` |
| `RAPIDOCR_FONT_PATH` | RapidOCR 字体文件路径；空表示不指定 | 空 |
| `RAPIDOCR_INFERENCE_NUM_THREADS` | RapidOCR 推理线程数 | `-1` |
| `RAPIDOCR_MAX_SIDE_LEN` | OCR 全图最大边限制 | `960` |
| `RAPIDOCR_MODEL_DIR` | RapidOCR 本地模型目录 | `<repo>/models/rapidocr` |
| `RAPIDOCR_NUM_STREAMS` | RapidOCR OpenVINO stream 数 | `2` |
| `RAPIDOCR_OPENVINO_CONFIG_PATH` | RapidOCR YAML 配置文件路径 | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_PERFORMANCE_HINT` | RapidOCR OpenVINO 性能提示 | `THROUGHPUT` |
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | RapidOCR OpenVINO request 数；同时决定 OCR worker 基线 | `2` |
| `RAPIDOCR_REC_BATCH_NUM` | RapidOCR 识别批大小 | `8` |
| `RAPIDOCR_SCHEDULING_CORE_TYPE` | RapidOCR OpenVINO 核调度类型 | `ANY_CORE` |
| `RAPIDOCR_USE_CLS` | 是否启用方向分类器 | `true` |

### Text-CLIP 容器

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | API Key；为空或 `no-key` 时关闭鉴权 | `mt_photos_ai_extra` |
| `LOG_LEVEL` | 日志级别；同时作用于 `mt_photos_ai.text_clip` 与 `uvicorn.*` | `WARNING` |
| `MODEL_PATH` | 文本模型根目录 | `<repo>/models` |
| `OV_CACHE_DIR` | Text-CLIP OpenVINO 编译缓存目录 | `<repo>/cache/openvino` |
| `PORT` | Text-CLIP 服务端口；同时影响镜像入口与健康检查 | `8061` |

补充说明：

- 开发机本地验证时，主服务建议把所有后端统一设为 `CPU`：`INFERENCE_DEVICE=CPU`、`CLIP_INFERENCE_DEVICE=CPU`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=CPU`；Text-CLIP 容器固定使用 `CPU`，不需要额外设备变量。
- 主容器的 Vision-CLIP / OCR / InsightFace 仍按请求懒加载，并可按 `NON_TEXT_IDLE_RELEASE_SECONDS` 自动释放；Text-CLIP 容器启动即加载文本模型，进程存活期间常驻内存，不参与空闲释放或主容器 `/restart`。
- `PORT` 会同时影响容器入口和健康检查；如果修改它，请同步调整 `docker-compose.yml` 的 `ports:` 或 `docker run -p`。
- 如果手动执行 `uvicorn server:app`，最早期的 uvicorn bootstrap 日志仍以 CLI `--log-level` 为准。

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
$env:LOG_LEVEL="INFO"
```

4. 如需使用 `/clip/txt`，先启动独立 Text-CLIP 服务：

```powershell
cd text-clip\app
python server.py
```

5. 如需使用主服务接口，再另开终端启动主服务：

```powershell
cd app
python server.py
```

如需在本地 Windows 开发机上单独跑 CUDA 版 Image-CLIP，可改用 `image-clip/` 子项目；该子项目使用独立依赖文件 `image-clip/requirement.txt`，可在仓库根目录直接执行 `python image-clip\starter.py`，或进入 `image-clip\` 后执行 `python starter.py`。更完整的环境变量与冒烟说明见 [image-clip/README.md](image-clip/README.md)。

如需手动执行 `uvicorn server:app`，请显式传入 `--port` / `--log-level`，例如 `uvicorn server:app --host 0.0.0.0 --port 8060 --log-level info`。

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
- 当前 Docker 构建会在 `pip install -r requirements.txt` 后移除传递安装的 `opencv-python`/`opencv-contrib-python`，再补装 `opencv-python-headless`；当前服务只使用 OpenCV 的图像解码、色彩转换和 CPU `resize`/`warpAffine` 路径，不需要 X11/OpenGL 运行时包。
- 因此当前运行时镜像不再包含 `libgl1`、`libsm6`、`libxext6`、`libxrender1`，也不再打包 `mesa-vulkan-drivers`、`intel-media-va-driver-non-free`。
- Intel GPU 固件属于宿主机职责；若宿主 Debian 13 需要补齐固件，请在宿主机安装 `firmware-misc-nonfree`（或兼容包名 `firmware-misc-non-free`），而不是放进应用容器。
- Dockerfile 已固化为清华 APT + 清华 PyPI 镜像；APT 基础列表、sid pin 文件和安装包名单都直接写在仓库文件中，便于回溯与审计。
- Dockerfile 使用 BuildKit cache mount 复用 `apt`/`pip` 下载缓存；在 Docker Engine 24+ / Compose v2 下，重复构建通常可直接命中这两类缓存。
- 当前镜像构建阶段会临时启用 sid 源，仅安装 `intel-opencl-icd` 与 `libze-intel-gpu1`。
- 独立 Text-CLIP 镜像使用 `text-clip/DockerFile-TextCLIP`，固定走 OpenVINO CPU，不安装 Intel GPU runtime，也不需要 `/dev/dri`。
- 容器内不安装 `xserver-xorg-video-intel`（该包用于 Xorg 显示栈，不是本服务的无头推理运行前提）。
- 服务上传读图链已统一改为 OpenCV 原生解码，镜像不再包含 `ffmpeg/ffprobe`、VAAPI/oneVPL/QSV 媒体栈，也不预装 `clinfo` 这类诊断工具。
- 若服务日志出现 `available_devices=['CPU']`，即使 `/dev/dri` 可见，也通常意味着容器里缺少可用的 Intel GPU OpenVINO/OpenCL runtime，或 `/dev/dri` 并非真实的 Intel DRM render node。
- 参考文档：
  - OpenVINO GPU 设备配置与依赖：<https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>
  - Intel Linux GPU Driver（OpenCL/Level Zero 运行时包）：<https://dgpu-docs.intel.com/driver/installation.html>

### 方式一：docker compose

1. 先准备镜像。`docker-compose.example.yml` 只引用 `image:`，不会在 `docker compose up` 时构建：

```bash
docker build -t mt-photos-ai-openvino .
docker build -f text-clip/DockerFile-TextCLIP -t mt-photos-ai-text-clip .
```

如果镜像已经在本地存在，或你改为从镜像仓库拉取，可以跳过这一步。

2. 准备配置文件：

```bash
cp docker-compose.example.yml docker-compose.yml
```

3. 按宿主机实际情况修改 `group_add` 中的 `video` / `render` GID（Debian 默认通常是 `44` / `109`）。

4. 按需调整 `docker-compose.yml`：

- 生产环境建议覆盖 `API_AUTH_KEY`
- 主容器需要映射 `/dev/dri` 并设置正确的 `video/render` 组；`mt-photos-ai-text-clip` 固定走 CPU，不需要 `/dev/dri`
- `mt-photos-ai-text-clip` 启动后会常驻加载文本模型，直接对外提供 `/clip/txt`
- 有 Intel iGPU 且已映射 `/dev/dri` 时，主容器建议使用 `INFERENCE_DEVICE=AUTO`、`CLIP_INFERENCE_DEVICE=AUTO`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=AUTO`
- 如需修改服务监听端口，请同时调整 `PORT` 和 `ports:` 映射
- 如需挂载自定义模型或自定义 OpenVINO cache 目录，再显式覆盖 `MODEL_PATH` / `OV_CACHE_DIR`
- 若 `/clip/img` 仍未跑满 GPU，可结合业务流量逐步调大 `CLIP_IMAGE_BATCH`，并保持 `CLIP_IMAGE_BATCH_WAIT_MS` 在个位数毫秒级，避免明显放大单请求尾延迟
- 如需限制 OCR 首次冷加载带来的单次长尾，可显式设置 `OCR_EXEC_TIMEOUT=30`
- `/represent` 当前固定为单 lane OpenVINO EP 推理 + 4 请求聚合预算；如需权衡吞吐与尾延迟，可只小幅调整 `INSIGHTFACE_BATCH_WAIT_MS`

5. 启动服务：

```bash
docker compose up -d
```

6. 查看状态：

```bash
docker compose ps
docker compose logs -f mt-photos-ai-openvino mt-photos-ai-text-clip
```

### 方式二：docker run

```bash
docker run -d \
  --name mt-photos-ai-text-clip \
  --init \
  -p 8061:8061 \
  -e API_AUTH_KEY=mt_photos_ai_extra \
  -e LOG_LEVEL=WARNING \
  -e PORT=8061 \
  mt-photos-ai-text-clip

docker run -d \
  --name mt-photos-ai-openvino \
  --init \
  -p 8060:8060 \
  --device /dev/dri:/dev/dri \
  --group-add $(getent group video | cut -d: -f3) \
  --group-add $(getent group render | cut -d: -f3) \
  -e API_AUTH_KEY=mt_photos_ai_extra \
  -e CLIP_IMAGE_BATCH=8 \
  -e CLIP_INFERENCE_DEVICE=AUTO \
  -e INFERENCE_DEVICE=AUTO \
  -e INSIGHTFACE_OV_DEVICE=AUTO \
  -e LOG_LEVEL=WARNING \
  -e NON_TEXT_IDLE_RELEASE_SECONDS=60 \
  -e OCR_EXEC_TIMEOUT=30 \
  -e PORT=8060 \
  -e RAPIDOCR_DEVICE=CPU \
  mt-photos-ai-openvino
```

如需自定义模型目录或 OpenVINO cache 目录，可继续追加 `-e MODEL_PATH=...`、`-e OV_CACHE_DIR=...`。如需修改任一服务的 `PORT`，请同步调整对应的 `-p <host_port>:<container_port>`。

### 容器内设备检查

首次部署建议执行以下检查：

```bash
docker exec -it mt-photos-ai-openvino ls -l /dev/dri
docker exec -it mt-photos-ai-openvino python -c "import openvino as ov; print(ov.Core().available_devices)"
```

若请求了 GPU 推理但容器内 GPU 设备不可用，服务会直接报错并终止启动。

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

推荐先在 Docker 容器内执行 InsightFace 专项冒烟：

```bash
docker build -f text-clip/DockerFile-TextCLIP -t mt-photos-ai-text-clip .
docker build -t mt-photos-ai-openvino .
docker run --rm -it \
  -e INFERENCE_DEVICE=CPU \
  -e CLIP_INFERENCE_DEVICE=CPU \
  -e RAPIDOCR_DEVICE=CPU \
  -e INSIGHTFACE_OV_DEVICE=CPU \
  mt-photos-ai-openvino \
  python scripts/smoke_insightface.py --device CPU
```

如宿主机已映射 Intel iGPU 的 `/dev/dri`，可进一步执行 GPU 冒烟：

```bash
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add $(getent group video | cut -d: -f3) \
  --group-add $(getent group render | cut -d: -f3) \
  -e INFERENCE_DEVICE=CPU \
  -e CLIP_INFERENCE_DEVICE=CPU \
  -e RAPIDOCR_DEVICE=CPU \
  -e INSIGHTFACE_OV_DEVICE=GPU \
  mt-photos-ai-openvino \
  python scripts/smoke_insightface.py --device GPU
```

脚本会校验：

- `OpenVINOExecutionProvider` 仍是 detection/recognition 的首位 provider，且 `device_type` 与请求一致
- InsightFace 运行事实已收敛到单 lane detector/recognition + 固定 CPU 预处理 + 最多 4 路预处理 worker + 默认 4 请求 admission/聚合上限
- 运行时模型目录已经收敛到 `_runtime_models/models/antelopev2`
- 本地 `/represent` pipeline 与原生 `FaceAnalysis.get` 的检测框、分数和 embedding 语义保持一致
- 原生 detector.detect 与 batched recognition 路径均保持稳定
- 并发 `/represent` 聚合路径与顺序 `/represent` 路径一致，不改变检测框、分数和 embedding 语义
- `release_models_for_restart()` 后 face runtime 引用已释放，且能够重新加载

服务启动后，也可以再做基础端点检查：

```bash
curl -s http://127.0.0.1:8060/
curl -s -X POST http://127.0.0.1:8060/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8061/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8061/clip/txt -H "api-key: mt_photos_ai_extra" -H "Content-Type: application/json" -d '{"text":"smoke"}'
```

除 `GET /` 外，业务端点在启用鉴权时都需要携带 `api-key` 请求头。

## 许可证

MIT License
