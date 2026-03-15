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
- Windows 本地 CUDA Image-CLIP 子项目入口：`image-clip/app/server.py`
- QA-CLIP 离线转换脚本：`scripts/convert.py`
- 独立 Text-CLIP 服务已自带 tokenizer 与词表资源，不再依赖主服务 `app/` 目录
- `requirements.txt` 当前固定 `insightface==0.7.3`，并显式包含 `onnx`，用于 InsightFace 首次懒加载时在受控 runtime copy 中修正 `glintr100.onnx` 的识别输出 batch 元数据；`scrfd_10g_bnkps.onnx` 保持原生 detector 路径

## 运行时环境变量

| 环境变量                                | 说明                                                        | 默认值                                |
|-------------------------------------|-----------------------------------------------------------|------------------------------------|
| `API_AUTH_KEY`                      | API Key；`no-key` 或空字符串表示关闭鉴权                              | `mt_photos_ai_extra`               |
| `INFERENCE_DEVICE`                  | OpenVINO 设备字符串，如 `GPU` / `CPU` / `AUTO`                 | `AUTO`                             |
| `CLIP_INFERENCE_DEVICE`             | 仅覆盖主服务 `/clip/img` 的 QA-CLIP 视觉设备；请求 `AUTO/GPU` 时需保证 GPU 可用，且会强制初始化 GPU Remote Context | 跟随 `INFERENCE_DEVICE`              |
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
| `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` | RapidOCR OpenVINO CPU 配置项；同时作为 OCR 多实例池和执行器默认 worker 数的基线 | `2`                                |
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
| `INSIGHTFACE_OV_DEVICE`             | 仅控制 InsightFace ORT OpenVINO EP 推理侧 `device_type`；`AUTO` 在 GPU 可见时会优先解析为 `GPU`，detector/recognition 的非推理预处理固定走原生 CPU 路径 | `AUTO`                             |
| `INSIGHTFACE_OV_ENABLE_OPENCL_THROTTLING` | 是否启用 OpenVINO EP 的 OpenCL 节流；吞吐优先场景建议关闭                  | `false`                            |
| `INSIGHTFACE_OV_NUM_THREADS`        | InsightFace OpenVINO EP CPU 线程数；`-1` 表示使用运行时默认值               | `-1`                               |
| `INSIGHTFACE_BATCH_WAIT_MS`         | `/represent` 单 lane 队列聚合窗口（毫秒）；默认以 `4` 个请求作为 admission 与聚合上限，若共享图片总名额更小则继续截断 | `5`                                |
| `PORT`                              | 服务端口；对 Docker 镜像入口、容器健康检查和 `python server.py` 生效 | `8060`                             |
| `LOG_LEVEL`                         | 日志级别：作用于 `mt_photos_ai.*`、`uvicorn.*` 和 RapidOCR logger | `WARNING`                          |

补充说明：

- 开发机本地验证时，主服务建议把所有后端统一设为 `CPU`：`INFERENCE_DEVICE=CPU`、`CLIP_INFERENCE_DEVICE=CPU`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=CPU`；独立 Text-CLIP 服务固定使用 `CPU`，不需要额外 GPU 变量。`RAPIDOCR_DET_DEVICE/RAPIDOCR_CLS_DEVICE/RAPIDOCR_REC_DEVICE` 当前仅保留兼容占位，不参与运行时选路。
- `LOG_LEVEL` 会在应用导入和 lifespan 启动阶段同步到 `mt_photos_ai.*`、`uvicorn.*` 和 RapidOCR logger；Docker 镜像入口也会把它传给 `uvicorn --log-level`。如果你手动执行 `uvicorn server:app`，最早期的 uvicorn bootstrap 日志仍以 CLI `--log-level` 为准。
- 服务会在 `uvicorn server:app` 和 `python server.py` 两种启动路径下都把 `mt_photos_ai.*` 日志输出到控制台；Windows 直接运行时还会追加写入 `<repo>/server.log`。
- 如需看到启动自检、模型族切换和 `/restart` 释放日志，请显式设置 `LOG_LEVEL=INFO`。
- `PORT` 当前对 Docker 镜像入口、容器健康检查和 `python server.py` 生效；如果手动执行 `uvicorn server:app`，请显式传 `--port`。
- 服务固定为单进程；容器入口和 `python server.py` 都会以单 worker 运行，若同一工作目录下已有实例持有运行锁，新的进程会直接启动失败。
- `CLIP_IMAGE_BATCH_SIZE` 仍作为兼容别名被读取；仅在未设置 `CLIP_IMAGE_BATCH` 时才会生效。
- `RAPIDOCR_DET_DEVICE` / `RAPIDOCR_CLS_DEVICE` / `RAPIDOCR_REC_DEVICE` 当前只用于兼容告警，不参与实际 stage 选路。
- 当 `CLIP_INFERENCE_DEVICE` 请求 `GPU` 或 `AUTO` 时，服务会强制初始化 OpenVINO GPU Remote Context。
- Remote Context 初始化会依次尝试默认 `GPU`、具体 `GPU.*` 设备，以及 `create_context("GPU", {})` 兼容路径；全部失败时直接终止启动，不允许 silent fallback。
- 主服务不再包含 Text-CLIP 模型，也不再提供 `/clip/txt`；该端点仅由独立 Text-CLIP CPU 服务提供。
- Vision-CLIP / OCR / InsightFace 采用“单活非文本模型族”切换策略：切换到新模型族前，会先等待当前已受理任务退场并同步释放旧族模型，避免三个大模型长期同时驻留。
- 非文本空闲释放计时现在只由 `/clip/img`、`/ocr`、`/represent` 刷新；`/check`、`/restart`、`/restart_v2` 不会再阻止 Vision-CLIP / OCR / InsightFace 自动释放。
- `/clip/img` 会先执行标准预处理（缩放、中心裁剪、PPP 归一化），再通过专用 `asyncio.Queue` 做受控微批，并以 `np.stack` 一次送入动态 batch 视觉模型。
- 为避免 MT-Photos 客户端在积压时主动取消，服务会对 `/clip/img`、`/ocr`、`/represent` 共享一个图片请求名额池；默认值和运行时硬上限都是 `10`，第 `11` 张会立即失败而不是继续挂起等待。
- RapidOCR 现已回到库内原生 `RapidOCR.__call__` CPU 执行链：服务只负责懒加载、应用层准入和超时控制，不再替换 session、也不再维护自定义 stage 级预处理/批调度。
- 非文本超时仍拆分为“排队超时”和“执行超时”；`/clip/img` 使用有界 `asyncio.Queue` 批队列，OCR/InsightFace 使用各自独立执行器；OCR/Face 的异步路径会先完成模型加载，再进入执行超时窗口。
- RapidOCR 会直接加载 `RAPIDOCR_OPENVINO_CONFIG_PATH` 指向的 YAML，并额外校验 `Det/Cls/Rec.engine_type=openvino`，避免回落到默认 ORT 配置。
- RapidOCR 当前基线为 `OpenVINO + CPU + PP-OCRv5 mobile + use_cls=true`；即使显式传入 `RAPIDOCR_DEVICE=AUTO/GPU` 或 stage 级设备变量，运行时也会告警并强制回到 CPU。
- RapidOCR 默认基线使用 `Det.limit_type=max`；若配置成 `min`，小图会被放大到 `limit_side_len`，通常会明显拉高检测时延。
- RapidOCR 慢请求日志现输出总耗时和当前 CPU 配置，便于区分是 OCR 本体慢还是排队/模型切换导致的尾延迟。
- OCR 异步执行会通过有界 `RapidOCR` 多实例池并发调用库内原生 `RapidOCR.__call__`；每个 worker 独占一个实例，避免共享 OpenVINO `InferRequest` 导致串行化或线程安全问题。并发上限仍受 `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` 和 `OCR_MAX_CONCURRENT_REQUESTS` 控制。
- 对于确实不含文字的图片，`/ocr` 会正常返回空结果；服务会过滤 RapidOCR 自身 `"The text detection result is empty"` 的预期 warning，避免日志刷屏。
- 服务现在默认启用本地 OpenVINO 编译缓存目录 `<repo>/cache/openvino`；如果需要自定义路径，可显式设置 `OV_CACHE_DIR`。
- 默认不会在启动后把 OCR 拉入内存；OCR 会在首次 `/ocr` 请求时懒加载。若显式开启 `OCR_PREWARM_ENABLED=true`，服务仅做一次性预热并立即释放 OCR，不会让 OCR 常驻。
- 默认还会在连续 `60s` 未收到业务请求时自动释放主容器内的 Vision-CLIP / OCR / InsightFace；独立 Text-CLIP 服务不受该计时器影响。如需调整主容器释放窗口，可设置 `NON_TEXT_IDLE_RELEASE_SECONDS`。
- `POST /restart` 会同步等待当前非文本任务退场并释放 Vision-CLIP / OCR / InsightFace；Text-CLIP 服务不参与该流程。返回 `{"result":"pass"}` 时本轮释放已经完成。
- InsightFace 在 GPU 可见时会把 `INSIGHTFACE_OV_DEVICE=AUTO` 收敛为 `GPU`，但这只影响 ORT OpenVINO EP 推理侧；detector 的 resize/letterbox、recognition 的 blob 组装和人脸对齐都固定走 CPU。日志会输出 `configured_device`、`runtime_device`、`preprocess_device`、`provider_runtime` 和 `face_preprocess_workers` 便于核对推理侧与 CPU 侧职责。
- 若 `models/insightface/models/antelopev2/glintr100.onnx` 的输出 batch 元数据仍写死为 `{1,512}`，服务会在受控 runtime copy 中把识别输出 batch 维修正为动态后再初始化 ORT session；仓库内原始模型文件不会被改写，detector 模型保持原生副本。
- InsightFace 固定使用新版 `FaceAnalysis(name=..., root=..., allowed_modules=..., providers=..., provider_options=...)` 初始化路径；若当前 `insightface` 包不支持这组参数，服务会直接失败，不再做兼容重试。
- InsightFace 运行时模型固定物化到 `<MODEL_PATH>/insightface/_runtime_models/models/antelopev2`；服务只会在这里生成受控 runtime copy，并仅对 recognition 模型做 batch 元数据修正，不再维护兼容目录或 source/runtime 回退。
- 无论 `INSIGHTFACE_OV_DEVICE` 取 `GPU`、`CPU` 还是 `AUTO`，InsightFace 都遵循“仅推理走 OpenVINO EP，其余尽量原生 CPU”基线：检测委托 `det_model.detect`，识别特征提取委托 `rec_model.get_feat`，五点对齐仍使用仓库内本地 CPU 仿射矩阵实现；`INSIGHTFACE_OV_DEVICE` 仅决定 ORT OpenVINO EP 的推理设备，且 `OpenVINOExecutionProvider` 必须保持首位 provider。
- `/represent` 当前固定为单个 detector/recognition lane；检测阶段按请求顺序走原生 detector，识别阶段会把同一聚合窗口里的全部 aligned face 一次送入 batched recognition。默认 admission=4、聚合上限=4、预处理 worker 上限=4，若共享图片总名额更小则继续截断。
- `INSIGHTFACE_MAX_WORKERS`、`INSIGHTFACE_MAX_CONCURRENT_REQUESTS`、`INSIGHTFACE_BATCH_SIZE` 已移除，避免继续暴露与单 lane 事实不一致的调参项。
- `/represent` 现在会通过有界单 lane 队列平滑跨请求调度；检测阶段保持单请求语义，识别阶段才做受控批量聚合，既减少自定义 detector 代码，又保留 recognition 侧的 GPU 利用率与响应稳定性。

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

如需在本地 Windows 开发机上单独跑 CUDA 版 Image-CLIP，可改用 `image-clip/` 子项目；该子项目使用独立依赖文件 `image-clip/requirement.txt`，启动方式见 [image-clip/README.md](image-clip/README.md)。

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

1. 准备配置文件，并确认 `docker-compose.yml` 里的两个服务都已指向你实际要使用的镜像标签；如使用仓库内 Dockerfile，可直接保留示例中的 `build:`：

```bash
cp docker-compose.example.yml docker-compose.yml
```

说明：
- `text-clip/DockerFile-TextCLIP` 必须以仓库根目录作为 Docker build context；命令行示例是 `docker build -f text-clip/DockerFile-TextCLIP -t mt-photos-ai-text-clip .`
- 如果在 IDE 里单独部署该 Dockerfile，`Dockerfile` 路径应指向 `text-clip/DockerFile-TextCLIP`，但 `Build context folder` 必须设为仓库根目录，而不是 `text-clip/`
- 如果构建日志里出现 `load build context => transferring context: 2B`，基本就说明 build context 错设成了 `text-clip/` 或其他空目录；这时 `COPY sources.list`、`COPY models/...`、`COPY text-clip/app` 会全部失败

2. 按宿主机实际情况修改 `group_add` 中的 `video` / `render` GID（Debian 默认通常是 `44` / `109`）。

3. 按需调整 `docker-compose.yml`：

- 生产环境建议覆盖 `API_AUTH_KEY`
- `mt-photos-ai-text-clip` 为独立 CPU 容器，直接对外提供 `/clip/txt`
- 有 Intel iGPU 且已映射 `/dev/dri` 时，建议使用 `INFERENCE_DEVICE=AUTO`、`CLIP_INFERENCE_DEVICE=AUTO`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=AUTO`；其中 `INSIGHTFACE_OV_DEVICE` 仅影响推理侧 OpenVINO EP，InsightFace 其余路径仍固定走原生 CPU。RapidOCR 当前固定走 CPU，如 OCR 首次请求存在冷加载编译开销，可显式补 `OCR_EXEC_TIMEOUT=30`
- `INFERENCE_QUEUE_MAX_SIZE` 建议保持 `10`；即使显式配得更大，服务也会按 `10` 截断，避免 MT-Photos 客户端在图片请求积压时超时取消
- 如需修改服务监听端口，请同时调整 `PORT` 和 `ports:` 映射；服务固定单进程，不再提供 worker 数配置项
- 如需挂载自定义模型、RapidOCR 配置或自定义 OpenVINO cache 目录，可再调整两个容器各自的 `MODEL_PATH` / `OV_CACHE_DIR`，以及主服务的 `RAPIDOCR_MODEL_DIR`、`RAPIDOCR_OPENVINO_CONFIG_PATH`
- 若 `/clip/img` 仍未跑满 GPU，可结合业务流量逐步调大 `CLIP_IMAGE_BATCH`，并保持 `CLIP_IMAGE_BATCH_WAIT_MS` 在个位数毫秒级，避免明显放大单请求尾延迟
- `/represent` 当前固定为单 lane OpenVINO EP 推理 + 4 请求聚合预算；如需权衡吞吐与尾延迟，可只小幅调整 `INSIGHTFACE_BATCH_WAIT_MS`，不再提供额外的 worker/batch 容量环境变量

4. 启动服务：

```bash
docker compose up -d --build
```

5. 查看状态：

```bash
docker compose ps
docker compose logs -f mt-photos-ai-openvino mt-photos-ai-text-clip
```

### 方式二：docker run

```bash
docker build \
  --build-arg APP_UID=$(id -u) \
  --build-arg APP_GID=$(id -g) \
  -f text-clip/DockerFile-TextCLIP \
  -t mt-photos-ai-text-clip .

docker build \
  --build-arg APP_UID=$(id -u) \
  --build-arg APP_GID=$(id -g) \
  -t mt-photos-ai-openvino .

docker run -d \
  --name mt-photos-ai-text-clip \
  --init \
  -p 8061:8061 \
  -e API_AUTH_KEY=mt_photos_ai_extra \
  -e MODEL_PATH=/models \
  -e OV_CACHE_DIR=/cache/openvino \
  -e PORT=8061 \
  -e LOG_LEVEL=WARNING \
  mt-photos-ai-text-clip

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
  -e MODEL_PATH=/models \
  -e CLIP_IMAGE_BATCH=8 \
  -e RAPIDOCR_DEVICE=CPU \
  -e INSIGHTFACE_OV_DEVICE=AUTO \
  -e OCR_EXEC_TIMEOUT=30 \
  -e NON_TEXT_IDLE_RELEASE_SECONDS=60 \
  -e PORT=8060 \
  -e LOG_LEVEL=WARNING \
  mt-photos-ai-openvino
```

如需修改任一服务的 `PORT`，请同步调整对应的 `-p <host_port>:<container_port>`。

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
