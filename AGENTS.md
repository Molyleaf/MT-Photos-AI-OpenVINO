# AGENTS / agents.md — 本仓库 AI/Agent Python(OpenVINO) 改动规范（简体中文）

> 你是任何形式的 AI/Agent（Copilot、Cursor、LLM Agent、自动重构工具等）。只要你在本仓库写/改 Python 或部署相关文件，就必须遵守本文。
>
> 严禁做这些事：**改接口语义、不按要求替换推理后端、非兼容性需求改 QA-CLIP 子库、线程/进程过度并发导致抖动、模型重复驻留导致内存暴涨、文档与依赖不一致**。
>
> 目标：产出 **行为稳定、响应兼容、资源可控、可回归验证** 的改动，而不是“看起来很优化”的改动。
> 
> 不要进行 `pip install -r requirements.txt` 验证

---

## 1. 仓库事实与目录结构（先对齐背景）

- 语言与运行时：**Python 3.12**（目标平台：Windows Server + Debian 13 容器）。
- Debian 容器镜像源基线：APT 使用 `https://mirrors.zju.edu.cn/debian/`，PyPI 使用 `https://mirrors.zju.edu.cn/pypi/web/simple`。
- 硬件基线：Intel i7-11800H（AVX512 VNNI + Xe 核显，共享内存架构）。
- 服务入口：`app/server.py`（当前仓库中等效于历史 `server_openvino.py` 的实现入口）。
- 模型编排：`app/models.py`。
- 模型转换：`app/convert.py`（QA-CLIP -> OpenVINO IR）。
- 模型目录：`models/qa-clip/openvino`、`models/insightface/models`。
- QA-CLIP 子库：`app/QA-CLIP`（来自 TencentARC-QQ 官方仓库）。
- 配置存储：`app/config`。
- 参考文件（对齐端点用）：`example/`。

---

## 2. QA-CLIP 子库边界（硬约束）

1. `app/QA-CLIP` 视为上游镜像目录，默认**禁止功能性改动**。
2. 允许改动仅限：
  - 兼容性修复（Python/OpenVINO/依赖版本适配）
  - 只优化性能且不改变语义的改动
3. CLIP 相关依赖必须从 `app/QA-CLIP/clip` 引用；禁止继续依赖历史 `/app/clip` 路径。
4. 若确需改动 `app/QA-CLIP`，必须在提交说明里写清楚：
  - 改动类型（兼容性 or 性能）
  - 不改语义的证据（接口/输出维度/精度基线）

---

## 3. 模型与后端基线（硬约束）

### 3.1 QA-CLIP（图文向量）

- 模型固定为：`TencentARC/QA-CLIP-ViT-L-14`。
- 向量维度固定：`768`。
- 推理目标设备：`GPU`（Intel Xe 核显），优先减少 Host<->Device 数据搬运。
- OpenVINO 侧优先启用 Remote Tensor API 相关互操作能力（零拷贝/少拷贝优先）。
- 当 `CLIP_INFERENCE_DEVICE=AUTO` 时，必须强制初始化 GPU Remote Context；初始化失败必须直接报错，禁止 silent fallback。
- 当 `CLIP_INFERENCE_DEVICE` 显式包含 `GPU`（如 `GPU`、`AUTO:GPU,CPU`）时，也必须显式完成 GPU Remote Context 初始化；失败直接报错，禁止静默继续。
- `/clip/img` 必须支持 **worker 内同尺寸微批**，以提高 Intel Xe GPU 利用率；微批只能发生在相邻任务且不得改变单请求输入输出语义。
- 当前服务上传读图链不再依赖 `ffmpeg/QSV`；Debian/Linux 容器部署与验收仍以 `/dev/dri` GPU 节点和 OpenVINO/OpenCL 可见性为准，不能把仅有 `/dev/dxg` 视作当前镜像的等价前提。
- `/clip/img` 视觉链路必须直接消费 `numpy BGR`，禁止 `BGR -> RGB -> PIL` 的多余拷贝链。
- CLIP 视觉预处理（`resize + 通道转换 + 归一化 + layout`）必须走 OpenVINO PrePostProcessing (PPP) API，禁止回退到手工 `numpy` 链。
- 当视觉模型输入为动态 shape 时，PPP 的 `resize` 目标尺寸必须显式固定到模型期望分辨率（当前基线 `224x224`），禁止依赖隐式推断导致运行时构图失败。

### 3.2 QA-CLIP 转换（Hugging Face -> OpenVINO IR）

- 必须从 Hugging Face 拉取后再转换为 IR，不允许提交来源不明的权重。
- 转换过程必须避免“双份内存常驻”：
  - 禁止在同一阶段同时常驻完整 PyTorch 模型副本 + 完整 OpenVINO 中间副本。
  - 视觉分支与文本分支按顺序转换，转换后及时释放前一阶段对象并 `gc.collect()`。
- FP16 压缩要求：
  - 使用官方推荐 `--compress_to_fp16`（或等价 API `compress_to_fp16=True`）。
  - 禁止继续使用旧参数 `--data_type FP16`。
- NNCF 约束：
  - 使用 NNCF 流程做转换/压缩集成。
  - **关键层不压缩**（输入投影、输出投影、LayerNorm/归一化等敏感层保持 FP32）。

### 3.3 RapidOCR

- 依赖固定：`rapidocr==3.7.0`。
- 禁止使用：`rapidocr-openvino`。
- 禁止对 RapidOCR 模型做量化或结构改写。
- 必须使用 RapidOCR 内置后端选择能力，指定 **OpenVINO** 后端；必须把 `app/config/cfg_openvino_cpu.yaml` 作为 `RapidOCR(config_path=...)` 传入，禁止继续依赖库内默认 YAML。
- RapidOCR OpenVINO 设备基线使用 `MULTI` 表达式；默认 `device_name=MULTI:GPU,CPU`。若配置来源为 `AUTO` / `GPU` / `GPU_FP16` 等语义，运行时也必须先归一化为 `MULTI:*` 再传给 OpenVINO。
- 当 RapidOCR 设备显式包含 `GPU`（如 `MULTI:GPU,CPU`、`MULTI:GPU.0,CPU`）时，若 OpenVINO 无 GPU 设备必须直接报错，禁止 silent fallback 到纯 CPU。
- 默认使用 **PP-OCRv5 mobile** 模型配置（`Det/Rec`）。
- 默认开启方向分类器（`Global.use_cls=true`），并预置分类模型。
- 预处理基线（面向 i7-11800H 核显）：`max_side_len=960`、`Det.limit_side_len=960`、`Det.limit_type=max`、`Rec.rec_batch_num=8`、`Cls.cls_batch_num=8`。
- OpenVINO 参数基线：`device_name=MULTI:GPU,CPU`、`performance_hint=THROUGHPUT`、`performance_num_requests=2`、`inference_num_threads=-1`、`num_streams=2`。
- 配置优先级必须为：**显式环境变量 `RAPIDOCR_*` > YAML(`cfg_openvino_cpu.yaml`) > 代码默认值**；禁止 YAML 覆盖显式运行时设备设置。
- 示例参数文件为 `app/config/cfg_openvino_cpu.yaml`；关键配置项包括 `device_name`、`inference_num_threads`、`performance_hint`、`performance_num_requests`、`enable_cpu_pinning`、`num_streams`、`enable_hyper_threading`、`scheduling_core_type`。
- 必须启用模型编译缓存，降低冷启动与多 Worker 反复编译开销。
- 默认缓存目录应收敛到仓库内可写路径（当前基线 `<PROJECT_ROOT>/cache/openvino`）；仅在显式设置 `OV_CACHE_DIR` 时覆盖默认值。
- RapidOCR v3 模型与字体资源需在镜像构建前预下载到本地路径（避免部署后在线下载）。
- RapidOCR 必须执行“本地模型强校验 + 缺失即失败”，移除线上下载回退逻辑。
- OCR 输入预处理必须基于 OpenCV BGR `numpy`（零拷贝优先）：连续 `uint8` 缓冲区直接透传，禁止引入 `PIL` 中转链。
- 默认应在启动后由单一 owner worker 做一次后台 RapidOCR 预热，把编译/冷加载成本前移，避免多 Worker 首个 OCR 请求直接命中冷启动超时。

### 3.4 InsightFace

- 固定为 **ONNX Runtime + OpenVINO Execution Provider**。
- 识别模型固定为 `antelopev2`；禁止切回 `buffalo_l`。
- InsightFace OpenVINO EP 的 `device_type` 基线必须使用 `MULTI` 表达式（默认 `MULTI:GPU,CPU`）；禁止继续把 `GPU_FP16` / `CPU_FP32` 直接透传给运行时。
- 不允许 silent fallback 到 CPUExecutionProvider；OpenVINO EP 不可用时必须直接报错。
- InsightFace OpenVINO EP 默认应显式传入 `cache_dir=<PROJECT_ROOT>/cache/openvino`，并把 `enable_opencl_throttling=false` 作为吞吐优先基线；需要保守模式时再通过环境变量覆盖。
- 必须兼容 `insightface` 旧版 `FaceAnalysis.__init__` 不支持 `providers/allowed_modules` 的情况；此时必须在会话级显式设置 `OpenVINOExecutionProvider` 并校验 provider 顺序。
- 对 `insightface` 旧版模型路由不兼容时，允许运行时构造仅含检测+识别必需 ONNX 的模型目录用于初始化，但不得改变 `/represent` 接口语义与返回字段。
- 对齐阶段必须使用 OpenCV + Intel OpenCL（`warpAffine` OpenCL 路径）；OpenCL 不可用或设备非 Intel 时必须直接报错，禁止静默回退 CPU。
- InsightFace 五点对齐仿射矩阵必须由仓库内本地实现生成，禁止继续依赖 `insightface.utils.face_align.estimate_norm` 内部已弃用的 `SimilarityTransform.estimate` 路径。
- 检测/识别模型输入的归一化与通道转换必须使用 OpenVINO PrePostProcessing (PPP) API，禁止继续依赖 `cv2.dnn.blobFromImage(s)`。

---

## 4. API 语义契约（以 `app/server.py` 为准）

> 任何重构都不能破坏以下端点语义和响应处理规则。

### 4.1 全局行为

- API Key 校验：
  - Header 名：`api-key`
  - `API_AUTH_KEY` 为空或 `"no-key"` 时跳过鉴权
  - 鉴权失败：HTTP 401，`detail="Invalid API key"`
  - 鉴权范围：除 `GET /` 外的所有业务端点
- 生命周期：
- 启动时初始化 `AIModels` 并启动常驻的 Text-CLIP 单例服务
  - 关闭时释放全部模型
- Text-CLIP 常驻内存；非文本模型（Vision-CLIP / OCR / InsightFace）默认基于空闲计时自动释放，当前基线 `NON_TEXT_IDLE_UNLOAD_SECONDS=300`，设为 `0` 时可禁用。
- 模型实例为空时：相关推理端点返回 HTTP 503（`"模型实例尚未初始化"`）。

### 4.2 图像读取辅助逻辑（`read_image_from_upload`）

- GIF：读取第一帧并转 BGR。
- GIF 解码优先使用 OpenCV 动画解码接口（`cv2.imdecodeanimation`，必要时回退 `cv2.imdecodemulti` / `cv2.imdecode`）读取首帧；禁止重新引入 `PIL` 中转链。
- 非 GIF：统一使用 `cv2.imdecode(..., IMREAD_UNCHANGED)`；禁止重新引入 `ffmpeg/ffprobe` 子进程解码链。
- 16-bit 图像：降为 8-bit。
- 宽高任一超过 `10000`：返回错误 `"height or width out of range"`。
- 统一输出 3 通道 BGR。
- 失败时返回 `(None, msg)`，而不是抛 HTTP 异常。

### 4.3 端点逐条契约

1. `GET /`
- 返回 HTML 状态页（服务运行信息）。

2. `POST /check`
- 响应模型：`CheckResponse`
- 返回字段：`result/title/help`
- `result` 固定 `"pass"`。

3. `POST /restart`
- 语义：触发按需模型释放（不重启进程）。
- 成功返回：`{"result":"pass"}`。

4. `POST /restart_v2`
- 语义：延迟 1 秒后 `os.execl` 重启进程。
- 立即返回：`{"result":"pass"}`。

5. `POST /ocr`
- 入参：`file`。
- 读图失败：`{"result":[],"msg":<错误信息>}`
- 成功：`{"result":<OCRResult>}`（成功时不返回 `msg`）
- 运行异常：`{"result":[],"msg":<异常文本>}`

6. `POST /clip/img`
- 入参：`file`。
- 读图失败：`{"result":[],"msg":<错误信息>}`
- 成功：`{"result":[<16位小数字符串>...]}`（成功时不返回 `msg`）
- 异常：`{"result":[],"msg":<异常文本>}`

7. `POST /clip/txt`
- 入参：`{"text": "..."}`
- 成功：`{"result":[<16位小数字符串>...]}`（成功时不返回 `msg`）
- 异常：`{"result":[],"msg":<异常文本>}`

8. `POST /represent`
- 入参：`file`。
- 读图失败：`{"result":[],"msg":<错误信息>}`
- 成功：
  - `{"detector_backend":"insightface","recognition_model":MODEL_NAME,"result":[...]}`
- 若异常包含 `set enforce_detection` 或 `Face could not be detected`：
  - 返回 `{"result":[]}`（无 `msg`）
- 其他异常：
  - 返回 `{"result":[],"msg":<异常文本>}`

---

## 5. 模型加载/卸载与调度策略（硬约束）

1. Text-CLIP 作为独立单例服务常驻内存，不参与 OCR / 图像 CLIP / 人脸模型的装卸切换。
2. `/clip/txt` 不再进入非文本推理队列，也不再依赖“文本优先级/回切窗口”状态机；文本请求始终走独立常驻实例。
3. 非文本模型（Vision-CLIP / OCR / InsightFace）仍采用队列化串行调度，并在单 worker 内保持“同一时刻一个主模型族”。
4. `/clip/img` 必须在单 worker 内做“相邻同尺寸请求”的受控微批；若尺寸不一致或队列被其他任务打断，必须立即退回单请求执行。
5. 非文本超时必须拆分为“排队超时”和“执行超时”；禁止继续用单个 `INFERENCE_TASK_TIMEOUT` 同时覆盖排队、切族、模型加载与执行，导致冷启动结构性误杀。
6. `WEB_CONCURRENCY` 不得放大 Text-CLIP 副本数；多 worker 下必须复用同一个后台 Text-CLIP 服务实例。
7. `POST /restart` 仅释放当前非文本模型族并清空空闲释放计时；常驻 Text-CLIP 服务保持可用，除非进程关闭或 `/restart_v2`。
8. 非文本模型默认在最后一次请求完成后进入空闲释放计时；倒计时期间若有新请求到达，必须续期而不是立即卸载。

---

## 6. 并发架构约束（硬约束）

- 推荐结构：**多进程 Worker + 有界队列 + 每 Worker 内批处理**。
- 禁止在单进程无限堆线程“硬顶并行度”。
- 必须控制总并行度：
  - `总并行度 = 进程数 × 每进程推理线程数`
- 原因：
  - OpenVINO CPU 插件/ONNX Runtime CPU 内部已有线程池；
  - 应用层过量线程会导致 oversubscription，吞吐下降且抖动增加。

---

## 7. AI 常见坏味道（黑名单：硬性禁止）

1. 修改 `app/QA-CLIP` 语义代码（非兼容性/纯性能优化）。
2. 继续使用 `/app/clip` 旧引用路径。
3. 把 RapidOCR 回退为 `rapidocr-openvino` 或对其模型私自量化。
4. 未经说明地变更端点响应字段、`msg` 出现时机、错误处理语义。
5. 无界队列、无上限线程、无证据地提高 worker/thread 参数。
6. 在同一进程长期同时常驻多个大模型，导致可避免的双份内存占用。
7. 只改代码不改文档/依赖，造成运行事实与文档不一致。

---

## 8. 抽象/封装/新依赖引入门槛（必须过闸）

当你想新增 helper / 抽象层 / 新依赖 / 新进程模型时，必须在回复中写“过闸理由”。

### 8.1 新增 helper（至少满足 2 条）

1. 同逻辑出现至少 2 处重复。
2. helper 承载明确边界（模型生命周期、队列调度、资源释放）。
3. helper 显著提升可测试性（便于 mock/回归）。
4. helper 显著降低复杂度（例如优先级队列状态机）。

### 8.2 新增依赖（至少满足 1 条）

1. 标准库/现有依赖无法实现同等能力。
2. 新依赖是对应后端的官方或主流实现（如 NNCF、ORT OpenVINO EP）。
3. 能给出明确收益与风险评估（性能、内存、兼容性）。

### 8.3 并发策略改动（必须同时满足）

1. 说明进程数、线程数、队列长度的设计依据。
2. 给出最小压测对比（至少延迟/吞吐/内存三项之一）。
3. 不破坏第 4 节端点语义与第 5 节模型调度约束。

---

## 9. 文档与依赖同步（强制）

- 涉及模型后端、推理设备、量化策略、端点返回行为的改动，必须同步修改：
  - `AGENTS.md`
  - `README.md`
  - `requirements.txt`
- 若引入/调整 RapidOCR OpenVINO 参数文件，需提供示例 `cfg_openvino_cpu.yaml` 并说明关键参数（含设备与批量策略）。

---

## 10. 开发/自检命令（至少执行到可验证）

- `python -V`（确认 3.12）
- 如需安装依赖，仅在明确允许联网安装时执行 `pip install -r requirements.txt`；默认不把它作为本仓库 Agent 自检步骤
- `python -m compileall app`
- `uvicorn server:app --host 0.0.0.0 --port 8060`（在 `app/` 目录）
- 关键端点冒烟：`/check`、`/clip/txt`、`/ocr`、`/represent`

---

## 11. 最终自检清单（必须逐条勾选）

- [ ] 是否保持 QA-CLIP 子库边界（未做语义改动，或已说明兼容/性能理由）
- [ ] 是否完全切换到 `app/QA-CLIP/clip` 引用路径
- [ ] 是否保持所有端点语义与响应处理兼容（含 `msg` 字段规则）
- [ ] QA-CLIP 是否固定为 ViT-L/14 且输出维度 768
- [ ] QA-CLIP 转换是否满足“无双份内存常驻 + FP16 压缩 + NNCF 约束”
- [ ] RapidOCR 是否为 `rapidocr==3.7.0` + OpenVINO(MULTI) + PP-OCRv5 mobile（Det/Rec）+ `use_cls=true`
- [ ] InsightFace 是否使用 ORT + OpenVINO EP
- [ ] 是否遵守“Text-CLIP 独立常驻单例 + 非文本单模型族串行切换”策略
- [ ] 是否采用多进程 Worker + 有界队列 + Worker 内批处理，并规避线程过度订阅
- [ ] 是否同步更新 `README.md` 与 `requirements.txt`

---

## 12. README / AGENTS 文档边界

- `README.md` 只保留最终用户直接需要的信息：部署前准备、运行时环境变量、Windows/Docker 部署、模型文件准备、设备检查、基础调用示例、许可证。
- 下列内容默认只应存在于 `AGENTS.md`，不要重新塞回 `README.md`，除非它直接影响最终用户操作：
  - 实现约束、后端选择原因、无 silent fallback 规则
  - 容器构建内部实践、镜像裁剪策略、依赖清理策略
  - 稳定性修复记录、兼容性说明、上线验收清单
  - `app/convert.py`、NNCF、IR 导出和模型转换流程说明
  - Agent/开发自检、压测、调度策略、文档同步要求

---

## 13. 当前部署实现现状与验收附录

### 13.1 Debian 容器当前实践

- 当前 Docker 基线镜像为 `python:3.12-slim-trixie`（Debian 13）。
- APT 镜像固定为 `https://mirrors.zju.edu.cn/debian/`，PyPI 镜像固定为 `https://mirrors.zju.edu.cn/pypi/web/simple`。
- `sources.list` 基线仅保留 `trixie`、`trixie-updates`、`trixie-security`；构建阶段允许临时增加 `sid` 源，仅用于安装 Intel GPU runtime 后立即清理。
- 构建阶段使用 `apt-get install --no-install-recommends`，并清理 apt 索引。
- 当前 `requirements.txt` 在 Python 3.12 / manylinux 下可直接使用 wheel 安装，不再默认保留 InsightFace 专用构建依赖。
- 服务以非 root 用户运行，可通过 `APP_UID` / `APP_GID` 对齐宿主机权限。
- 容器健康检查使用 `GET /`，且不依赖 API Key。
- 仓库应提供 `.dockerignore` 以降低构建上下文体积。
- 镜像内需包含 OpenVINO/OpenCL 运行基线依赖：`libdrm2`、`libze1`、`ocl-icd-libopencl1`、`mesa-opencl-icd`、`intel-opencl-icd`、`libze-intel-gpu1`，以及 Python/OpenCV 运行时基础库 `ca-certificates`、`libglib2.0-0`、`libgomp1`；`clinfo` 仅作为临时诊断工具，默认不随运行时镜像打包。
- Docker 构建阶段在 `pip install -r requirements.txt` 后，必须移除传递安装的 GUI 版 OpenCV（至少 `opencv-python`，如存在也移除 `opencv-contrib-python`），清理 pip 缓存，并显式安装 `opencv-python-headless`。
- 由于当前服务仅使用 OpenCV 的图像解码、色彩转换与 OpenCL/`warpAffine` 路径，镜像默认不再包含 `libgl1`、`libsm6`、`libxext6`、`libxrender1`，也不包含 `mesa-vulkan-drivers`、`intel-media-va-driver-non-free`、VAAPI、oneVPL、QSV 相关媒体栈依赖。
- Intel iGPU 固件属于宿主机职责；如宿主 Debian 13 需要固件，应在宿主机安装 `firmware-misc-nonfree`（兼容包名 `firmware-misc-non-free`），而不是打包进应用容器。
- 容器镜像不安装 `xserver-xorg-video-intel`（Xorg 显示栈组件，不属于无头推理运行基线）。
- Debian 13 容器若要启用 OpenVINO GPU，必须补齐 Intel compute runtime（`intel-opencl-icd` / `libze-intel-gpu1`）；推荐在构建阶段通过临时 sid 源 + pin 方式安装，并在镜像层清理 sid 源文件。
- 镜像内只打包 InsightFace `antelopev2` 模型，不保留 `buffalo_l` 分支。
- `docker-compose` 默认不挂载 `/models`，模型随镜像静态打包。
- 启动阶段必须增加 `/dev/dri` 自检：请求 GPU 推理时，`/dev/dri` 不可用则直接报错并终止启动。
- Debian 宿主若要启用 Intel iGPU，必须正确映射 `/dev/dri`，并确保 `VIDEO_GID` / `RENDER_GID` 与宿主机设备组一致（默认示例 `44/109`）。

### 13.2 最近稳定性修复（2026-03）

- 修复 `/clip/img` 在 OpenVINO 动态输入模型上的 PPP 构建失败：视觉预处理 `resize` 显式固定为 `224x224`，保持输出维度 `768` 与接口语义不变。
- 收敛上传读图链：`read_image_from_upload` 改为 OpenCV 原生解码，GIF 首帧优先走 `cv2.imdecodeanimation`，并移除 `ffmpeg/ffprobe` 与 `PIL` 运行时依赖。
- 修复 RapidOCR 后端锁定问题：服务会直接把 `RAPIDOCR_OPENVINO_CONFIG_PATH` 作为 `config_path` 传给 `RapidOCR`，并在初始化后校验 `Det/Cls/Rec.engine_type=openvino`，避免回落到默认 ORT 配置。
- 修复 RapidOCR 设备覆盖优先级：显式环境变量（如 `RAPIDOCR_DEVICE=MULTI:GPU,CPU`）优先级高于 `cfg_openvino_cpu.yaml`，避免 YAML 旧值覆盖运行时设置。
- 收敛 OpenVINO cache 配置：默认启用仓库内 `cache/openvino` 编译缓存目录，显式设置 `OV_CACHE_DIR` 时覆盖默认值，避免多 Worker 反复冷编译。
- 修复 InsightFace PPP 预处理输出取值失败：改为显式 `infer({compiled_model.input(0): ...})` + `compiled_model.output(0)` 取输出，并在构建时执行 shape/dtype 自检，兼容匿名 `Result` 端口映射。
- 收敛 RapidOCR 检测缩放策略：默认 `Det.limit_type=max`，避免把小图放大到 `limit_side_len` 触发结构性慢路径。
- 修复 InsightFace OpenCL 对齐补丁兼容性：`norm_crop` 现在兼容 `UMat/ndarray` 输入，并对 `estimate_norm` 返回值执行显式矩阵校验，避免 `warpAffine` 参数类型错误。
- 收敛 Text-CLIP 调度：文本模型改为独立常驻单例 RPC 服务，多 worker 下共享同一后台实例，不再参与非文本队列插队与回切。
- 收敛 `/clip/img` 吞吐路径：视觉请求改为单 worker 内相邻同尺寸微批，维持 PPP 预处理与接口语义不变，同时提升 GPU 利用率。
- 收敛 OCR 冷启动：Text-CLIP owner worker 会在启动后后台预热一次 RapidOCR，并通过 OpenVINO cache 把编译成本前移。
- 修复 InsightFace 五点对齐弃用 API warning：本地重写相似变换矩阵估计，保持 OpenCL `warpAffine` 路径与输出语义不变。
- 修复 RapidOCR 方向分类批次越界：对 OpenVINO 返回结果按当前批次长度裁剪，并移除 crop 深拷贝以降低 OCR CPU 开销。
- 恢复非文本模型族空闲自动卸载：OCR / Vision-CLIP / InsightFace 默认在 `NON_TEXT_IDLE_UNLOAD_SECONDS=300` 到期后释放，文本 CLIP 仍常驻。
- 收敛吞吐基线：RapidOCR 默认配置调整为 `THROUGHPUT + performance_num_requests=2 + num_streams=2 + rec/cls batch=8`；InsightFace OpenVINO EP 默认补齐 `cache_dir` 并关闭 OpenCL throttling。
- 收敛 OpenVINO 同步推理调用：InsightFace PPP runner、RapidOCR OpenVINO patch 与 CLIP 本地输入路径统一改为显式 `set_input_tensor(...) + infer() + get_output_tensor(0)`，规避匿名 `Result` 端口映射兼容问题并减少共享字典分发开销。
- 收敛非文本超时语义：`INFERENCE_TASK_TIMEOUT` 仅作为兼容基线，运行时拆分为 `INFERENCE_QUEUE_TIMEOUT` 与 `INFERENCE_EXEC_TIMEOUT`，避免排队时间挤占执行窗口。
- 修复 InsightFace 旧版本兼容问题：当 `FaceAnalysis.__init__` 不支持 `providers` 参数时，运行时显式强制 `OpenVINOExecutionProvider`，并兼容旧路由器仅加载检测+识别必需模型文件。
- 修复 QA-CLIP GPU Remote Context 初始化兼容问题：当 `get_default_context("GPU")` 失败时，继续尝试具体 `GPU.*` 设备与 `create_context("GPU", {})` 兼容路径；若仍无法得到 GPU Remote Context，保持硬失败，不允许 silent fallback。
- 修复 QA-CLIP 在 `available_devices=['CPU']` 误报场景下的提前退出：即使设备枚举未列出 GPU，也继续执行 Remote Context 显式探测；仅在全部 GPU 上下文路径均失败后再硬失败。
- 收敛 Docker 依赖分类：当前 `requirements.txt` 在 Python 3.12 / manylinux 下可全量使用 wheel 安装，`build-essential`、`gcc`、`g++`、`libpq-dev` 不再作为 InsightFace 构建依赖保留在镜像中。

### 13.3 `/dev/dri` Intel iGPU 上线验收

#### A. `docker-compose.yml` 必备参数

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
      - INFERENCE_DEVICE=AUTO
      - CLIP_INFERENCE_DEVICE=AUTO
      - RAPIDOCR_DEVICE=MULTI:GPU,CPU
      - INSIGHTFACE_OV_DEVICE=MULTI:GPU,CPU
      - OPENCV_OPENCL_DEVICE=Intel:GPU:0
      - WEB_CONCURRENCY=1
      - INFERENCE_QUEUE_MAX_SIZE=64
      - INFERENCE_TASK_TIMEOUT=10
      - INFERENCE_EXEC_TIMEOUT=30
```

说明：
- `INFERENCE_DEVICE` 可保持 `AUTO`，`CLIP_INFERENCE_DEVICE` 推荐使用 `AUTO`；非文本 OpenVINO 路径（RapidOCR / InsightFace / PPP）推荐使用 `MULTI:GPU,CPU`，且验收时仍要求 GPU 实际可用。
- `VIDEO_GID/RENDER_GID` 必须与宿主机 `video/render` 组一致。

#### B. 启动与设备可见性判定

通过标准：
- `docker compose ps` 显示容器 `Up`。
- `docker inspect <container> --format '{{.State.Health.Status}}'` 返回 `healthy`。
- 启动日志包含“启动自检通过：GPU 设备节点可访问”。
- 容器内 `ls -l /dev/dri` 可见 `card*` 与 `renderD*` 节点。
- 容器用户对 `/dev/dri` 节点具备读写权限。

失败判定（任一命中即失败）：
- 日志出现“已请求 GPU 推理，但容器内不存在 /dev/dri”。
- 日志出现“/dev/dri 未发现 card*/renderD* 节点”。
- 日志出现“/dev/dri 设备权限不足”。

#### C. 端点语义与后端判定

通过标准：
- `POST /check` 返回 `{"result":"pass", ...}`。
- `POST /clip/txt` 成功返回 `768` 维字符串数组（成功无 `msg`）。
- `POST /clip/img` 成功返回 `768` 维字符串数组（成功无 `msg`）。
- `POST /ocr` 成功返回 `{"result":{"texts","scores","boxes"}}`（成功无 `msg`）。
- `POST /represent` 返回 `{"detector_backend":"insightface","recognition_model":"antelopev2","result":[...]}` 或在人脸不存在时 `result=[]`。

失败判定（任一命中即失败）：
- 日志出现 `No silent fallback is allowed`（排除主动构造失败用例）。
- 日志出现 `OpenCL is unavailable` 或非 Intel OpenCL 设备错误。
- 日志出现 `provider validation failed` 或 `OpenVINOExecutionProvider` 非首位。

#### D. 回归脚本建议（最小）

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=200 mt-photos-ai-openvino

curl -s http://127.0.0.1:8060/
curl -s -X POST http://127.0.0.1:8060/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8060/clip/txt -H "api-key: mt_photos_ai_extra" -H "Content-Type: application/json" -d '{"text":"smoke"}'
```

---

## 14. `app/convert.py` 运行环境

| 环境变量 | 可选值 | 默认值 |
|---|---|---|
| `PROJECT_ROOT` | 项目根目录路径 | 自动探测 |
| `MODEL_PATH` | 模型根目录路径（导出会写入 `qa-clip/openvino`） | `<PROJECT_ROOT>/models` |
| `HF_CACHE_DIR` | Hugging Face 缓存目录路径 | `<PROJECT_ROOT>/cache/huggingface` |
| `QA_CLIP_ENABLE_NNCF_WEIGHT_COMPRESSION` | `0`（关闭）或 `1`（开启） | `0` |
| `QA_CLIP_NNCF_WEIGHT_MODE` | NNCF 压缩模式名（常见：`INT8_ASYM` / `INT8_SYM` / `NF4` / `E2M1`） | `INT8_ASYM` |

`convert.py` 在未预设时还会自动设置以下变量：`HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`、`HF_HUB_DISABLE_SYMLINKS_WARNING`。
