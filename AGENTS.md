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
- Debian 容器镜像源基线：APT 使用 `https://mirrors.tuna.tsinghua.edu.cn/debian/`，PyPI 使用 `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。
- 硬件基线：Intel i7-11800H（AVX512 VNNI + Xe 核显，共享内存架构）。
- 主服务入口：`app/server.py`（当前仓库中等效于历史 `server_openvino.py` 的实现入口）。
- Text-CLIP 独立服务入口：`text-clip/app/server.py`。
- Windows 本地 CUDA Image-CLIP 并行子项目命令行入口：`image-clip/starter.py`；服务实现入口：`image-clip/app/server.py`；依赖文件为 `image-clip/requirement.txt`。
- 模型编排：主服务使用 `app/models/`（入口 `app/models/runtime.py`，按 `clip_image.py`、`rapidocr_lib.py`、`insightface.py` 拆分）；独立 Text-CLIP 服务代码位于 `text-clip/app/models/`。
- 模型转换：`scripts/convert.py`（QA-CLIP -> OpenVINO IR）。
- 模型目录：`models/qa-clip/openvino`、`models/insightface/models`。
- Text-CLIP tokenizer 资源：`text-clip/app/models/QA-CLIP/clip`（保留 `bert_tokenizer.py` 与 `vocab.txt`）。
- 配置存储：`app/config`。
- 参考文件（对齐端点用）：`example/`。

---

## 2. QA-CLIP tokenizer 资源边界（硬约束）

1. `text-clip/app/models/QA-CLIP/clip` 当前仅承载 Text-CLIP 所需 tokenizer 与词表资源。
2. 允许改动仅限：
  - 兼容性修复（Python/OpenVINO/依赖版本适配）
  - 只优化性能且不改变 tokenizer 语义的改动
3. Text-CLIP 若仍引用 QA-CLIP tokenizer 资源，必须从 `text-clip/app/models/QA-CLIP/clip` 引用；禁止继续依赖历史 `/app/clip` 路径。
4. 若确需改动这些资源，必须在提交说明里写清楚：
  - 改动类型（兼容性 or 性能）
  - 不改语义的证据（分词规则/词表/输出维度基线）

---

## 3. 模型与后端基线（硬约束）

### 3.1 QA-CLIP（图文向量）

- 模型固定为：`TencentARC/QA-CLIP-ViT-L-14`。
- 向量维度固定：`768`。
- `image-clip/app` 作为并行本地开发子项目时，可独立使用 `PyTorch + CUDA` 提供 `/clip/img`；但它不改变主服务 `/clip/img` 的 OpenVINO 基线、接口语义和部署方式。
- 主服务 `/clip/img` 推理目标设备：`GPU`（Intel Xe 核显），优先减少 Host<->Device 数据搬运。
- 主服务 `/clip/img` 的 OpenVINO 侧优先启用 Remote Tensor API 相关互操作能力（零拷贝/少拷贝优先）。
- Text-CLIP 必须拆到独立容器，代码位于 `text-clip/app`；主容器**不得**再保留本地 Text-CLIP 模型实例、RPC 子服务或文本 tokenizer 运行链。
- 独立 Text-CLIP 容器固定使用 OpenVINO `CPU`；**禁止**为它保留 GPU Remote Context、`/dev/dri` 依赖或 Intel GPU runtime 裁剪以外的冗余包。
- 主服务不得再提供 `/clip/txt`；该端点仅允许由独立 Text-CLIP 服务暴露。
- 当 `CLIP_INFERENCE_DEVICE=AUTO` 时，必须强制初始化 GPU Remote Context；初始化失败必须直接报错，禁止 silent fallback。
- 当 `CLIP_INFERENCE_DEVICE` 显式包含 `GPU`（如 `GPU`、`AUTO:GPU,CPU`）时，也必须显式完成 GPU Remote Context 初始化；失败直接报错，禁止静默继续。
- `/clip/img` 必须支持 **标准预处理后的受控批处理**：单张请求先完成缩放、中心裁剪与 PPP 归一化，再按 `CLIP_IMAGE_BATCH` 聚合成批；批处理不得改变单请求输入输出语义。
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
- RapidOCR 当前运行基线收敛为 **库内原生 OpenVINO CPU 路径**；`RAPIDOCR_DEVICE` 默认与实际运行时都必须为 `CPU`，显式传入 `AUTO/GPU` 仅允许记录告警后强制回到 `CPU`。
- `RAPIDOCR_DET_DEVICE/RAPIDOCR_CLS_DEVICE/RAPIDOCR_REC_DEVICE` 仅保留兼容环境变量名，当前实现**不再参与**运行时 stage 选路；禁止再恢复本地 stage 级 session 包装。
- 默认使用 **PP-OCRv5 mobile** 模型配置（`Det/Rec`）。
- 默认开启方向分类器（`Global.use_cls=true`），并预置分类模型。
- 运行基线：`max_side_len=960`、`Det.limit_side_len=960`、`Det.limit_type=max`、`Rec.rec_batch_num=8`、`Cls.cls_batch_num=8`。
- OpenVINO 参数基线：`device_name=CPU`、`performance_hint=THROUGHPUT`、`performance_num_requests=2`、`inference_num_threads=-1`、`num_streams=2`。
- `RAPIDOCR_PERFORMANCE_NUM_REQUESTS` 当前除透传给 RapidOCR/OpenVINO 外，还作为 OCR 多实例池与应用层执行器默认 worker 数基线。
- 配置优先级必须为：**显式环境变量 `RAPIDOCR_*` > YAML(`cfg_openvino_cpu.yaml`) > 代码默认值**；但设备相关项最终仍必须收敛到 `CPU`。
- 示例参数文件为 `app/config/cfg_openvino_cpu.yaml`；关键配置项包括 `device_name`、`inference_num_threads`、`performance_hint`、`performance_num_requests`、`enable_cpu_pinning`、`num_streams`、`enable_hyper_threading`、`scheduling_core_type`。
- 必须启用模型编译缓存，降低冷启动与多 Worker 反复编译开销。
- 默认缓存目录应收敛到仓库内可写路径（当前基线 `<PROJECT_ROOT>/cache/openvino`）；仅在显式设置 `OV_CACHE_DIR` 时覆盖默认值。
- RapidOCR v3 模型与字体资源需在镜像构建前预下载到本地路径（避免部署后在线下载）。
- RapidOCR 必须执行“本地模型强校验 + 缺失即失败”，移除线上下载回退逻辑。
- RapidOCR 与 OpenVINO 的衔接必须直接走库原生实现；严禁 monkey patch 第三方类/模块，也禁止继续替换 `text_det/text_cls/text_rec.session` 或自定义 stage session 包装。
- OCR 运行链必须直接委托 `RapidOCR.__call__` / 库内 `run_ocr_steps`，不要再维护仓库内自定义 det/cls/rec 预处理、批调度和输出拼装分支。
- OCR 并发必须通过**有界 RapidOCR 多实例池**实现；每个 worker 独占一个库原生 `RapidOCR` 实例，禁止多个线程共享同一个 OpenVINO `InferRequest`/session 对象后再用全局锁硬串行。
- OCR 输入预处理必须基于 OpenCV BGR `numpy`（零拷贝优先）：连续 `uint8` 缓冲区直接透传，禁止引入 `PIL` 中转链。
- OCR 执行超时允许通过 `OCR_EXEC_TIMEOUT` 单独覆盖；默认不得低于 `30s`，且异步路径中模型加载/切换等待不得挤占 OCR 纯执行超时窗口。
- OCR 必须提供应用层有界准入；执行超时后必须先触发协作取消，再等待已受理任务退场，禁止把超时任务脱离调用方继续在后台无界堆积。
- 默认禁止在启动后自动拉起 RapidOCR；OCR 只允许在首次 `/ocr` 请求时进入内存。
- 如显式设置 `OCR_PREWARM_ENABLED=true`，只允许做一次性后台预热并在完成后立即释放 OCR 模型；预热线程不得在 `/restart` 或释放后把 OCR 再次拉回内存。
- 默认应在连续 `60s` 未收到业务请求时自动释放主容器内的 Vision-CLIP / OCR / InsightFace；独立 Text-CLIP 容器不参与这一路径。允许通过 `NON_TEXT_IDLE_RELEASE_SECONDS` 覆盖，`<=0` 表示关闭该兜底释放。

### 3.4 InsightFace

- 固定为 **ONNX Runtime + OpenVINO Execution Provider**。
- 识别模型固定为 `antelopev2`；禁止切回 `buffalo_l`。
- InsightFace OpenVINO EP 的 `device_type` 基线为 `AUTO`；禁止继续把 `GPU_FP16` / `CPU_FP32` 直接透传给运行时；当 `AUTO` 且 GPU 可见时，运行时必须显式收敛到 `GPU`，并在日志里输出 `configured_device/runtime_device/preprocess_device/provider_runtime`。
- 不允许 silent fallback 到 CPUExecutionProvider；OpenVINO EP 不可用时必须直接报错。
- InsightFace OpenVINO EP 默认应显式传入 `cache_dir=<PROJECT_ROOT>/cache/openvino`，并把 `enable_opencl_throttling=false` 作为吞吐优先基线；需要保守模式时再通过环境变量覆盖。
- 必须固定到新版 `FaceAnalysis(name=..., root=..., allowed_modules=..., providers=..., provider_options=...)` 初始化路径；禁止 compat dataclass、构造参数重试、source fallback、`set_providers` 后置修正。
- InsightFace 运行时模型目录固定为 `<MODEL_PATH>/insightface/_runtime_models/models/antelopev2`；禁止再维护额外的兼容目录、双 root 或 source/runtime 回退。
- InsightFace 必须支持 `INSIGHTFACE_OV_DEVICE=GPU|CPU|AUTO` 三种显式推理模式；无论推理设备取值为何，除 ORT OpenVINO EP 推理外，其余路径都应尽量委托 `insightface` 原生 CPU 实现：检测走 `det_model.detect`，识别特征提取走 `rec_model.get_feat`，且 OpenVINOExecutionProvider 必须保持首位 provider。
- `/represent` 当前固定为**单个 detector/recognition lane**；检测保持单请求原生路径，允许扩的是按请求分组的人脸对齐与 recognition 入批前的 CPU 预处理并行度，禁止把 ORT/OpenVINO detection/recognition backend 直接扩成多 lane。
- `/represent` 应用层准入与聚合上限当前固定收敛为 `4`；若共享图片总名额更小，则继续受共享名额池截断。
- InsightFace 预处理执行器当前固定收敛为 `4` 个 worker 上限，并继续复用原生 detector/recognition CPU 预处理 + 本地 OpenCV 对齐路径；禁止再暴露与单 lane 事实不一致的 `INSIGHTFACE_MAX_WORKERS`、`INSIGHTFACE_MAX_CONCURRENT_REQUESTS`、`INSIGHTFACE_BATCH_SIZE` 一类环境变量。
- 同一张输入图的对齐阶段必须复用单次准备后的连续 `numpy BGR` 源图对象，禁止重复拷贝链。
- InsightFace 五点对齐仿射矩阵必须由仓库内本地实现生成，禁止继续依赖 `insightface.utils.face_align.estimate_norm` 内部已弃用的 `SimilarityTransform.estimate` 路径。
- 当 `antelopev2/glintr100.onnx` 输出元数据仍声明静态 `{1,512}` 时，运行时必须在受控 runtime root 中把识别输出 batch 维修正为动态后再初始化 ORT session；禁止通过退回逐张识别来规避 `VerifyOutputSizes` warning。
- `scrfd_10g_bnkps.onnx` 当前保持原生 detector 路径，只复制到受控 runtime root，不再维护 detector 动态 batch 元数据补丁或仓库内 batched detector head 解包逻辑。
- FaceAnalysis 用于模型发现、provider 管理、session 生命周期以及原生 detect/recognition 调度；仓库内仅保留服务级队列/准入/结果转换与本地对齐矩阵实现，严禁 monkey patch `insightface` 模块或模型实例方法。
- `/represent` 必须提供应用层有界准入、单 lane OpenVINO EP 执行 worker、有限预处理 worker 与专用有界批队列；允许跨请求聚合 recognition batch 并平滑调度检测，但不得改变单请求响应语义、字段和错误处理规则。

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
- 启动时初始化主服务 `AIModels`
  - 关闭时释放全部模型
- 服务日志必须在 `uvicorn server:app` 与 `python server.py` 两种启动路径下都稳定输出到控制台；Windows 直跑时可额外写入 `<PROJECT_ROOT>/server.log`，但不能替代控制台输出。
- `LOG_LEVEL` 必须同时作用于 `mt_photos_ai.*`、`uvicorn.*` 与当前接入的第三方运行日志；若手动执行 `uvicorn server:app`，其最早期 bootstrap 日志仍需通过 CLI `--log-level` 对齐。
- 服务必须固定为**单进程**；禁止继续暴露 `WEB_CONCURRENCY` 一类 worker 配置项，也禁止让第二个服务进程在同一工作目录下成功启动。
- 主容器不得常驻 Text-CLIP 模型；Vision-CLIP / OCR / InsightFace 按需懒加载，并采用“单活非文本模型族”切换：切换到新模型族前，必须等待当前模型族任务退场并同步释放旧族模型。独立 Text-CLIP 容器常驻其自身模型即可。
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
- 语义：同步等待当前非文本任务退场并释放 Vision-CLIP / OCR / InsightFace（不重启进程）。
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

7. `POST /represent`
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

1. Text-CLIP 必须作为独立容器服务常驻内存，固定走 CPU；主容器不得再持有本地 Text-CLIP 模型实例。
2. `/clip/img` 使用独立批队列；单张请求先完成标准预处理与 PPP 归一化，再按 `CLIP_IMAGE_BATCH` 受控聚合。
3. `/clip/img` 批队列实现必须基于 `asyncio.Queue`，禁止继续回到 `Condition + deque + 轮询` 的手写实现。
4. 非文本模型族必须采用“单活租约”切换：同一时刻只允许一个活跃的 Vision-CLIP / OCR / InsightFace 模型族常驻；切换前必须等待旧族已受理任务完全退场并同步卸载旧族模型。
5. 非文本模型族切换状态机必须显式建模，当前基线为 `transitions`；禁止再回到多布尔位 + 条件变量拼接出的隐式状态机。状态机事件必须由 `transitions` 绑定或由显式包装器触发，禁止在 model 上预先定义同名 `NotImplementedError` 占位方法覆盖 trigger 入口。
6. RapidOCR 必须拆成检测 / 分类 / 识别三个独立异步阶段；这些阶段是 OCR 模型族内部并行，不得绕开第 4 条把 OCR 与其他非文本模型族并行常驻。
7. InsightFace 使用独立执行路径；但其准入必须遵守第 4 条，不得在 Vision-CLIP 或 OCR 仍持有活跃租约时并行常驻。
8. 非文本超时必须拆分为“排队超时”和“执行超时”；禁止继续用单个 `INFERENCE_TASK_TIMEOUT` 同时覆盖全部阶段。
9. `/represent` 必须通过专用有界批队列平滑跨请求调度，并在 InsightFace 模型族内部聚合识别批；不得绕开第 4 条让多个非文本模型族并行常驻。
10. 非文本空闲释放计时只允许由 `/clip/img`、`/ocr`、`/represent` 刷新；`/check`、`/restart`、`/restart_v2` 不得阻止 Vision-CLIP / OCR / InsightFace 自动释放。
11. `POST /restart` 返回前必须完成 Vision-CLIP / OCR / InsightFace 的同步释放；独立 Text-CLIP 服务保持可用，除非它自己的容器被关闭或重启。
12. 关闭路径必须等待已受理的 Vision-CLIP / OCR / InsightFace 任务退场后，再回收执行器与 native runtime 引用。
13. `/clip/img`、`/ocr`、`/represent` 必须共享同一个应用层图片准入名额池；已受理图片总量（排队 + 执行）硬上限为 `10`，超出时必须立即失败，禁止继续挂起等待导致 MT-Photos 客户端超时取消。

---

## 6. 并发架构约束（硬约束）

- 推荐结构：**单进程 FastAPI 异步服务 + 有界批队列/阶段执行器**。
- 禁止在单进程无限堆线程“硬顶并行度”。
- 必须在运行时对单进程做硬约束；同一工作目录下的第二个服务进程必须因运行锁直接失败，而不是并行持有另一份非文本模型。
- 非文本模型族准入必须显式串行化，确保“单活模型族 + 独立 Text-CLIP 容器”的内存上界可控。
- 必须控制总并行度：
  - `总并行度 = 各阶段执行器线程数之和`
- 原因：
  - OpenVINO CPU 插件/ONNX Runtime CPU 内部已有线程池；
  - 应用层过量线程会导致 oversubscription，吞吐下降且抖动增加。

---

## 7. AI 常见坏味道（黑名单：硬性禁止）

1. 修改 `text-clip/app/models/QA-CLIP/clip` 语义代码（非兼容性/纯性能优化）。
2. 继续使用 `/app/clip` 旧引用路径。
3. 把 RapidOCR 回退为 `rapidocr-openvino` 或对其模型私自量化。
4. 未经说明地变更端点响应字段、`msg` 出现时机、错误处理语义。
5. 无界队列、无上限线程、无证据地提高 worker/thread 参数。
6. 在同一进程长期同时常驻多个大模型，导致可避免的双份内存占用。
7. 只改代码不改文档/依赖，造成运行事实与文档不一致。
8. 通过 monkey patch 修改 `rapidocr` / `insightface` / OpenVINO 相关第三方模块或对象方法语义。

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
  - `image-clip/requirement.txt`（若改动独立 Windows CUDA Image-CLIP 子项目）
  - `text-clip/requirement.txt`（若改动独立 Text-CLIP 容器）
- 若引入/调整 RapidOCR OpenVINO 参数文件，需提供示例 `cfg_openvino_cpu.yaml` 并说明关键参数（含设备与批量策略）。

---

## 10. 开发/自检命令（至少执行到可验证）

- 开发机本地验证时，所有后端统一设为 `CPU`：`INFERENCE_DEVICE=CPU`、`CLIP_INFERENCE_DEVICE=CPU`、`RAPIDOCR_DEVICE=CPU`、`INSIGHTFACE_OV_DEVICE=CPU`。
- `python -V`（确认 3.12）
- 如需安装依赖，仅在明确允许联网安装时执行 `pip install -r requirements.txt`；默认不把它作为本仓库 Agent 自检步骤
- `python -m compileall app`
- `python -m compileall text-clip/app`
- `python -m compileall scripts`
- `python -m compileall image-clip tests`
- `cd image-clip && python starter.py`
- `python scripts/smoke_image_clip.py --device cuda`（独立 Windows 本地 CUDA Image-CLIP 子项目）
- `python -m unittest discover -s tests -p "test_image_clip_starter.py"`
- `docker build -t mt-photos-ai-openvino .`
- `docker build -f text-clip/DockerFile-TextCLIP -t mt-photos-ai-text-clip .`
- `docker run --rm -it -e INFERENCE_DEVICE=CPU -e CLIP_INFERENCE_DEVICE=CPU -e RAPIDOCR_DEVICE=CPU -e INSIGHTFACE_OV_DEVICE=CPU mt-photos-ai-openvino python scripts/smoke_insightface.py --device CPU`
- `uvicorn server:app --host 0.0.0.0 --port 8060`（在 `app/` 目录）
- `uvicorn server:app --host 0.0.0.0 --port 8061`（在 `text-clip/app/` 目录）
- 如需验证 `PORT` / `LOG_LEVEL` 这类由服务包装层处理的环境变量，可在 `app/` 目录执行 `python server.py`；若继续手动执行 `uvicorn server:app`，需显式传 `--port` / `--log-level`
- 如需验证独立 Text-CLIP 服务的 `PORT` / `LOG_LEVEL`，可在 `text-clip/app/` 目录执行 `python server.py`
- 关键端点冒烟：主服务 `/check`、`/clip/img`、`/ocr`、`/represent`；独立 Text-CLIP 服务 `/check`、`/clip/txt`

---

## 11. 最终自检清单（必须逐条勾选）

- [ ] 是否保持 Text-CLIP tokenizer 资源边界（未做语义改动，或已说明兼容/性能理由）
- [ ] 若仍需要 tokenizer 资源，是否完全切换到 `text-clip/app/models/QA-CLIP/clip` 引用路径
- [ ] 是否保持所有端点语义与响应处理兼容（含 `msg` 字段规则）
- [ ] QA-CLIP 是否固定为 ViT-L/14 且输出维度 768
- [ ] QA-CLIP 转换是否满足“无双份内存常驻 + FP16 压缩 + NNCF 约束”
- [ ] RapidOCR 是否为 `rapidocr==3.7.0` + OpenVINO（当前固定走库内原生 CPU 路径）+ PP-OCRv5 mobile（Det/Rec）+ `use_cls=true`
- [ ] InsightFace 是否使用 ORT + OpenVINO EP（仅推理）+ 原生 CPU 检测/识别预处理
- [ ] 是否遵守“Text-CLIP 独立 CPU 容器 + 主容器非文本单模型族串行切换”策略
- [ ] 是否保持 OCR 默认懒加载，且显式预热后会立即释放
- [ ] 是否采用单进程 FastAPI 异步服务 + 有界批队列/阶段执行器，并规避线程过度订阅
- [ ] 是否同步更新 `README.md`、`requirements.txt` 与 `text-clip/requirement.txt`

---

## 12. README / AGENTS 文档边界

- `README.md` 只保留最终用户直接需要的信息：部署前准备、运行时环境变量、Windows/Docker 部署、模型文件准备、设备检查、基础调用示例、许可证。
- 下列内容默认只应存在于 `AGENTS.md`，不要重新塞回 `README.md`，除非它直接影响最终用户操作：
  - 实现约束、后端选择原因、无 silent fallback 规则
  - 容器构建内部实践、镜像裁剪策略、依赖清理策略
  - 稳定性修复记录、兼容性说明、上线验收清单
  - `scripts/convert.py`、NNCF、IR 导出和模型转换流程说明
  - Agent/开发自检、压测、调度策略、文档同步要求

---

## 13. 当前部署实现现状与验收附录

### 13.1 Debian 容器当前实践

- 当前 Docker 基线镜像为 `python:3.12-slim-trixie`（Debian 13）。
- APT 镜像固定为 `https://mirrors.tuna.tsinghua.edu.cn/debian/`，PyPI 镜像固定为 `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。
- `sources.list` 基线仅保留 `trixie`、`trixie-updates`、`trixie-security`；构建阶段允许临时增加 `sid` 源，仅用于安装 Intel GPU runtime 后立即清理。
- 构建阶段使用 `apt-get install --no-install-recommends`，并清理 apt 索引。
- 服务以非 root 用户运行，可通过 `APP_UID` / `APP_GID` 对齐宿主机权限。
- 容器健康检查使用 `GET /`，且不依赖 API Key。
- 仓库应提供 `.dockerignore` 以降低构建上下文体积。
- 镜像内需包含 OpenVINO/OpenCL 运行基线依赖：`libdrm2`、`libze1`、`ocl-icd-libopencl1`、`mesa-opencl-icd`、`intel-opencl-icd`、`libze-intel-gpu1`，以及 Python/OpenCV 运行时基础库 `ca-certificates`、`libglib2.0-0`、`libgomp1`；`clinfo` 仅作为临时诊断工具，默认不随运行时镜像打包。
- 独立 Text-CLIP 镜像固定使用 CPU；其基线依赖仅保留 Python/OpenVINO CPU 运行所需最小集合，不安装 Intel GPU runtime，也不映射 `/dev/dri`。
- Docker 构建阶段在 `pip install -r requirements.txt` 后，必须移除传递安装的 GUI 版 OpenCV（至少 `opencv-python`，如存在也移除 `opencv-contrib-python`），清理 pip 缓存，并显式安装 `opencv-python-headless`。
- 由于当前服务仅使用 OpenCV 的图像解码、色彩转换与 CPU `resize`/`warpAffine` 路径，镜像默认不再包含 `libgl1`、`libsm6`、`libxext6`、`libxrender1`，也不包含 `mesa-vulkan-drivers`、`intel-media-va-driver-non-free`、VAAPI、oneVPL、QSV 相关媒体栈依赖。
- Intel iGPU 固件属于宿主机职责；如宿主 Debian 13 需要固件，应在宿主机安装 `firmware-misc-nonfree`（兼容包名 `firmware-misc-non-free`），而不是打包进应用容器。
- 容器镜像不安装 `xserver-xorg-video-intel`（Xorg 显示栈组件，不属于无头推理运行基线）。
- Debian 13 容器若要启用 OpenVINO GPU，必须补齐 Intel compute runtime（`intel-opencl-icd` / `libze-intel-gpu1`）；推荐在构建阶段通过临时 sid 源 + pin 方式安装，并在镜像层清理 sid 源文件。
- 主服务镜像不再打包 `openvino_text_fp16.*`；这些文件仅应进入独立 Text-CLIP 镜像。
- 镜像内只打包 InsightFace `antelopev2` 模型，不保留 `buffalo_l` 分支。
- `docker-compose` 默认不挂载 `/models`，模型随镜像静态打包。
- 启动阶段必须增加 `/dev/dri` 自检：请求 GPU 推理时，`/dev/dri` 不可用则直接报错并终止启动。
- Debian 宿主若要启用 Intel iGPU，必须正确映射 `/dev/dri`，并确保 `VIDEO_GID` / `RENDER_GID` 与宿主机设备组一致（默认示例 `44/109`）。

### 13.2 最近稳定性修复（2026-03）

- InsightFace 已进一步简化为“原生 detector.detect + 本地五点对齐 + 原生 get_feat”的 CPU 执行链，仓库内不再维护 batched detector head 解包与 detector 动态 batch 补丁；当前回归重点收敛为 `/represent` 语义一致性、recognition 批处理一致性和 release/reload 生命周期。


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
      - RAPIDOCR_DEVICE=CPU
      - INSIGHTFACE_OV_DEVICE=AUTO
      - CLIP_IMAGE_BATCH=8
      - INFERENCE_QUEUE_MAX_SIZE=10
      - INFERENCE_TASK_TIMEOUT=10
      - INFERENCE_EXEC_TIMEOUT=30
      - OCR_EXEC_TIMEOUT=30
      - NON_TEXT_IDLE_RELEASE_SECONDS=60
      - PORT=8060
      - LOG_LEVEL=WARNING
  mt-photos-ai-text-clip:
    environment:
      - API_AUTH_KEY=mt_photos_ai_extra
      - MODEL_PATH=/models
      - OV_CACHE_DIR=/cache/openvino
      - PORT=8061
      - LOG_LEVEL=WARNING
```

说明：
- `INFERENCE_DEVICE` 可保持 `AUTO`，`CLIP_INFERENCE_DEVICE` 推荐使用 `AUTO`；非文本 OpenVINO 路径会在 GPU 可见时按 GPU 优先收敛，但 RapidOCR 当前固定走库内原生 `CPU` 路径；InsightFace 仅推理侧 EP 会收敛到 `GPU`，仓库内预处理固定走 `CPU`。
- `mt-photos-ai-text-clip` 固定走 CPU，并独立对外提供 `/clip/txt`；该容器不需要 `/dev/dri`、`VIDEO_GID` 或 `RENDER_GID`。
- `/represent` 当前固定为单 lane OpenVINO EP 推理 + 4 请求聚合预算 + 4 路 CPU 预处理 worker；如需权衡吞吐与尾延迟，只允许小幅调整 `INSIGHTFACE_BATCH_WAIT_MS`，不要重新引入额外的 worker/batch 容量环境变量。
- `INFERENCE_QUEUE_MAX_SIZE` 默认示例应保持为 `10`；即使显式配置得更大，运行时也必须按 `10` 截断，确保图片请求总量不超过 MT-Photos 客户端可接受范围。
- 如需限制 OCR 纯执行窗口，可额外设置 `OCR_EXEC_TIMEOUT`；否则默认至少保留 `30s`，避免模型切换/冷加载把执行超时提前耗尽。
- 如需调整空闲模型回收窗口，可额外设置 `NON_TEXT_IDLE_RELEASE_SECONDS`；设为 `0` 或负数可关闭该兜底释放。
- `PORT` 当前会同时影响 Docker 入口命令与容器健康检查；若修改它，`docker-compose` 里的 `ports:` 映射也必须同步修改。
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
- `POST /clip/img` 成功返回 `768` 维字符串数组（成功无 `msg`）。
- `POST /ocr` 成功返回 `{"result":{"texts","scores","boxes"}}`（成功无 `msg`）。
- `POST /represent` 返回 `{"detector_backend":"insightface","recognition_model":"antelopev2","result":[...]}` 或在人脸不存在时 `result=[]`。

失败判定（任一命中即失败）：
- 日志出现 `No silent fallback is allowed`（排除主动构造失败用例）。
- 日志显示 `preprocess_device` 非 `CPU`。
- 日志出现 `provider validation failed` 或 `OpenVINOExecutionProvider` 非首位。

#### D. 回归脚本建议（最小）

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=200 mt-photos-ai-openvino
docker compose logs --tail=200 mt-photos-ai-text-clip
docker run --rm -it -e INFERENCE_DEVICE=CPU -e CLIP_INFERENCE_DEVICE=CPU -e RAPIDOCR_DEVICE=CPU -e INSIGHTFACE_OV_DEVICE=CPU mt-photos-ai-openvino python scripts/smoke_insightface.py --device CPU

curl -s http://127.0.0.1:8060/
curl -s -X POST http://127.0.0.1:8060/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8061/check -H "api-key: mt_photos_ai_extra"
curl -s -X POST http://127.0.0.1:8061/clip/txt -H "api-key: mt_photos_ai_extra" -H "Content-Type: application/json" -d '{"text":"smoke"}'
```

---

## 14. `scripts/convert.py` 运行环境

| 环境变量 | 可选值 | 默认值 |
|---|---|---|
| `PROJECT_ROOT` | 项目根目录路径 | 自动探测 |
| `MODEL_PATH` | 模型根目录路径（导出会写入 `qa-clip/openvino`） | `<PROJECT_ROOT>/models` |
| `HF_CACHE_DIR` | Hugging Face 缓存目录路径 | `<PROJECT_ROOT>/cache/huggingface` |
| `QA_CLIP_ENABLE_NNCF_WEIGHT_COMPRESSION` | `0`（关闭）或 `1`（开启） | `0` |
| `QA_CLIP_NNCF_WEIGHT_MODE` | NNCF 压缩模式名（常见：`INT8_ASYM` / `INT8_SYM` / `NF4` / `E2M1`） | `INT8_ASYM` |

`scripts/convert.py` 在未预设时还会自动设置以下变量：`HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`、`HF_HUB_DISABLE_SYMLINKS_WARNING`。
