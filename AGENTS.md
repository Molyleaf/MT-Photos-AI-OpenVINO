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
- Windows Server GPU-PV 场景（容器映射 `/dev/dxg`）可用于 OpenVINO GPU 推理/Remote Context；但 `ffmpeg QSV` 硬解链路仍以 `/dev/dri`（VAAPI/oneVPL）为主，需在文档中明确能力边界。
- `/clip/img` 视觉链路必须直接消费 `numpy BGR`，禁止 `BGR -> RGB -> PIL` 的多余拷贝链。
- CLIP 视觉预处理（`resize + 通道转换 + 归一化 + layout`）必须走 OpenVINO PrePostProcessing (PPP) API，禁止回退到手工 `numpy` 链。

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

- 依赖固定：`rapidocr==3.6.0`。
- 禁止使用：`rapidocr-openvino`。
- 禁止对 RapidOCR 模型做量化或结构改写。
- 必须使用 RapidOCR 内置后端选择能力，指定 **OpenVINO CPU** 后端（不使用 GPU）。
- 默认使用 **PP-OCRv5 mobile** 模型配置（`Det/Rec`）。
- 默认关闭方向分类器（`Global.use_cls=false`）；如需开启，必须明确下载并配置分类模型。
- 即使 `Global.use_cls=false`，也应预置 `ch_ppocr_mobile_v2.0_cls_infer.onnx`，避免 RapidOCR 初始化阶段触发在线下载。
- OpenVINO CPU 参数基线（面向 i7-11800H 低延迟）：`performance_hint=LATENCY`、`inference_num_threads=-1`、`num_streams=1`、`enable_cpu_pinning=true`。
- 必须启用模型编译缓存，降低冷启动与多 Worker 反复编译开销。
- RapidOCR v3 模型与字体资源需在镜像构建前预下载到本地路径（避免部署后在线下载）。
- RapidOCR 必须执行“本地模型强校验 + 缺失即失败”，移除线上下载回退逻辑。
- OCR 输入预处理必须基于 OpenCV BGR `numpy`（零拷贝优先）：连续 `uint8` 缓冲区直接透传，禁止引入 `PIL` 中转链。

### 3.4 InsightFace

- 固定为 **ONNX Runtime + OpenVINO Execution Provider**。
- 识别模型固定为 `antelopev2`；禁止切回 `buffalo_l`。
- 保留 CPU 回退路径，但默认 EP 优先顺序应以 OpenVINO EP 在前。
- 对齐阶段必须使用 OpenCV，并优先启用 Intel OpenCL 驱动（`warpAffine` OpenCL 路径）；不可用时才允许回退 CPU。
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
  - 启动时初始化 `AIModels` 并预热加载 Text-CLIP
  - 关闭时释放全部模型
- 空闲计时：
  - 默认 `SERVER_IDLE_TIMEOUT=300`
  - 中间件会在非 `"/check"`, `"/"`, `"/clip/txt"` 请求上重置空闲计时器
  - 空闲超时触发按需模型释放
- 模型实例为空时：相关推理端点返回 HTTP 503（`"模型实例尚未初始化"`）。

### 4.2 图像读取辅助逻辑（`read_image_from_upload`）

- GIF：读取第一帧并转 BGR。
- 非 GIF：优先 `ffmpeg(QSV)` 解码；若失败需显式记录原因，再按顺序尝试 `ffmpeg(CPU)` 与 `cv2.imdecode(..., IMREAD_UNCHANGED)`；若 `ffprobe` 失败或未返回尺寸，也应继续尝试 `ffmpeg(QSV/CPU)` 后再回退 `cv2.imdecode`；若检测到高位深图像，应直接走 `cv2.imdecode` 以保持 16-bit 处理语义。
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

1. 待机状态下，Text-CLIP 常驻内存（优化文本请求时延）。
2. 在收到 `POST /restart` 之后立即释放当前模型；若在统一恢复窗口 `TEXT_MODEL_RESTORE_DELAY_MS`（默认 2000ms）内没收到其它类型请求，才加载 Text-CLIP。防止收到 `POST /restart` 之后立刻加载 Text-CLIP，又收到其它请求导致混乱。
3. 处理 OCR/图像 CLIP/人脸任务时，卸载 Text-CLIP，加载目标模型。
4. 同一时刻内存/显存中仅允许一个主模型族常驻（Text-CLIP / Vision-CLIP / OCR / Insightface 互斥）。
5. 请求并发控制必须采用队列化（或等价可证明正确的串行化调度）。
6. `/clip/txt` 具备低优先级：
  - 若当前正在处理其他任务，不抢断当前正在执行的推理；
  - 当前任务结束后，优先处理排队中的文本 CLIP 请求，再继续后续队列；
  - 必须正确处理模型切换与资源回收。
7. 非文本任务结束且队列瞬时为空时，不应立即回切 Text-CLIP；必须等待 `TEXT_MODEL_RESTORE_DELAY_MS` 窗口后再恢复，避免连续单次 `/clip/img` 触发模型频繁重载。

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
- 若引入 RapidOCR OpenVINO CPU 参数文件，需提供示例 `cfg_openvino_cpu.yaml` 并说明关键参数。

---

## 10. 开发/自检命令（至少执行到可验证）

- `python -V`（确认 3.12）
- `pip install -r requirements.txt`
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
- [ ] RapidOCR 是否为 `rapidocr==3.6.0` + OpenVINO CPU + PP-OCRv5 mobile（Det/Rec）
- [ ] InsightFace 是否使用 ORT + OpenVINO EP
- [ ] 是否遵守“待机常驻 Text-CLIP + 单模型族互斥加载 + 文本请求优先”策略
- [ ] 是否采用多进程 Worker + 有界队列 + Worker 内批处理，并规避线程过度订阅
- [ ] 是否同步更新 `README.md` 与 `requirements.txt`
