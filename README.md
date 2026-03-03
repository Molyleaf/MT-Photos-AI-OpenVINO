# MT-Photos AI (OpenVINO)

统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）的 FastAPI 服务。

## 运行基线

- Python: **3.12**
- 目标平台: Windows Server
- 硬件基线: Intel i7-11800H + Xe iGPU
- QA-CLIP 向量维度: **768**

## 模型与后端

- QA-CLIP 固定为 `TencentARC/QA-CLIP-ViT-L-14`，模型来源为 Hugging Face。
- QA-CLIP 代码依赖从 `app/QA-CLIP/clip` 引用，不使用历史 `/app/clip` 路径。
- QA-CLIP 推理默认设备 `GPU`（可通过 `INFERENCE_DEVICE` 覆盖），OpenVINO 优先启用 GPU remote context。
- 推理运行时不依赖 `torch/torchvision/transformers/nncf`，仅在执行 `app/convert.py` 转换时按需安装。
- RapidOCR 固定 `rapidocr==3.6.0`，使用 OpenVINO **CPU** 后端。
- InsightFace 固定 ONNX Runtime，provider 顺序为 `OpenVINOExecutionProvider -> CPUExecutionProvider`。

## 并发与调度

- 架构: **多进程 Worker（Uvicorn） + 每 Worker 有界队列 + Worker 内文本批处理**。
- 每 Worker 内只运行一个模型调度线程，推理请求统一入队串行执行。
- `/clip/txt` 为非抢断优先调度：不抢断正在执行的任务，但会在当前任务结束后优先处理文本队列。
- 同一时刻仅一个主模型族常驻（Text-CLIP / Vision-CLIP / OCR / Face 互斥）。
- 非文本任务结束且队列瞬时为空时，不会立即回切 Text-CLIP；会等待 `TEXT_MODEL_RESTORE_DELAY_MS` 指定窗口，减少连续 `/clip/img` 请求的模型抖动。
- `/restart` 会立即释放当前模型；若 `TEXT_MODEL_RESTORE_DELAY_MS` 窗口内没有收到 OCR/图像 CLIP/人脸请求，再恢复 Text-CLIP 常驻。

## 环境变量

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | API 鉴权密钥，设为 `no-key` 跳过鉴权 | `mt_photos_ai_extra` |
| `INFERENCE_DEVICE` | QA-CLIP 推理设备 | `GPU` |
| `CLIP_INFERENCE_DEVICE` | 仅覆盖 QA-CLIP 推理设备 | 跟随 `INFERENCE_DEVICE` |
| `MODEL_PATH` | 模型根目录 | `<repo>/models` |
| `MODEL_NAME` | InsightFace 模型名 | `buffalo_l` |
| `WEB_CONCURRENCY` | Uvicorn 进程数 | `2` |
| `INFERENCE_QUEUE_MAX_SIZE` | 每 Worker 队列上限 | `64` |
| `TEXT_CLIP_BATCH_SIZE` | 每 Worker 文本批处理上限 | `8` |
| `INFERENCE_TASK_TIMEOUT` | 队列任务超时（秒） | `120` |
| `SERVER_IDLE_TIMEOUT` | 空闲释放触发时间（秒） | `300` |
| `TEXT_MODEL_RESTORE_DELAY_MS` | Text-CLIP 延迟恢复窗口（毫秒），用于 `/restart` 与“队列瞬时空闲后回切 Text” | `5000` |
| `OV_CACHE_DIR` | OpenVINO 编译缓存目录 | `<repo>/cache/openvino` |
| `RAPIDOCR_OPENVINO_CONFIG_PATH` | RapidOCR OpenVINO 配置文件 | `app/config/cfg_openvino_cpu.yaml` |
| `RAPIDOCR_MODEL_DIR` | RapidOCR 本地模型目录（建议镜像构建前预下载） | `<repo>/models/rapidocr` |
| `RAPIDOCR_FONT_PATH` | RapidOCR 字体路径（可选） | 空 |
| `INSIGHTFACE_OV_DEVICE` | InsightFace OpenVINO EP 设备类型 | `CPU_FP32` |

> 兼容性：若未设置 `TEXT_MODEL_RESTORE_DELAY_MS`，服务会回退读取旧变量 `RESTART_TEXT_RESTORE_DELAY_MS`。
> 总并行度建议按 `WEB_CONCURRENCY × inference_num_threads` 评估，避免线程过订阅。

## RapidOCR OpenVINO CPU 配置

示例文件: [app/config/cfg_openvino_cpu.yaml](app/config/cfg_openvino_cpu.yaml)

关键参数（均已在服务端支持）：

- `inference_num_threads`
- `performance_hint`
- `performance_num_requests`
- `enable_cpu_pinning`
- `num_streams`
- `enable_hyper_threading`
- `scheduling_core_type`

## QA-CLIP 转换

入口脚本: `app/convert.py`

特性：

- 从 Hugging Face 拉取 QA-CLIP 后转换 OpenVINO IR
- 不依赖仓库顶层 `scripts/` 目录
- 视觉/文本分支顺序转换，阶段结束立即 `gc.collect()`
- 导出使用 `compress_to_fp16=True`
- 预置 NNCF 关键层忽略策略（LayerNorm/投影等）；仅在显式开启压缩时才需要安装 `nncf`

转换前按需安装依赖（不写入 `requirements.txt`）：

```bash
pip install openvino torch transformers
```

如需启用 NNCF 权重压缩，再额外安装：

```bash
pip install nncf
```

执行：

```bash
python app/convert.py
```

## 本地运行与自检

```bash
python -V
pip install -r requirements.txt
python -m compileall app
cd app
uvicorn server:app --host 0.0.0.0 --port 8060
```

建议冒烟端点：`/check`、`/clip/txt`、`/ocr`、`/represent`。

## 许可证

MIT License
