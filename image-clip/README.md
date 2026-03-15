# Image-CLIP Windows CUDA 服务

这是一个独立于主服务的本地开发版 `Image-CLIP` 子项目，只提供 `/clip/img`，不影响仓库现有的 OpenVINO 主服务实现。

## 目录

- 服务入口：`image-clip/app/server.py`
- 依赖文件：`image-clip/requirement.txt`
- 模型：固定使用 `TencentARC/QA-CLIP-ViT-L-14`
- 输出维度：固定 `768`

## 运行前提

- Python `3.12`
- Windows 本地开发机
- 已安装可用的 CUDA 版 PyTorch，且 `torch.cuda.is_available()` 返回 `True`
- 可访问或已缓存 `TencentARC/QA-CLIP-ViT-L-14`

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r image-clip\requirement.txt
```

如果你已经单独安装了适配本机 CUDA 的 `torch`，继续安装其余依赖即可。

## 环境变量

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `API_AUTH_KEY` | API Key；`no-key` 或空字符串表示关闭鉴权 | `mt_photos_ai_extra` |
| `MODEL_PATH` | 模型根目录；若存在 `qa-clip/huggingface/config.json`，服务优先从该本地目录加载 | `<repo>/models` |
| `HF_CACHE_DIR` | Hugging Face 缓存目录 | `<repo>/cache/huggingface` |
| `HF_LOCAL_FILES_ONLY` | 是否只使用本地缓存加载 Hugging Face 模型 | `false` |
| `IMAGE_CLIP_DEVICE` | CUDA 设备，只接受 `cuda` / `cuda:0` 这类 CUDA 设备表达式 | `cuda` |
| `IMAGE_CLIP_USE_FP16` | 是否以 `fp16` 在 CUDA 上运行，显存紧张时可开启 | `false` |
| `CLIP_IMAGE_BATCH` | `/clip/img` 微批上限 | `8` |
| `CLIP_IMAGE_BATCH_WAIT_MS` | `/clip/img` 微批等待窗口（毫秒） | `5` |
| `INFERENCE_QUEUE_MAX_SIZE` | 图片请求总名额硬上限，运行时最多 `10` | `10` |
| `INFERENCE_TASK_TIMEOUT` | 兼容旧配置的超时基线 | `10` |
| `INFERENCE_QUEUE_TIMEOUT` | 排队超时（秒） | 跟随 `INFERENCE_TASK_TIMEOUT` |
| `INFERENCE_EXEC_TIMEOUT` | 执行超时（秒），默认至少 `30` 秒 | `max(30, INFERENCE_TASK_TIMEOUT)` |
| `PORT` | 服务端口 | `8062` |
| `LOG_LEVEL` | 日志级别 | `WARNING` |

## 启动

```powershell
$env:IMAGE_CLIP_DEVICE="cuda"
$env:LOG_LEVEL="INFO"
cd image-clip\app
python server.py
```

## 冒烟

```powershell
python scripts\smoke_image_clip.py --device cuda
python scripts\smoke_image_clip.py --device cuda --image C:\path\to\image.jpg

curl.exe http://127.0.0.1:8062/
curl.exe -X POST http://127.0.0.1:8062/check -H "api-key: mt_photos_ai_extra"
curl.exe -X POST http://127.0.0.1:8062/clip/img -H "api-key: mt_photos_ai_extra" -F "file=@C:\path\to\image.jpg"
```

脚本会直接调用 `image-clip/app` 运行时，不经过 HTTP，主要校验：

- CUDA 初始化成功，且不是静默落回 CPU
- 单张、同图微批、混合微批输出维度固定为 `768`
- 同图顺序推理、批推理、释放后重载三条路径的 embedding 一致性

成功调用 `/clip/img` 时，响应格式与主服务保持一致：

```json
{"result":["0.1234567890123456","..."]}
```
