# MT-Photos AI (OpenVINO)

基于 FastAPI 的照片 AI 服务，统一提供 OCR、图文向量（QA-CLIP）和人脸向量（InsightFace）能力。

## 运行基线

- Python: **3.12**
- 平台: Windows Server
- 硬件: Intel i7-11800H（AVX512 VNNI + Xe iGPU）
- 向量维度: **768**

## 模型与后端约束

- QA-CLIP 固定为 `TencentARC/QA-CLIP-ViT-L-14`，权重来自 Hugging Face。
- QA-CLIP 代码依赖从 `app/QA-CLIP/clip` 引用，不再使用历史 `/app/clip`。
- QA-CLIP 转换为 OpenVINO IR 时使用 `compress_to_fp16`，并按 NNCF 约束保留关键层精度策略。
- RapidOCR 固定为 `rapidocr==3.6.0`，使用 OpenVINO CPU 后端，不做模型量化或结构改写。
- InsightFace 使用 ONNX Runtime + OpenVINO Execution Provider。

> 详细协作与改动规则见 [AGENTS.md](AGENTS.md)。

## 环境变量

| 环境变量 | 描述 | 默认值 |
|---|---|---|
| `API_AUTH_KEY` | API 鉴权密钥，设为 `no-key` 可关闭鉴权。 | `mt_photos_ai_extra` |
| `INFERENCE_DEVICE` | QA-CLIP OpenVINO 推理设备（如 `GPU`、`AUTO`）。 | `AUTO` |
| `MODEL_NAME` | InsightFace 模型名（`buffalo_l` / `antelopev2`）。 | `buffalo_l` |
| `WEB_CONCURRENCY` | Uvicorn worker 数。 | `1` |
| `SERVER_IDLE_TIMEOUT` | 空闲释放模型时间（秒）。 | `300` |
| `OV_CACHE_DIR` | OpenVINO 编译缓存目录。 | 由运行环境决定 |

## RapidOCR OpenVINO CPU 配置

示例配置见 [example/cfg_openvino_cpu.yaml](example/cfg_openvino_cpu.yaml)。

推荐至少关注这些参数：

- `inference_num_threads`
- `performance_hint`
- `performance_num_requests`
- `enable_cpu_pinning`
- `num_streams`
- `enable_hyper_threading`
- `scheduling_core_type`

## 部署注意事项

- RapidOCR v3 模型和字体资源建议在镜像构建阶段预下载到本地路径，避免部署后再下载。
- 启用 OpenVINO 编译缓存可明显降低冷启动和多 Worker 重启时的编译开销。
- Xe 核显为共享内存架构，优先使用 OpenVINO GPU 互操作能力以减少不必要拷贝。

## 许可证

MIT License
