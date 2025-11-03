# MT-Photos AI

使用 QA-CLIP + Insightface + RapidOCR + OpenVINO 的一站式 AI 服务。
对非 OpenVINO 设备做了自动 fallback，欢迎 PR。

| 环境变量               | 描述                                                                                       | 默认值                   |
|--------------------|------------------------------------------------------------------------------------------|-----------------------|
| `API_AUTH_KEY`     | 用于保护 API 端点的密钥。                                                                          | `"mt_photos_ai_extra"` |
| `INFERENCE_DEVICE` | 指定 OpenVINO 的推理设备，可选值如 `"CPU"`, `"GPU"`, `"AUTO"`。`AUTO` 会自动选择最佳设备。                      | `"AUTO"`              |
| `MODEL_NAME`       | Insightface 使用的模型名称，填"buffalo_l"或"antelopv2"，镜像已经自带这两个模型，无需下载。请注意antelopv2未必比buffalo_l好。 | `"buffalo_l"`         |
| `WEB_CONCURRENCY`  | 控制 worker 数量。注意：每个 worker 都会加载自己的模型实例，会增加内存使用。 | `"1"`                   |

>**向量维度需要改成 768**

> Docker Hub: https://hub.docker.com/r/molyleaf/mt-photos-ai-openvino

> GitHub: https://github.com/molyleaf/mt-photos-ai-openvino


---
## 以下是AI写的

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.3-purple.svg)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

一个基于 **Intel OpenVINO™** 加速的、用于照片分析的高性能统一 AI 服务。该服务通过 FastAPI 框架提供了一系列 RESTful API，用于人脸识别、图像/文本特征提取 (CLIP) 和光学字符识别 (OCR)，为 `MT-Photos` 或其他相册管理应用提供强大的 AI 后端支持。

## ✨ 主要功能

* **🚀 高性能推理**：所有模型均通过 OpenVINO™ 进行优化和推理，在 Intel CPU 和 GPU 上实现低延迟、高吞吐量的 AI 计算。
* **👨‍👩‍👧 人脸识别**：集成 `InsightFace` 模型，提供高精度的人脸检测和特征向量提取功能，用于人物聚类。
* **🖼️ 图像与文本特征提取**：集成 `BAAI/AltCLIP` 多模态模型，可将任意图像和文本转换为统一的特征向量，用于语义搜索和图文匹配。
* **📄 文字识别 (OCR)**：集成 `RapidOCR`，能够快速、准确地识别图像中的文字信息。
* **📦 容器化部署**：提供 `Dockerfile`，支持一键构建和部署，极大简化了生产环境的配置。
* **🔒 安全认证**：所有 API 端点均通过 API 密钥进行保护。

## 🛠️ 技术栈

* **推理引擎**: Intel OpenVINO™
* **服务框架**: FastAPI
* **AI 模型**:
    * **人脸识别**: InsightFace (buffalo_l)
    * **多模态**: Chinese-CLIP / QA-CLIP
    * **OCR**: RapidOCR
* **容器化**: Docker

## 🙏 致谢

本项目依赖于以下优秀的开源项目和模型：

* [Intel OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino)
* [InsightFace](https://github.com/deepinsight/insightface)
* [RapidOCR](https://github.com/RapidAI/RapidOCR)
* [QA-CLIP](https://github.com/TencentARC-QQ/QA-CLIP)
* [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

## 📄 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。
