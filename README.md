# MT-Photos AI (AltCLIP + OpenVINO 版本)

使用 Chinese-CLIP / QA-CLIP + Insightface + RapidOCR + OpenVINO 的一站式 AI 服务。
ONNX 版本稍后做，欢迎 PR。

| 环境变量               | 描述                                                                                       | 默认值                   |
|--------------------|------------------------------------------------------------------------------------------|-----------------------|
| `API_AUTH_KEY`     | 用于保护 API 端点的密钥。                                                                          | `"mt_photos_ai_extra"` |
| `INFERENCE_DEVICE` | 指定 OpenVINO 的推理设备，可选值如 `"CPU"`, `"GPU"`, `"AUTO"`。`AUTO` 会自动选择最佳设备。                      | `"AUTO"`              |
| `MODEL_NAME`       | Insightface 使用的模型名称，填"buffalo_l"或"antelopv2"，镜像已经自带这两个模型，无需下载。请注意antelopv2未必比buffalo_l好。 | `"buffalo_l"`         |
| `WEB_CONCURRENCY`  | 控制 worker 数量。注意：每个 worker 都会加载自己的模型实例，会增加内存使用。 | `"1"`                   |

请使用最新的Docker镜像，旧版可能有bug。

- Docker Hub: https://hub.docker.com/r/molyleaf/mt-photos-ai-openvino

- GitHub: https://github.com/molyleaf/mt-photos-ai-openvino

提供 Chinese-CLIP 和 QA-CLIP 镜像：

``` docker pull molyleaf/mt-photos-ai-openvino:1.0.1.6-QA-CLIP ```

``` docker pull molyleaf/mt-photos-ai-openvino:1.2.2-Chinese-CLIP ```

**向量维度需要改成 768**

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

## 🚀 快速开始

### 1. 环境准备

* Python 3.13

### 2. 克隆项目

```bash
git clone <your-repository-url>
cd mt-photos-ai-openvino
```

### 3. 安装依赖

建议在 Python 虚拟环境中安装。

```bash
pip install -r requirements.txt
```

### 4. 准备模型文件

在项目根目录下创建一个 `models` 文件夹。所有模型文件都将存放于此。

```bash
mkdir models
```

**a. 转换 Alt-CLIP 模型**

运行模型转换脚本，它会自动从 Hugging Face 下载模型并将其转换为 OpenVINO IR 格式。

```bash
python scripts/convert_models.py
```
转换成功后，模型文件将位于 `./models/alt-clip/openvino/` 目录下。

### 5. 启动服务

```bash
# 设置用于保护 API 的密钥
export API_AUTH_KEY="YOUR_SECRET_KEY"

# 启动 Uvicorn 服务
uvicorn app.server_openvino:app --host 0.0.0.0 --port 8060
```

服务启动成功后，您可以在 `http://127.0.0.1:8060/docs` 查看交互式 API 文档。

## 🐳 Docker 部署

使用 Docker 是推荐的部署方式，可以确保环境一致性。

### 1. 准备模型文件

在构建 Docker 镜像之前，您需要**在本地**先执行 `步骤 4`，确保 `./models` 目录中已经包含了转换好的 Alt-CLIP 模型和下载好的 InsightFace 模型。

### 2. 构建镜像

在项目根目录（`Dockerfile` 所在的目录）下运行以下命令：

```bash
docker build -t mt-photos-ai-openvino:1.0 .
```

### 3. 运行容器

```bash
docker run -d --name mt-photos-ai \
  -p 8060:8060 \
  -e API_AUTH_KEY="YOUR_SECRET_KEY" \
  -e INFERENCE_DEVICE="AUTO" \
  mt-photos-ai-openvino:1.0
```

## ⚙️ 配置

服务通过环境变量进行配置：

| 环境变量           | 描述                                                                                                      | 默认值   |
| ------------------ | --------------------------------------------------------------------------------------------------------- | -------- |
| `API_AUTH_KEY`     | 用于保护 API 端点的密钥。如果未设置，则服务不设防。                                                       | `None`   |
| `INFERENCE_DEVICE` | 指定 OpenVINO 的推理设备，可选值如 `"CPU"`, `"GPU"`, `"AUTO"`。`AUTO` 会自动选择最佳设备。                 | `"AUTO"` |
| `MODEL_PATH`       | 容器内模型存放的基础路径。**通常无需修改**。                                                              | `/models` |

## 📡 API 使用示例

所有请求都需要在请求头中包含 API 密钥：`api-key: YOUR_SECRET_KEY`。

### 人脸识别

**POST** `/represent`

```bash
curl -X POST "[http://127.0.0.1:8060/represent](http://127.0.0.1:8060/represent)" \
  -H "api-key: YOUR_SECRET_KEY" \
  -F "file=@/path/to/your/image.jpg"
```

### 图像特征提取

**POST** `/clip/img`

```bash
curl -X POST "[http://127.0.0.1:8060/clip/img](http://127.0.0.1:8060/clip/img)" \
  -H "api-key: YOUR_SECRET_KEY" \
  -F "file=@/path/to/your/image.jpg"
```

### 文本特征提取

**POST** `/clip/txt`

```bash
curl -X POST "[http://127.0.0.1:8060/clip/txt](http://127.0.0.1:8060/clip/txt)" \
  -H "api-key: YOUR_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "一只猫在沙发上睡觉"}'
```

### OCR 文字识别

**POST** `/ocr`

```bash
curl -X POST "[http://127.0.0.1:8060/ocr](http://127.0.0.1:8060/ocr)" \
  -H "api-key: YOUR_SECRET_KEY" \
  -F "file=@/path/to/your/image.jpg"
```

## 🙏 致谢

本项目依赖于以下优秀的开源项目和模型：

* [Intel OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino)
* [InsightFace](https://github.com/deepinsight/insightface)
* [RapidOCR](https://github.com/RapidAI/RapidOCR)
* [QA-CLIP](https://github.com/TencentARC-QQ/QA-CLIP)
* [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

## 📄 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。