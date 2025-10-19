# ---- Stage 1: Model Conversion Builder ----
# 使用包含完整开发工具的镜像作为构建环境
FROM openvino/dev-py:latest AS builder

WORKDIR /builder

# 安装模型转换所需的依赖
COPY scripts/requirements_convert.txt .
RUN pip install --no-cache-dir -r requirements_convert.txt

# 复制并执行模型转换脚本
COPY scripts/convert_models.py .
# 脚本将自动下载并转换模型到 /models/alt-clip/openvino
RUN python convert_models.py --output_dir /models/alt-clip/openvino

# ---- Stage 2: Final Runtime Image ----
# 使用轻量级的OpenVINO运行时镜像作为最终基础
FROM openvino/runtime:latest

WORKDIR /app

# 安装运行时依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 从构建阶段复制转换好的Alt-CLIP OpenVINO IR模型
COPY --from=builder /models/alt-clip/openvino /app/models/alt-clip/openvino

# 复制应用程序源代码
# 注意：这里假设你的项目根目录是构建上下文(.)
COPY app/ /app/app/

# 暴露服务端口
EXPOSE 8060

# 设置默认启动命令
# 注意：模型将从容器内的 /models 目录加载，这需要通过-v挂载
CMD ["uvicorn", "app.server_openvino:app", "--host", "0.0.0.0", "--port", "8060"]