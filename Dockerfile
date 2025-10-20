# ---- Stage 1: Model Conversion ----
# Use a development image that includes all necessary tools for conversion
FROM openvino/dev-py:latest AS builder

WORKDIR /builder

# Copy requirements and model conversion script first to leverage Docker layer caching
COPY openvino/requirements.txt .
COPY scripts/convert_models.py ./scripts/

# Install all dependencies needed for the conversion process
RUN pip install --no-cache-dir -r requirements.txt

# Run the conversion script.
# This will download the BAAI/AltCLIP-m18 model from Hugging Face,
# convert it to ONNX, and then to OpenVINO IR FP16 format.
# The output will be in /builder/models/alt-clip/openvino
RUN python scripts/convert_models.py --output_dir /builder/models/alt-clip

# ---- Stage 2: Final Runtime Image ----
# Use a lightweight runtime image for the final application
FROM openvino/runtime:2023.3

WORKDIR /app

# Copy requirements for the runtime environment
COPY openvino/requirements.txt .

# Install only the runtime dependencies
# We exclude torch, onnx, etc., to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the converted OpenVINO IR models from the builder stage
COPY --from=builder /builder/models/alt-clip/openvino /models/alt-clip/openvino

# Copy the pre-downloaded InsightFace models
# These should be placed in the `models` directory in your project root
COPY models/insightface/buffalo_l /models/insightface/buffalo_l

# Copy the application source code
COPY app/server_openvino.py .
COPY app/common/ /app/common/

# Expose the port the server will run on
EXPOSE 8060

# Set the default command to run the application
# Use an environment variable for the API key for security
ENV API_AUTH_KEY=""
# Use an environment variable to control the inference device (e.g., "CPU", "GPU", "AUTO")
ENV INFERENCE_DEVICE="AUTO"

CMD ["uvicorn", "server_openvino.py", "--host", "0.0.0.0", "--port", "8060"]
