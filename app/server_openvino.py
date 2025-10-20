# server.py
# FastAPI server for MT-Photos AI Unified

import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import io
from typing import Optional, List, Dict, Any

# Import the model management class
from common.models import models, AIModels

# --- API Key Authentication ---
API_AUTH_KEY = os.environ.get("API_AUTH_KEY")
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    """Dependency to validate the API key."""
    if not API_AUTH_KEY: # If no key is set, allow access
        return
    if api_key_header != API_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# --- FastAPI App Initialization ---
app = FastAPI(
    title="MT-Photos AI Unified Server (OpenVINO)",
    description="A high-performance, unified AI service for photo analysis, accelerated by OpenVINO.",
    version="1.0.0",
    dependencies=[Depends(get_api_key)]
)

# --- Pydantic Models for Request/Response ---
class TextClipRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    result: str

class OCRResult(BaseModel):
    texts: List[str]
    scores: List[float]
    boxes: List[List[List[int]]]

class OCRResponse(BaseModel):
    result: OCRResult

class ClipResponse(BaseModel):
    results: List[float]

class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int

class RepresentResult(BaseModel):
    embedding: List[float]
    facial_area: FacialArea
    face_confidence: float

class RepresentResponse(BaseModel):
    detector_backend: str = "insightface"
    recognition_model: str = "buffalo_l"
    result: List[RepresentResult]

# --- Helper Function ---
async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Reads an uploaded image file and converts it to a CV2 image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Check if models were loaded on startup."""
    if models is None:
        # This will prevent the app from starting if models fail to load
        raise RuntimeError("FATAL: AI models could not be initialized. Check logs for errors.")

@app.post("/check", response_model=CheckResponse)
async def check_service():
    """Checks service availability and API key validity."""
    return {"result": "pass"}

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """Performs OCR on an uploaded image."""
    image = await read_image_from_upload(file)
    ocr_results = models.get_ocr_results(image)
    return {"result": ocr_results}

@app.post("/clip/img", response_model=ClipResponse)
async def clip_image_endpoint(file: UploadFile = File(...)):
    """Extracts a feature vector from an uploaded image."""
    image = await read_image_from_upload(file)
    # Convert from BGR (cv2 default) to RGB for CLIP model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embedding = models.get_image_embedding(image_rgb)
    return {"results": embedding}

@app.post("/clip/txt", response_model=ClipResponse)
async def clip_text_endpoint(request: TextClipRequest):
    """Extracts a feature vector from a text string."""
    embedding = models.get_text_embedding(request.text)
    return {"results": embedding}

@app.post("/represent", response_model=RepresentResponse)
async def represent_endpoint(file: UploadFile = File(...)):
    """Detects faces and extracts feature vectors from an uploaded image."""
    image = await read_image_from_upload(file)
    face_results = models.get_face_representation(image)
    return {"result": face_results}

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
