"""
AI Dynamic OMR Extractor - FastAPI Server.
Single-purpose, high-intelligence extraction service:
Upload Image -> Click Extract -> Instant JSON Output.
"""

from pathlib import Path
from typing import Any, Dict
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, DEBUG_DIR
from dynamic_engine import decode_image_bytes, process_dynamic_omr

app = FastAPI(
    title="AI Dynamic OMR Extractor",
    description="Intelligent zero-config OMR extractor that automatically detects and extracts any sheet.",
    version="3.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/debug_images", StaticFiles(directory=str(DEBUG_DIR)), name="debug_images")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the clean, distraction-free single-screen UI."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-dynamic-omr-extractor"}


@app.post("/api/extract")
@app.post("/extract")
async def extract_omr(file: UploadFile = File(...)):
    """
    Intelligently extracts Roll Number, Question Answers, and Status from any OMR sheet.
    Accepts JPG, PNG, WEBP.
    Returns clean JSON with recognized answers and annotated image URL.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        cv_img = decode_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = file.filename or "sheet.jpg"

    try:
        result = process_dynamic_omr(cv_img, filename=filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
