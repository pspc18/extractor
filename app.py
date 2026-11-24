"""
AI Dynamic Extractor - FastAPI Server.
Services:
1. High-Volume OMR Sheet Scanner & Evaluator (returns JSON + annotated image URL & base64)
2. Intelligent Aadhaar Card Extractor (Front & Back OCR: Name, DOB, Gender, Aadhaar, Care of, Address, Pincode)
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import BASE_DIR, DEBUG_DIR
from dynamic_engine import decode_image_bytes, decode_base64_image, process_dynamic_omr
from aadhaar_engine import (
    extract_aadhaar_from_image,
    extract_aadhaar_dual_side,
    decode_base64_image as decode_aadhaar_b64
)
from aadhaar_qr import (
    decode_aadhaar_qr,
    detect_and_decode_qr_from_image
)

app = FastAPI(
    title="AI Dynamic Extractor (OMR & Aadhaar)",
    description="Zero-cost, high-precision AI extraction engine for OMR Sheets & Aadhaar Cards.",
    version="4.0.0"
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


# Pydantic Schemas for JSON base64 requests
class OMRExtractBase64Request(BaseModel):
    image: str  # Base64 string or data:image/jpeg;base64,...
    filename: Optional[str] = "sheet.jpg"


class AadhaarExtractBase64Request(BaseModel):
    front: Optional[str] = None  # Base64 string of front side
    back: Optional[str] = None   # Base64 string of back side
    image: Optional[str] = None  # Base64 string of single/combined card
    filename: Optional[str] = "aadhaar.jpg"


class AadhaarDecodeQRRequest(BaseModel):
    qr_data: str  # Raw QR scan string (UIDAI Secure QR big-int, XML string, or JSON)


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the Unified Web Dashboard for OMR & Aadhaar Extraction."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-dynamic-extractor",
        "capabilities": ["omr-scanner", "aadhaar-extractor"]
    }


# =============================================================================
# 1. OMR EXTRACTION ENDPOINTS
# =============================================================================

@app.post("/api/extract")
@app.post("/api/omr/extract")
@app.post("/extract")
async def extract_omr_file(file: UploadFile = File(...)):
    """
    OMR Multipart API: Upload an OMR sheet image (JPG, PNG, WebP).
    Returns complete JSON with roll number, answers, summary, annotated_image_url,
    AND annotated_image_base64 (data:image/jpeg;base64,...) for direct API consumption.
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
        raise HTTPException(status_code=500, detail=f"OMR extraction failed: {str(e)}")


@app.post("/api/omr/extract-base64")
async def extract_omr_base64(req: OMRExtractBase64Request):
    """
    OMR JSON API: Send base64-encoded image string in JSON body.
    Returns complete JSON with answers and annotated image.
    """
    try:
        cv_img = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    filename = req.filename or "sheet.jpg"
    try:
        result = process_dynamic_omr(cv_img, filename=filename)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OMR extraction failed: {str(e)}")


# =============================================================================
# 2. AADHAAR CARD EXTRACTION ENDPOINTS
# =============================================================================

@app.post("/api/aadhaar/extract")
async def extract_aadhaar_multipart(
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Aadhaar Multipart API:
    - Option A: Upload both 'front' and 'back' images together.
    - Option B: Upload single image in 'file' (or 'front') for Front, Back, or Combined card.
    Extracts: Name, DOB, YOB, Gender, Aadhaar Number, VID, Care of (Father/Husband),
             Address, District, State, and PIN Code.
    """
    front_img = None
    back_img = None
    front_filename = "front.jpg"
    back_filename = "back.jpg"

    # Single file fallback
    if file is not None and front is None and back is None:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        cv_img = decode_image_bytes(content)
        result = extract_aadhaar_from_image(cv_img, filename=file.filename or "aadhaar.jpg")
        return JSONResponse(content=result)

    # Front file
    if front is not None:
        content_f = await front.read()
        if content_f:
            front_img = decode_image_bytes(content_f)
            front_filename = front.filename or "front.jpg"

    # Back file
    if back is not None:
        content_b = await back.read()
        if content_b:
            back_img = decode_image_bytes(content_b)
            back_filename = back.filename or "back.jpg"

    if front_img is None and back_img is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one image ('front', 'back', or 'file')."
        )

    # If both provided, run dual-side merger
    if front_img is not None and back_img is not None:
        result = extract_aadhaar_dual_side(
            front_image=front_img,
            back_image=back_img,
            front_filename=front_filename,
            back_filename=back_filename
        )
        return JSONResponse(content=result)
    elif front_img is not None:
        result = extract_aadhaar_from_image(front_img, filename=front_filename)
        return JSONResponse(content=result)
    else:
        result = extract_aadhaar_from_image(back_img, filename=back_filename)
        return JSONResponse(content=result)


@app.post("/api/aadhaar/extract-base64")
async def extract_aadhaar_base64_api(req: AadhaarExtractBase64Request):
    """
    Aadhaar JSON API: Send base64 strings in JSON body for 'front', 'back', or single 'image'.
    """
    front_img = None
    back_img = None

    if req.image:
        cv_img = decode_aadhaar_b64(req.image)
        result = extract_aadhaar_from_image(cv_img, filename=req.filename or "aadhaar.jpg")
        return JSONResponse(content=result)

    if req.front:
        front_img = decode_aadhaar_b64(req.front)
    if req.back:
        back_img = decode_aadhaar_b64(req.back)

    if front_img is None and back_img is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 'front', 'back', or 'image' base64."
        )

    if front_img is not None and back_img is not None:
        result = extract_aadhaar_dual_side(
            front_image=front_img,
            back_image=back_img,
            front_filename="front.jpg",
            back_filename="back.jpg"
        )
    elif front_img is not None:
        result = extract_aadhaar_from_image(front_img, filename="front.jpg")
    else:
        result = extract_aadhaar_from_image(back_img, filename="back.jpg")

    return JSONResponse(content=result)


@app.post("/api/aadhaar/decode-qr")
async def decode_aadhaar_qr_api(req: AadhaarDecodeQRRequest):
    """
    Decodes raw Aadhaar QR code string (UIDAI Secure QR big-int, XML, or JSON).
    Instant execution (<10ms), zero OCR latency, 100% digitally verified data.
    """
    if not req.qr_data or not req.qr_data.strip():
        raise HTTPException(status_code=400, detail="Missing 'qr_data' in request body.")

    decoded = decode_aadhaar_qr(req.qr_data)
    if not decoded:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Could not parse valid Aadhaar data from the provided QR code string."
            }
        )

    return JSONResponse(content={
        "success": True,
        "extraction_method": "QR_CODE",
        "qr_type": decoded.get("qr_type"),
        "confidence": 1.0,
        "data": {
            "name": decoded.get("name"),
            "name_hindi": None,
            "dob": decoded.get("dob"),
            "gender": decoded.get("gender"),
            "aadhaar_number": decoded.get("aadhaar_number"),
            "care_of": decoded.get("care_of"),
            "address": decoded.get("address"),
            "pincode": decoded.get("pincode"),
            "photo_base64": decoded.get("photo_base64")
        }
    })


@app.post("/api/aadhaar/extract-qr")
async def extract_aadhaar_qr_file(file: UploadFile = File(...)):
    """
    Scans an uploaded image specifically for an Aadhaar QR code.
    Returns 100% verified demographics decoded from the QR symbol.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    cv_img = decode_image_bytes(content)
    decoded = detect_and_decode_qr_from_image(cv_img)
    if not decoded:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No valid Aadhaar QR code found in the uploaded image. Please ensure the QR code is clear and not obstructed."
            }
        )

    return JSONResponse(content={
        "success": True,
        "filename": file.filename or "aadhaar_qr.jpg",
        "extraction_method": "QR_CODE",
        "qr_type": decoded.get("qr_type"),
        "confidence": 1.0,
        "data": {
            "name": decoded.get("name"),
            "name_hindi": None,
            "dob": decoded.get("dob"),
            "gender": decoded.get("gender"),
            "aadhaar_number": decoded.get("aadhaar_number"),
            "care_of": decoded.get("care_of"),
            "address": decoded.get("address"),
            "pincode": decoded.get("pincode"),
            "photo_base64": decoded.get("photo_base64")
        }
    })

