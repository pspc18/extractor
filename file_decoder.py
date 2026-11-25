"""
Universal File and Image Decoder for Antigravity Extractor.
Supports:
1. Standard Images: JPG, JPEG, PNG, WebP, BMP, TIFF (via OpenCV)
2. Scanned Documents: PDF (rendered in-memory at 300 DPI via PyMuPDF/fitz with 0% quality loss)
   - STRICT ENFORCEMENT: Only single-page PDF is processed for OMR sheet scanning.
   - Multi-page PDFs raise a clear, actionable validation error.
3. iPhone Photos: HEIC / HEIF format (lossless in-memory conversion to BGR numpy array via pillow-heif)
"""

import base64
import io
import re
from typing import Optional
import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pillow_heif
    pillow_heif_available = True
except Exception:
    pillow_heif_available = False


def is_pdf(data: bytes) -> bool:
    """Checks if raw byte stream is a PDF document."""
    return data.startswith(b'%PDF')


def is_heif(data: bytes) -> bool:
    """Checks if raw byte stream has ISO BMFF ftyp box with HEIF/HEIC brand."""
    if len(data) < 12:
        return False
    if data[4:8] == b'ftyp':
        brand = data[8:12].lower()
        if brand in (b'heic', b'heix', b'hevc', b'hevx', b'mif1', b'msf1'):
            return True
    return False


def decode_pdf_bytes(pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
    """
    Renders single-page PDF into high-resolution, uncompressed OpenCV BGR image.
    Enforces single-page constraint: raises ValueError if multiple pages detected.
    """
    if fitz is None:
        raise ValueError("PDF processing engine (PyMuPDF) is not installed on this server.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")

    total_pages = len(doc)
    if total_pages == 0:
        raise ValueError("Uploaded PDF document is empty (0 pages).")

    if total_pages > 1:
        raise ValueError(
            f"Multi-page PDF detected ({total_pages} pages). Only single-page PDF is supported for OMR sheet scanning. "
            f"Please upload a single-page PDF."
        )

    # Render single page at 300 DPI for maximum OMR bubble accuracy
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))

    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def decode_heif_bytes(heif_bytes: bytes) -> np.ndarray:
    """
    Decodes iPhone HEIF/HEIC bytes directly into OpenCV BGR numpy array
    with zero lossy recompression (1:1 lossless pixel tensor mapping).
    """
    if not pillow_heif_available:
        # Check if Pillow can open it natively
        try:
            from PIL import Image
            pil_img = Image.open(io.BytesIO(heif_bytes))
            rgb = np.array(pil_img)
            if len(rgb.shape) == 2:
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[2] == 4:
                return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            else:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(
                "iPhone HEIC/HEIF image format detected, but HEIF decoder is not available on this server. "
                "Please upload the image in JPG/PNG format or ensure pillow-heif is installed on the host."
            )

    try:
        from PIL import Image
        heif_file = pillow_heif.open_heif(io.BytesIO(heif_bytes))
        pil_img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
        rgb = np.array(pil_img)
        if len(rgb.shape) == 2:
            return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:
            return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to decode iPhone HEIF/HEIC image: {str(e)}")


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Universal File & Image Decoder:
    - PDF: In-memory 300 DPI rasterization (single-page only).
    - HEIC/HEIF: iPhone format lossless in-memory decoding.
    - JPG, PNG, WebP, BMP: OpenCV standard high-speed decoding.
    """
    if not image_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    # 1. Handle PDF Documents
    if is_pdf(image_bytes):
        return decode_pdf_bytes(image_bytes, dpi=300)

    # 2. Fast Path: Standard Raster Images (JPG, PNG, WebP)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    # 3. Handle iPhone HEIC / HEIF Images
    if is_heif(image_bytes):
        return decode_heif_bytes(image_bytes)

    # 4. Fallback check for HEIF if ftyp was offset or unusual brand
    try:
        return decode_heif_bytes(image_bytes)
    except Exception:
        pass

    raise ValueError("Uploaded file is not a supported image or document format (JPG, PNG, WebP, HEIC, PDF supported).")


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decodes a base64 string (with or without data URI prefix) into OpenCV BGR image."""
    if not base64_str:
        raise ValueError("Empty base64 string provided.")

    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 payload: {str(e)}")

    return decode_image_bytes(image_bytes)
