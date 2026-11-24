"""
Aadhaar QR Code Extraction & Decoding Engine.
Supports:
1. UIDAI Secure QR Code (V2 & V3 - Gzip/Zlib compressed big-endian integer/bytes)
   - Decodes: Name, DOB, Gender, Aadhaar Last 4 / Reference ID, Care Of, Full Address, Pincode
   - Extracts embedded JPEG cardholder photo if present
2. UIDAI Legacy XML QR Code (<PrintLetterBarcodeData ...> / <QPData ...>)
   - Decodes: Full 12-digit Aadhaar UID, Name, DOB, Gender, Care Of, Address, Pincode
3. JSON / Standardized QR payloads
4. High-Performance Camera & Image Detection via zxing-cpp and OpenCV fallback
"""

import base64
import gzip
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import zlib

import cv2
import numpy as np

try:
    import zxingcpp
    _HAS_ZXING = True
except Exception:
    _HAS_ZXING = False


def _decompress_payload(raw_bytes: bytes) -> Optional[bytes]:
    """Decompresses gzip or zlib compressed binary payloads."""
    if not raw_bytes:
        return None

    # Try standard gzip
    if raw_bytes.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(raw_bytes)
        except Exception:
            pass

    # Try zlib with varying window bits
    for w in (zlib.MAX_WBITS, 16 + zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(raw_bytes, w)
        except Exception:
            continue

    # Try direct gzip fallback
    try:
        return gzip.decompress(raw_bytes)
    except Exception:
        pass

    return None


def _clean_address_parts(*parts: str) -> str:
    """Assembles a clean, de-duplicated, comma-separated address string."""
    seen = set()
    clean = []
    for p in parts:
        if not p or not isinstance(p, str):
            continue
        p_clean = p.strip()
        if not p_clean:
            continue
        # Strip redundant C/O prefix if present in address line
        p_clean = re.sub(r"^(?:C/O|S/O|D/O|W/O|Care\s+of)[:\s]+", "", p_clean, flags=re.IGNORECASE).strip()
        if p_clean and p_clean.lower() not in seen:
            clean.append(p_clean)
            seen.add(p_clean.lower())
    return ", ".join(clean)


def _normalize_dob(dob_str: str) -> str:
    """Normalizes any date string into standard DD/MM/YYYY format."""
    if not dob_str:
        return ""
    d = dob_str.strip()
    # Check DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", d)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{day:02d}/{month:02d}/{year:04d}"
    # Check YYYY-MM-DD
    m2 = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", d)
    if m2:
        year, month, day = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return f"{day:02d}/{month:02d}/{year:04d}"
    # Year only (e.g. 1995)
    if re.match(r"^\d{4}$", d):
        return f"01/01/{d}"
    return d


def _normalize_gender(g_str: str) -> str:
    """Normalizes gender to MALE, FEMALE, or OTHER."""
    if not g_str:
        return ""
    g = g_str.strip().upper()
    if g.startswith("M") and "FE" not in g:
        return "MALE"
    if g.startswith("F") or "FEMALE" in g or g == "FEM":
        return "FEMALE"
    if g.startswith("T") or "TRANS" in g:
        return "TRANSGENDER"
    return g


def parse_xml_aadhaar(xml_str: str) -> Optional[Dict[str, Any]]:
    """Parses Legacy UIDAI XML QR Code string (<PrintLetterBarcodeData ...>)."""
    if not xml_str:
        return None
    # Locate XML tag
    match = re.search(
        r"<(?:\?xml.*?\?>\s*)?(?:PrintLetterBarcodeData|QPData)[^>]*>",
        xml_str,
        re.DOTALL | re.IGNORECASE
    )
    if not match:
        return None

    cleaned_xml = match.group(0).strip()
    if not cleaned_xml.endswith("/>") and not cleaned_xml.endswith(">"):
        cleaned_xml = cleaned_xml + "/>"
    elif cleaned_xml.endswith(">") and not cleaned_xml.endswith("/>"):
        cleaned_xml = cleaned_xml[:-1] + "/>"

    try:
        root = ET.fromstring(cleaned_xml)
        attr = root.attrib

        uid = attr.get("uid", "").strip()
        name = attr.get("name", "").strip()
        gender = _normalize_gender(attr.get("gender", ""))
        dob = _normalize_dob(attr.get("dob", "") or attr.get("yob", ""))
        co = attr.get("co", "").strip()

        # Build address
        clean_addr = _clean_address_parts(
            attr.get("house", ""),
            attr.get("street", ""),
            attr.get("lm", ""),
            attr.get("loc", ""),
            attr.get("vtc", ""),
            attr.get("po", ""),
            attr.get("subdist", ""),
            attr.get("dist", ""),
            attr.get("state", ""),
            attr.get("pc", "")
        )
        pincode = attr.get("pc", "").strip()
        formatted_aadhaar = f"{uid[:4]} {uid[4:8]} {uid[8:]}" if len(uid) == 12 else uid

        return {
            "name": name,
            "dob": dob,
            "gender": gender,
            "aadhaar_number": formatted_aadhaar,
            "care_of": co,
            "address": clean_addr,
            "pincode": pincode,
            "photo_base64": None,
            "qr_type": "UIDAI_XML_QR",
            "source": "QR_CODE"
        }
    except Exception:
        return None


def parse_secure_aadhaar_qr(raw_input: Union[str, bytes]) -> Optional[Dict[str, Any]]:
    """
    Parses UIDAI Secure QR Code (Big-endian integer string or raw compressed bytes).
    Extracts complete demographic profile and embedded photo if present.
    """
    decompressed: Optional[bytes] = None

    if isinstance(raw_input, str):
        cleaned_str = raw_input.strip()
        # Case 1: Big integer string
        if cleaned_str.isdigit() and len(cleaned_str) > 80:
            try:
                big_int = int(cleaned_str)
                byte_length = (big_int.bit_length() + 7) // 8
                raw_bytes = big_int.to_bytes(byte_length, byteorder="big")
                decompressed = _decompress_payload(raw_bytes)
            except Exception:
                decompressed = None
        # Case 2: Base64 string of compressed bytes
        if decompressed is None:
            try:
                b64_bytes = base64.b64decode(cleaned_str)
                decompressed = _decompress_payload(b64_bytes)
            except Exception:
                pass
    elif isinstance(raw_input, (bytes, bytearray)):
        decompressed = _decompress_payload(bytes(raw_input))
        if decompressed is None and raw_input.isdigit() and len(raw_input) > 80:
            try:
                big_int = int(raw_input.decode("ascii"))
                byte_length = (big_int.bit_length() + 7) // 8
                raw_bytes = big_int.to_bytes(byte_length, byteorder="big")
                decompressed = _decompress_payload(raw_bytes)
            except Exception:
                pass

    if not decompressed:
        return None

    # Locate cardholder JPEG photo if embedded
    photo_base64 = None
    jpeg_start = decompressed.find(b"\xff\xd8")
    if jpeg_start != -1:
        text_section = decompressed[:jpeg_start]
        jpeg_end = decompressed.find(b"\xff\xd9", jpeg_start)
        if jpeg_end != -1:
            photo_bytes = decompressed[jpeg_start : jpeg_end + 2]
        else:
            photo_bytes = decompressed[jpeg_start:]
        if photo_bytes and len(photo_bytes) > 10:
            photo_base64 = "data:image/jpeg;base64," + base64.b64encode(photo_bytes).decode("ascii")
    else:
        text_section = decompressed

    # Split fields by delimiter \xff (255)
    raw_fields = text_section.split(b"\xff")
    fields = [f.decode("utf-8", errors="replace").strip() for f in raw_fields]

    if len(fields) < 5:
        return None

    # Map standard UIDAI Secure QR fields:
    # 0: Email/Mobile flag
    # 1: Reference ID (first 4 digits contain last 4 of Aadhaar)
    # 2: Name
    # 3: DOB (DD-MM-YYYY)
    # 4: Gender ('M'/'F')
    # 5: Care of
    # 6: District
    # 7: Landmark
    # 8: House
    # 9: Location
    # 10: Pincode
    # 11: Post Office
    # 12: State
    # 13: Street
    # 14: Sub District
    # 15: VTC (Village / Town / City)
    email_mobile_flag = fields[0] if len(fields) > 0 else ""
    reference_id = fields[1] if len(fields) > 1 else ""
    name = fields[2] if len(fields) > 2 else ""
    dob_raw = fields[3] if len(fields) > 3 else ""
    gender_raw = fields[4] if len(fields) > 4 else ""
    care_of = fields[5] if len(fields) > 5 else ""
    district = fields[6] if len(fields) > 6 else ""
    landmark = fields[7] if len(fields) > 7 else ""
    house = fields[8] if len(fields) > 8 else ""
    location = fields[9] if len(fields) > 9 else ""
    pincode = fields[10] if len(fields) > 10 else ""
    post_office = fields[11] if len(fields) > 11 else ""
    state = fields[12] if len(fields) > 12 else ""
    street = fields[13] if len(fields) > 13 else ""
    subdistrict = fields[14] if len(fields) > 14 else ""
    vtc = fields[15] if len(fields) > 15 else ""

    # Build clean Aadhaar number
    # If reference_id starts with 4 digits, format as masked Aadhaar: "XXXX XXXX 1234"
    masked_aadhaar = ""
    ref_match = re.search(r"(\d{4})", reference_id)
    if ref_match:
        last4 = ref_match.group(1)
        masked_aadhaar = f"XXXX XXXX {last4}"

    # Build direct, clean, human-readable address
    clean_addr = _clean_address_parts(
        house, street, landmark, location, vtc, post_office, subdistrict, district, state, pincode
    )

    return {
        "name": name,
        "dob": _normalize_dob(dob_raw),
        "gender": _normalize_gender(gender_raw),
        "aadhaar_number": masked_aadhaar,
        "reference_id": reference_id,
        "care_of": care_of,
        "address": clean_addr,
        "pincode": pincode,
        "photo_base64": photo_base64,
        "qr_type": "UIDAI_SECURE_QR",
        "source": "QR_CODE"
    }


def parse_json_aadhaar_qr(qr_str: str) -> Optional[Dict[str, Any]]:
    """Parses JSON-encoded Aadhaar QR payloads."""
    if not qr_str or not (qr_str.strip().startswith("{") and qr_str.strip().endswith("}")):
        return None
    try:
        data = json.loads(qr_str.strip())
        name = data.get("name") or data.get("full_name") or ""
        dob = _normalize_dob(data.get("dob") or data.get("date_of_birth") or data.get("yob") or "")
        gender = _normalize_gender(data.get("gender") or data.get("sex") or "")
        uid = data.get("aadhaar") or data.get("aadhaar_number") or data.get("uid") or ""
        address = data.get("address") or data.get("addr") or ""
        pincode = data.get("pincode") or data.get("pin") or data.get("postal_code") or ""
        co = data.get("care_of") or data.get("co") or ""

        if not name and not uid:
            return None

        formatted_aadhaar = f"{uid[:4]} {uid[4:8]} {uid[8:]}" if len(uid) == 12 else uid

        return {
            "name": name,
            "dob": dob,
            "gender": gender,
            "aadhaar_number": formatted_aadhaar,
            "care_of": co,
            "address": address,
            "pincode": pincode,
            "photo_base64": data.get("photo_base64") or data.get("photo"),
            "qr_type": "JSON_QR",
            "source": "QR_CODE"
        }
    except Exception:
        return None


def decode_aadhaar_qr(raw_input: Union[str, bytes]) -> Optional[Dict[str, Any]]:
    """
    Unified Aadhaar QR Decoder.
    Automatically identifies whether raw input is:
    1. UIDAI Secure QR (Compressed BigInt / Bytes)
    2. UIDAI XML QR
    3. JSON QR
    Returns standardized demographic dictionary or None.
    """
    if not raw_input:
        return None

    # Check 1: Secure QR
    res_secure = parse_secure_aadhaar_qr(raw_input)
    if res_secure and (res_secure.get("name") or res_secure.get("pincode")):
        return res_secure

    # Convert to string for text-based checks
    str_input = raw_input.decode("utf-8", errors="ignore") if isinstance(raw_input, (bytes, bytearray)) else str(raw_input)

    # Check 2: XML QR
    res_xml = parse_xml_aadhaar(str_input)
    if res_xml and (res_xml.get("name") or res_xml.get("aadhaar_number")):
        return res_xml

    # Check 3: JSON QR
    res_json = parse_json_aadhaar_qr(str_input)
    if res_json:
        return res_json

    return None


def detect_and_decode_qr_from_image(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Scans an image (NumPy array) for Aadhaar QR codes.
    Applies multi-scale contrast and rotation transforms for maximum detection rate.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    # 1. Primary Scan via zxing-cpp
    if _HAS_ZXING:
        # Scan raw image
        barcodes = zxingcpp.read_barcodes(image)
        for b in barcodes:
            # Check bytes first (essential for Secure QR)
            if hasattr(b, "bytes") and b.bytes:
                decoded = decode_aadhaar_qr(b.bytes)
                if decoded:
                    return decoded
            if hasattr(b, "text") and b.text:
                decoded = decode_aadhaar_qr(b.text)
                if decoded:
                    return decoded

        # Grayscale & CLAHE contrast boost
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        barcodes_enh = zxingcpp.read_barcodes(enhanced)
        for b in barcodes_enh:
            if hasattr(b, "bytes") and b.bytes:
                decoded = decode_aadhaar_qr(b.bytes)
                if decoded:
                    return decoded
            if hasattr(b, "text") and b.text:
                decoded = decode_aadhaar_qr(b.text)
                if decoded:
                    return decoded

        # 90 and 270 degree rotations for sideways captures
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            rotated = cv2.rotate(gray, rot)
            rot_barcodes = zxingcpp.read_barcodes(rotated)
            for b in rot_barcodes:
                if hasattr(b, "bytes") and b.bytes:
                    decoded = decode_aadhaar_qr(b.bytes)
                    if decoded:
                        return decoded
                if hasattr(b, "text") and b.text:
                    decoded = decode_aadhaar_qr(b.text)
                    if decoded:
                        return decoded

    # 2. Fallback Scan via OpenCV QRCodeDetector
    try:
        detector = cv2.QRCodeDetector()
        val, pts, _ = detector.detectAndDecode(image)
        if val:
            decoded = decode_aadhaar_qr(val)
            if decoded:
                return decoded
    except Exception:
        pass

    return None
