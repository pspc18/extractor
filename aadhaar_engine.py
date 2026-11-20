"""
AI Dynamic Aadhaar Card Extractor.
Extracts:
- Name (English & Hindi)
- Father / Husband / Guardian Name (Care Of / S/O / D/O / W/O / C/O)
- Date of Birth (DOB) and Year of Birth (YOB)
- Gender (MALE, FEMALE, TRANSGENDER)
- 12-digit Aadhaar Number (formatted XXXX XXXX XXXX or masked)
- 16-digit Virtual ID (VID)
- Complete Address, Street/House, District, State, and 6-Digit PIN Code
- Handles Front side, Back side, or Both sides (combined or separate uploads)
"""

import base64
from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr_engine = RapidOCR()
except Exception:
    _ocr_engine = None


# Comprehensive list of Indian States & Union Territories
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh",
    "Jammu and Kashmir", "Ladakh", "Puducherry", "Andaman and Nicobar",
    "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep"
]

HEADER_NOISE_WORDS = {
    "government", "govt", "india", "unique", "identification", "authority",
    "uidai", "enrolment", "help", "mera", "aadhaar", "pehchan", "bharat",
    "sarkar", "download", "issue", "date", "electronic", "information",
    "भारतीय", "विशिष्ट", "पहचान", "प्राधिकरण", "सरकार"
}


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file provided.")
    return image


def decode_base64_image(base64_str: str) -> np.ndarray:
    if not base64_str:
        raise ValueError("Empty base64 string provided.")
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(image_bytes)


def clean_spacing(text: str) -> str:
    """Inserts spaces in camelCase or merged words commonly caused by OCR."""
    if not text:
        return ""
    # Split camelCase
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split between letters and numbers
    t = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', t)
    t = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', t)
    # Common Aadhaar prefixes
    t = re.sub(r'([SsCcDdWhH]/[Oo])([A-Za-z])', r'\1 \2', t)
    t = re.sub(r'(Government)(of)(India)', r'\1 \2 \3', t, flags=re.I)
    t = re.sub(r'(Unique)(Identification)', r'\1 \2', t, flags=re.I)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def run_ocr(image: np.ndarray) -> List[Tuple[str, float, List[List[float]]]]:
    """Runs RapidOCR on image and returns list of (text, confidence, box)."""
    if _ocr_engine is None:
        return []
    # If image is very small or very large, normalize
    h, w = image.shape[:2]
    target_w = 1200
    if w < 600 or w > 2400:
        scale = target_w / float(w)
        resized = cv2.resize(image, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    else:
        resized = image

    results, _ = _ocr_engine(resized)
    parsed = []
    if results:
        for item in results:
            box = item[0]
            raw_txt = str(item[1]).strip()
            conf = float(item[2]) if len(item) > 2 else 0.90
            txt = clean_spacing(raw_txt)
            if txt:
                parsed.append((txt, conf, box))
    return parsed


def extract_aadhaar_number(text: str) -> Optional[str]:
    """Finds 12-digit Aadhaar number (standard 4-4-4 or masked)."""
    # Standard format: 4 digits + space + 4 digits + space + 4 digits
    m = re.search(r'\b([2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4})\b', text)
    if m:
        raw = re.sub(r'\s+', '', m.group(1))
        if len(raw) == 12:
            return f"{raw[:4]} {raw[4:8]} {raw[8:]}"
    # Masked format: XXXX XXXX 1234
    m_mask = re.search(r'\b([xX*]{4}\s?[xX*]{4}\s?[0-9]{4})\b', text)
    if m_mask:
        raw = re.sub(r'\s+', '', m_mask.group(1))
        return f"{raw[:4]} {raw[4:8]} {raw[8:]}"
    return None


def extract_vid(text: str) -> Optional[str]:
    """Finds 16-digit Virtual ID (VID)."""
    m = re.search(r'VID[\s:]*([0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4})', text, re.I)
    if m:
        raw = re.sub(r'\s+', '', m.group(1))
        return f"{raw[:4]} {raw[4:8]} {raw[8:12]} {raw[12:]}"
    return None


def extract_dob(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts DOB (DD/MM/YYYY) and YOB (YYYY)."""
    # 1. Full DOB
    m_dob = re.search(r'(?:DOB|D0B|Date of Birth|जन्म तिथि|जन्म तारीख|D\.O\.B)[\s:]*([0-3]?[0-9][/\-.][0-1]?[0-9][/\-.][12][0-9]{3})', text, re.I)
    if m_dob:
        dob_str = m_dob.group(1).replace('-', '/').replace('.', '/')
        # Normalize to DD/MM/YYYY
        parts = dob_str.split('/')
        if len(parts) == 3:
            day = parts[0].zfill(2)
            month = parts[1].zfill(2)
            year = parts[2]
            return f"{day}/{month}/{year}", year

    # 2. Year of Birth only
    m_yob = re.search(r'(?:Year of Birth|जन्म वर्ष|YOB)[\s:]*([12][0-9]{3})', text, re.I)
    if m_yob:
        y = m_yob.group(1)
        return None, y

    # 3. Fallback: match standalone date
    m_std = re.search(r'\b([0-3][0-9]/[0-1][0-9]/[12][0-9]{3})\b', text)
    if m_std:
        parts = m_std.group(1).split('/')
        return m_std.group(1), parts[2]

    return None, None


def extract_gender(text: str) -> Optional[str]:
    """Extracts Gender: MALE, FEMALE, or TRANSGENDER."""
    m = re.search(r'\b(FEMALE|महिला)\b', text, re.I)
    if m:
        return "FEMALE"
    m_trans = re.search(r'\b(TRANSGENDER|किन्नर|तृतीय लिंग)\b', text, re.I)
    if m_trans:
        return "TRANSGENDER"
    m_male = re.search(r'\b(MALE|पुरुष)\b', text, re.I)
    if m_male:
        return "MALE"
    return None


def extract_pincode(text: str) -> Optional[str]:
    """Extracts 6-digit Indian PIN code."""
    pins = re.findall(r'\b([1-9][0-9]{5})\b', text)
    if pins:
        # Return the last valid 6-digit pin found (usually at end of address)
        return pins[-1]
    return None


def extract_state_and_district(address_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detects State and probable District from address text."""
    found_state = None
    for state in INDIAN_STATES:
        if re.search(r'\b' + re.escape(state) + r'\b', address_text, re.I):
            found_state = state
            break

    district = None
    # If state is found, look for city/district immediately preceding it
    if found_state:
        # Split by comma or dashes
        tokens = [t.strip() for t in re.split(r'[,;\n\-]', address_text) if t.strip()]
        for i, token in enumerate(tokens):
            if found_state.lower() in token.lower() and i > 0:
                prev = tokens[i - 1]
                # Filter out numbers and common noise
                clean_prev = re.sub(r'[^A-Za-z\s]', '', prev).strip()
                if 2 < len(clean_prev) < 25 and not any(w.lower() in HEADER_NOISE_WORDS for w in clean_prev.split()):
                    district = clean_prev
                break

    return found_state, district


def extract_care_of(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts Care of (C/O, S/O, D/O, W/O, H/O) name and relation type.
    """
    patterns = [
        (r'\b(?:S/O|SO|आत्मज|पुत्र|सुपुत्र)[\s:]+([A-Za-z\s\.]+?)(?:,|$|\n|[0-9])', "FATHER"),
        (r'\b(?:D/O|DO|सुपुत्री|पुत्री)[\s:]+([A-Za-z\s\.]+?)(?:,|$|\n|[0-9])', "FATHER"),
        (r'\b(?:W/O|WO|पत्नी)[\s:]+([A-Za-z\s\.]+?)(?:,|$|\n|[0-9])', "HUSBAND"),
        (r'\b(?:H/O|HO|पति)[\s:]+([A-Za-z\s\.]+?)(?:,|$|\n|[0-9])', "HUSBAND"),
        (r'\b(?:C/O|CO)[\s:]+([A-Za-z\s\.]+?)(?:,|$|\n|[0-9])', "GUARDIAN"),
    ]

    for pat, rel in patterns:
        m = re.search(pat, text, re.I)
        if m:
            raw_name = m.group(1).strip()
            # Clean up noise words
            clean_name = re.sub(r'^(?:Late|Shri|Mr\.?|Smt\.?)\s+', '', raw_name, flags=re.I).strip()
            if 2 <= len(clean_name.split()) <= 4:
                return clean_name, rel

    return None, None


def extract_name_candidates(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Intelligently identifies English and Hindi names from card front text lines."""
    name_en = None
    name_hi = None

    stop_words = HEADER_NOISE_WORDS.union({
        "dob", "d0b", "date", "birth", "gender", "male", "female", "transgender",
        "year", "yob", "father", "mother", "husband", "wife", "address", "vid"
    })

    for line in lines:
        clean = line.strip()
        words = clean.split()
        if not words or len(words) > 5:
            continue

        # Hindi Name Detection
        if re.search(r'[\u0900-\u097F]', clean) and not name_hi:
            if not re.search(r'(?:जन्म|पुरुष|महिला|तिथि|वर्ष|पता|सरकार|प्राधिकरण)', clean):
                if not any(w.lower() in stop_words for w in words):
                    name_hi = clean

        # English Name Detection
        if re.match(r'^[A-Za-z\s\.]+$', clean) and not name_en:
            clean_lower = clean.lower()
            if not any(sw in clean_lower for sw in stop_words):
                if 2 <= len(words) <= 4 and len(clean) >= 4:
                    name_en = clean.title()

    return name_en, name_hi


def clean_address(raw_addr: str) -> str:
    """Cleans up raw OCR address string into a neat comma-separated address."""
    # Remove 'Address:' or 'पता:' prefix
    addr = re.sub(r'^(?:Address|पता|To)[\s:]*', '', raw_addr, flags=re.I).strip()
    # Normalize whitespaces and commas
    addr = re.sub(r'[\r\n]+', ', ', addr)
    addr = re.sub(r'\s*,\s*', ', ', addr)
    addr = re.sub(r'\s+', ' ', addr)
    addr = addr.strip(' ,-')
    return addr


def detect_card_side(full_text: str) -> str:
    """Detects if the card is FRONT, BACK, or BOTH."""
    is_front = bool(re.search(r'\b(DOB|D0B|Date of Birth|जन्म तिथि|MALE|FEMALE|पुरुष|महिला)\b', full_text, re.I))
    is_back = bool(re.search(r'\b(Address|पता|C/O|S/O|D/O|W/O|PIN)\b', full_text, re.I))

    if is_front and is_back:
        return "BOTH"
    elif is_front:
        return "FRONT"
    elif is_back:
        return "BACK"
    return "UNKNOWN"


def extract_aadhaar_from_image(image: np.ndarray, filename: str = "aadhaar.jpg") -> Dict[str, Any]:
    """Processes a single Aadhaar card image (can be Front, Back, or Combined)."""
    ocr_results = run_ocr(image)
    if not ocr_results:
        return {
            "success": False,
            "error": "No text detected in the uploaded image. Please ensure good lighting and clear capture.",
            "filename": filename
        }

    lines = [item[0] for item in ocr_results]
    full_text = "\n".join(lines)
    side = detect_card_side(full_text)

    # 1. Aadhaar Number & VID
    aadhaar_num = extract_aadhaar_number(full_text)
    vid = extract_vid(full_text)

    # 2. Front Details
    dob, yob = extract_dob(full_text)
    gender = extract_gender(full_text)
    name_en, name_hi = extract_name_candidates(lines)

    # 3. Back Details (Address & Care of)
    care_of_name, relation = extract_care_of(full_text)
    pincode = extract_pincode(full_text)

    # Address extraction
    address_str = None
    m_addr = re.search(r'(?:Address|पता)[\s:]*(.+?)(?:[1-9][0-9]{5}|\Z)', full_text, re.DOTALL | re.I)
    if m_addr:
        raw_addr = m_addr.group(1).strip()
        address_str = clean_address(raw_addr)
        if pincode and pincode not in address_str:
            address_str = f"{address_str}, {pincode}"

    state, district = extract_state_and_district(address_str or full_text)

    # Confidence calculation
    confidences = [item[1] for item in ocr_results]
    avg_conf = round(float(np.mean(confidences)), 3) if confidences else 0.85

    return {
        "success": True,
        "filename": filename,
        "card_side": side,
        "confidence": avg_conf,
        "data": {
            "aadhaar_number": aadhaar_num,
            "vid": vid,
            "name": name_en,
            "name_hindi": name_hi,
            "dob": dob,
            "yob": yob,
            "gender": gender,
            "care_of": care_of_name,
            "relation": relation,
            "address": address_str,
            "district": district,
            "state": state,
            "pincode": pincode
        },
        "raw_text_lines": lines
    }


def extract_aadhaar_dual_side(
    front_image: Optional[np.ndarray] = None,
    back_image: Optional[np.ndarray] = None,
    front_filename: str = "front.jpg",
    back_filename: str = "back.jpg"
) -> Dict[str, Any]:
    """Combines extraction from both Front and Back images for 100% complete Aadhaar profile."""
    res_front = extract_aadhaar_from_image(front_image, front_filename) if front_image is not None else None
    res_back = extract_aadhaar_from_image(back_image, back_filename) if back_image is not None else None

    # Merge data
    data_front = res_front.get("data", {}) if res_front and res_front.get("success") else {}
    data_back = res_back.get("data", {}) if res_back and res_back.get("success") else {}

    merged_data = {
        "aadhaar_number": data_front.get("aadhaar_number") or data_back.get("aadhaar_number"),
        "vid": data_front.get("vid") or data_back.get("vid"),
        "name": data_front.get("name") or data_back.get("name"),
        "name_hindi": data_front.get("name_hindi") or data_back.get("name_hindi"),
        "dob": data_front.get("dob") or data_back.get("dob"),
        "yob": data_front.get("yob") or data_back.get("yob"),
        "gender": data_front.get("gender") or data_back.get("gender"),
        "care_of": data_back.get("care_of") or data_front.get("care_of"),
        "relation": data_back.get("relation") or data_front.get("relation"),
        "address": data_back.get("address") or data_front.get("address"),
        "district": data_back.get("district") or data_front.get("district"),
        "state": data_back.get("state") or data_front.get("state"),
        "pincode": data_back.get("pincode") or data_front.get("pincode"),
    }

    conf_list = []
    if res_front and "confidence" in res_front:
        conf_list.append(res_front["confidence"])
    if res_back and "confidence" in res_back:
        conf_list.append(res_back["confidence"])

    overall_conf = round(float(np.mean(conf_list)), 3) if conf_list else 0.90

    return {
        "success": True,
        "card_side": "BOTH",
        "confidence": overall_conf,
        "data": merged_data,
        "front_status": res_front.get("success", False) if res_front else None,
        "back_status": res_back.get("success", False) if res_back else None
    }
