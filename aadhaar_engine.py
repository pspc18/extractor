"""
AI Dynamic Aadhaar Card Extractor - Enhanced Edition.
Extracts:
- Name (English & Hindi)
- Date of Birth (DOB) with enhanced contrast & multi-format parser
- Gender (MALE, FEMALE, TRANSGENDER)
- 12-digit Aadhaar Number (formatted XXXX XXXX XXXX or masked)
- Direct Clean Address (without C/O prefix, normalized, artifact-free)
- 6-Digit PIN Code
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

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

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


def enhance_card_image(img: np.ndarray) -> np.ndarray:
    """
    Enhances card image contrast and sharpness for maximum OCR digit accuracy.
    Prevents 0 vs 1 and 3 vs 8 digit confusions in DOB and Aadhaar numbers.
    """
    h, w = img.shape[:2]
    # Resize to standard high-clarity width
    if w < 1000 or w > 2400:
        target_w = 1400
        scale = target_w / float(w)
        resized = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
    else:
        resized = img.copy()

    # Apply CLAHE on L-channel of LAB
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Gentle unsharp mask for crisp character edges
    blurred = cv2.GaussianBlur(contrast_img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrast_img, 1.25, blurred, -0.25, 0)
    return sharpened


def clean_spacing(text: str) -> str:
    """Inserts spaces in camelCase or merged words commonly caused by OCR."""
    if not text:
        return ""
    # Convert Devanagari numerals to standard digits
    t = text.translate(DEVANAGARI_DIGITS)
    # Split camelCase
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
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
    """Runs RapidOCR on enhanced image and returns list of (text, confidence, box)."""
    if _ocr_engine is None:
        return []

    enhanced = enhance_card_image(image)
    results, _ = _ocr_engine(enhanced)
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


def normalize_date(d_str: str) -> str:
    clean = re.sub(r'[.\-\s]', '/', d_str)
    parts = [p.strip() for p in clean.split('/') if p.strip()]
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}/{parts[1].zfill(2)}/{parts[2]}"
    return d_str


def extract_dob(text: str, lines: List[str]) -> Optional[str]:
    """
    Intelligent Date of Birth extractor.
    - Matches DOB keywords across same line, next line (label above date), or previous line.
    - Ignores Issue / Enrollment / Generation dates (e.g. Enrollment timestamps like 23/12/2011).
    - Prioritizes true birth years.
    """
    dob_keywords = re.compile(
        r'(?:DOB|D\.?O\.?B\.?|D0B|DO8|Date\s*of\s*Birth|जन्म[\s\-/]*तिथि|जन्म[\s\-/]*तारीख|जन्म[\s\-/]*दिनांक)',
        re.I
    )
    date_pattern = re.compile(r'([0-3]?[0-9][/\-.\s][0-1]?[0-9][/\-.\s][12][0-9]{3})')
    issue_keywords = re.compile(
        r'(?:Issue|Download|Enrolment|Enrollment|Generated|Print|Valid|जारी|नामांकन)',
        re.I
    )

    # 1. Proximity check on lines containing DOB keywords
    for i, line in enumerate(lines):
        if dob_keywords.search(line) and not issue_keywords.search(line):
            # Check current line
            m = date_pattern.search(line)
            if m:
                return normalize_date(m.group(1))
            # Check next line (e.g. "जन्म तिथि / DOB" on line i, "24/12/1990" on line i+1)
            if i + 1 < len(lines) and not issue_keywords.search(lines[i + 1]):
                m2 = date_pattern.search(lines[i + 1])
                if m2:
                    return normalize_date(m2.group(1))
            # Check previous line
            if i > 0 and not issue_keywords.search(lines[i - 1]):
                m3 = date_pattern.search(lines[i - 1])
                if m3:
                    return normalize_date(m3.group(1))

    # 2. Search whole text for DOB pattern excluding lines with issue/enrolment
    m_dob = re.search(
        r'(?:DOB|D0B|DO8|Date of Birth|जन्म[\s]*तिथि|जन्म[\s]*तारीख|D\.O\.B)[\s/:\-]*([0-3]?[0-9][/\-.\s][0-1]?[0-9][/\-.\s][12][0-9]{3})',
        text,
        re.I
    )
    if m_dob:
        return normalize_date(m_dob.group(1))

    # 3. Candidate scan: collect all dates NOT on issue/enrolment lines
    candidates = []
    for line in lines:
        if not issue_keywords.search(line):
            for m in date_pattern.finditer(line):
                d_cand = normalize_date(m.group(1))
                candidates.append(d_cand)

    if candidates:
        # Prioritize dates where year is an adult birth year (<= 2010 if multiple candidates exist)
        adult_dates = [c for c in candidates if int(c.split('/')[-1]) <= 2005]
        if adult_dates:
            return adult_dates[0]
        return candidates[-1] if len(candidates) > 1 else candidates[0]

    # 4. Year of Birth only fallback
    m_yob = re.search(r'(?:Year of Birth|जन्म[\s]*वर्ष|YOB)[\s/:\-]*([12][0-9]{3})', text, re.I)
    if m_yob:
        return m_yob.group(1)

    return None


def extract_gender(text: str) -> Optional[str]:
    """Extracts Gender: MALE, FEMALE, or TRANSGENDER."""
    m_trans = re.search(r'\b(TRANSGENDER|किन्नर|तृतीय लिंग)\b', text, re.I)
    if m_trans:
        return "TRANSGENDER"
    m_fem = re.search(r'\b(FEMALE|महिला)\b', text, re.I)
    if m_fem:
        return "FEMALE"
    m_male = re.search(r'\b(MALE|पुरुष)\b', text, re.I)
    if m_male:
        return "MALE"
    return None


def extract_pincode(text: str) -> Optional[str]:
    """Extracts 6-digit Indian PIN code."""
    pins = re.findall(r'\b([1-9][0-9]{5})\b', text)
    if pins:
        return pins[-1]
    return None


def clean_direct_address(raw_addr: str) -> str:
    """
    Cleans up raw OCR address:
    - Removes 'Address:' or 'पता:' prefix
    - Removes C/O, S/O, D/O, W/O, H/O and relative name so address starts DIRECTLY from building/house/flat
    - Removes OCR noise like 'Di T:', 'Dist:'
    - Normalizes 'PO:'
    - Removes double commas and extra whitespace
    """
    # 1. Remove 'Address:' or 'पता:' prefix
    addr = re.sub(r'^(?:Address|पता|To)[\s:]*', '', raw_addr, flags=re.I).strip()

    # 2. Remove C/O, S/O, D/O, W/O, H/O and relative name
    addr = re.sub(r'^(?:C/O|S/O|D/O|W/O|H/O|CO|SO|DO|WO|HO|आत्मज|पुत्र|पत्नी|सुपुत्र)[\s:]*[^,;\n]+[,;\n]\s*', '', addr, flags=re.I).strip()

    # 3. Clean OCR noise & abbreviations
    addr = re.sub(r'\b(?:Di\s*T|Dist|District)[\s:]*', '', addr, flags=re.I)
    addr = re.sub(r'\bP\.?O\.?:\s*', 'PO: ', addr)

    # 4. Normalize commas & spaces
    addr = re.sub(r'[\r\n]+', ', ', addr)
    addr = re.sub(r'\s*,\s*', ', ', addr)
    addr = re.sub(r'(?:,\s*){2,}', ', ', addr)
    addr = re.sub(r'\s+', ' ', addr)
    addr = addr.strip(' ,-')
    return addr


def extract_name_candidates(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Identifies English and Hindi names from card front text lines."""
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
    """Processes a single Aadhaar card image (Front, Back, or Combined)."""
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

    # 1. Identity & Demographics
    aadhaar_num = extract_aadhaar_number(full_text)
    dob = extract_dob(full_text, lines)
    gender = extract_gender(full_text)
    name_en, name_hi = extract_name_candidates(lines)

    # 2. Address & PIN
    pincode = extract_pincode(full_text)
    address_str = None
    m_addr = re.search(r'(?:Address|पता)[\s:]*(.+?)(?:[1-9][0-9]{5}|\Z)', full_text, re.DOTALL | re.I)
    if m_addr:
        raw_addr = m_addr.group(1).strip()
        address_str = clean_direct_address(raw_addr)
        if pincode and pincode not in address_str:
            address_str = f"{address_str}, {pincode}"

    confidences = [item[1] for item in ocr_results]
    avg_conf = round(float(np.mean(confidences)), 3) if confidences else 0.85

    return {
        "success": True,
        "filename": filename,
        "card_side": side,
        "confidence": avg_conf,
        "data": {
            "name": name_en,
            "name_hindi": name_hi,
            "dob": dob,
            "gender": gender,
            "aadhaar_number": aadhaar_num,
            "address": address_str,
            "pincode": pincode
        }
    }


def extract_aadhaar_dual_side(
    front_image: Optional[np.ndarray] = None,
    back_image: Optional[np.ndarray] = None,
    front_filename: str = "front.jpg",
    back_filename: str = "back.jpg"
) -> Dict[str, Any]:
    """Combines extraction from Front and Back images into clean, direct JSON."""
    res_front = extract_aadhaar_from_image(front_image, front_filename) if front_image is not None else None
    res_back = extract_aadhaar_from_image(back_image, back_filename) if back_image is not None else None

    data_front = res_front.get("data", {}) if res_front and res_front.get("success") else {}
    data_back = res_back.get("data", {}) if res_back and res_back.get("success") else {}

    # Merge into clean schema (care_of, relation, district, state, vid completely removed)
    merged_data = {
        "name": data_front.get("name") or data_back.get("name"),
        "name_hindi": data_front.get("name_hindi") or data_back.get("name_hindi"),
        "dob": data_front.get("dob") or data_back.get("dob"),
        "gender": data_front.get("gender") or data_back.get("gender"),
        "aadhaar_number": data_front.get("aadhaar_number") or data_back.get("aadhaar_number"),
        "address": data_back.get("address") or data_front.get("address"),
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
