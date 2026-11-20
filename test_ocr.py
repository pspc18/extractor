import re
import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

for path in [r"d:\extractor\uploads\job_870c4cf9\omr-1.jpeg", "samples/omr-2.jpg"]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    header = img[0:int(h * 0.25), 0:w]
    res, _ = ocr(header)
    exam_name = None
    exam_code = None
    org_name = None
    if res:
        texts = [line[1].strip() for line in res if line[1].strip()]
        for t in texts:
            # Clean spaces around colon
            t_norm = re.sub(r'\s*:\s*', ':', t)
            # Find Exam Name
            m1 = re.search(r'EXAM:([A-Za-z0-9\-_\s]+)', t_norm, re.IGNORECASE)
            if m1 and not exam_name and 'CODE' not in t_norm.upper():
                exam_name = m1.group(1).strip()
            # Find Exam Code
            m2 = re.search(r'(?:EXAM)?CODE:([A-Za-z0-9\-_]+)', t_norm, re.IGNORECASE)
            if m2 and not exam_code:
                exam_code = m2.group(1).strip()
        if texts:
            org_name = texts[0]
    print(f"File: {path}")
    print(f"  Organization: {org_name}")
    print(f"  Exam Name   : {exam_name}")
    print(f"  Exam Code   : {exam_code}")
