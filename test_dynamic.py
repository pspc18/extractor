import cv2
from pathlib import Path
from dynamic_engine import process_dynamic_omr

test_paths = [
    Path(r"d:\extractor\uploads\job_870c4cf9\omr-1.jpeg"),
    Path("samples/omr-2.jpg"),
    Path("samples/omr-mobile-tilt.jpg"),
    Path("samples/omr-black-pen.jpg"),
    Path(r"d:\omr_extractor\sample-omr-filled-50.png"),
    Path(r"d:\omr_extractor\sample-omr-random-10-two-skipped.png"),
]

for p in test_paths:
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    res = process_dynamic_omr(img, filename=p.name)
    print(f"[OK] {res['filename']:<36} | Exam: {str(res['exam_name']):<16} | Roll: {res['roll_number']:<8} | Qs: {res['summary']['total_questions']:<3} | Attempted: {res['summary']['attempted']:<3} | Q50: {str(res['answers'].get('50'))}")
