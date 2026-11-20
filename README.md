# High-Volume OMR Sheet Scanner & Evaluation System

A production-grade, self-hosted, zero-cost Optical Mark Recognition (OMR) processing engine designed to process **3,000+ sheets/day** (benchmark capacity: **150,000+ sheets/day**) at **₹0 API cost** using **Python, OpenCV, NumPy, ProcessPoolExecutor, FastAPI, and SQLite**.

---

## Key Features

1. **Auto-Alignment via 4-Point Perspective Transform:**
   - Detects 4 solid black corner fiducials (`MARKER_TARGET_CENTERS`) reliably, even on smartphone camera captures with perspective tilt, rotations, and uneven lighting.
   - Normalizes sheets to standardized `1200x1600` canvas resolution.

2. **Dual-Ink Bubble Extraction:**
   - Detects both **Blue ballpoint pen** ink (via BGR colorimetry & HSV saturation) and **Black ballpoint / charcoal pencil** marks (via adaptive Otsu fill-ratios).
   - Local annulus background normalization prevents false positives caused by shadows or dark paper grain.

3. **Production Edge-Case Handling:**
   - **Unattempted row:** Detected as `null`.
   - **Multiple bubbles filled in one row:** Detected as `"MULTIPLE"`.
   - **Ambiguous / half-filled bubble or failed corner detection:** Automatically flagged with `status: "NEEDS_REVIEW"` and detailed reasons.

4. **Multi-Core High-Throughput Engine:**
   - Uses Python's `ProcessPoolExecutor` to distribute image processing across all CPU cores.
   - Achieves **~1.8 to 2.5 sheets/sec per 8-core CPU** (~160,000 sheets/24 hours), far exceeding the 3,000 sheets/day requirement.

5. **Human-In-The-Loop Web Dashboard:**
   - Built with FastAPI, Tailwind CSS, and vanilla JS.
   - Bulk upload of individual images or `.zip` archives.
   - Optional Answer Key JSON upload with positive/negative grading (+4/-1).
   - Dedicated **"Needs Review"** console showing the annotated debug image side-by-side with interactive correction controls.
   - One-click export to **JSON** and **CSV**.

6. **Offline CLI (`batch_cli.py`):**
   - Headless CLI for offline processing of large folders.

7. **Rapid Calibration (`calibrate.py`):**
   - Overlays configured coordinate boxes on `samples/omr-2.jpg` and outputs `debug_grid.jpg` for tuning in seconds.

---

## Project Structure

```
d:\extractor/
├── config.py                 # Canvas geometry, fiducial markers, bubble ROIs, and thresholds
├── omr_engine.py             # OpenCV alignment, bubble fill scoring, grading, and annotation
├── calibrate.py              # Visual calibration tool -> debug_grid.jpg
├── generate_template.py      # Generates printable blank template & synthetic sample sheets
├── database.py               # SQLite schema & repository for batch jobs and sheet results
├── batch_processor.py        # ProcessPoolExecutor parallel multi-core engine
├── batch_cli.py              # Offline command-line batch runner with JSON/CSV export
├── app.py                    # FastAPI backend server with REST APIs
├── start.bat                 # One-click Windows launch script
├── requirements.txt          # Dependencies
├── static/
│   └── index.html            # Web Dashboard (Tailwind CSS + JS)
├── samples/
│   ├── omr_template.png      # Clean printable blank template (1200x1600)
│   ├── omr-2.jpg             # Reference sample (Roll: 053826, 5 attempted questions)
│   ├── omr-mobile-tilt.jpg   # Mobile camera capture simulation with perspective tilt
│   ├── omr-black-pen.jpg     # Black ballpoint pen sample
│   ├── answer_key.json       # 50-question answer key for instant grading
│   └── sample_batch/         # 10 test sheets for batch benchmarking
└── tests/
    └── test_omr_pipeline.py  # Automated test suite
```

---

## Quickstart

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Verify Visual Calibration
To inspect or verify coordinate bounding boxes against `samples/omr-2.jpg`:
```bash
python calibrate.py
```
This generates `debug_grid.jpg` showing the fiducial marker targets, roll number bounding boxes, and 50 question bubble circles with jitter windows.

### 3. Run Offline Batch CLI
Process an entire folder with answer key scoring:
```bash
python batch_cli.py --input ./samples/sample_batch --key ./samples/answer_key.json --output ./results
```
Output:
- `results/omr_results_<job_id>.json`
- `results/omr_results_<job_id>.csv`

### 4. Launch Web Dashboard
Run via command line or double-click `start.bat`:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## OMR Sheet Template Specifications

| Section | Coordinates / Specification |
|---|---|
| **Canvas Dimensions** | Standardized `1200 x 1600` (WxH) |
| **Corner Fiducials** | 4 solid black squares (`60x60` px) at centers: `[80, 80]`, `[1120, 80]`, `[80, 1520]`, `[1120, 1520]` |
| **Roll Number** | 6 vertical columns (`D1`–`D6`), 10 bubbles (`0`–`9`), Col X: `[380, 460, 540, 620, 700, 780]`, Start Y: `340`, Step Y: `28` |
| **Answers (Q1–Q25)** | Left block: Q label at X=`150`, Option circles A, B, C, D at X: `[230, 290, 350, 410]`, Start Y: `740`, Step Y: `30` |
| **Answers (Q26–Q50)** | Right block: Q label at X=`740`, Option circles A, B, C, D at X: `[820, 880, 940, 1000]`, Start Y: `740`, Step Y: `30` |
| **Bubble Radius** | Radius = `12` px |

---

## JSON Output Schema (Per Sheet)

```json
{
  "filename": "omr-2.jpg",
  "status": "COMPLETED",
  "confidence": 0.98,
  "roll_number": "053826",
  "summary": {
    "total_questions": 50,
    "attempted": 5,
    "unattempted": 45,
    "multiple_marked": 0
  },
  "answers": {
    "1": "A",
    "2": null,
    "3": "C",
    "4": null
  },
  "review_reasons": [],
  "score_details": {
    "total_score": 20.0,
    "max_score": 200.0,
    "percentage": 10.0,
    "correct_count": 5,
    "incorrect_count": 0,
    "unattempted_count": 45,
    "multiple_count": 0
  },
  "debug_image_path": "d:\\extractor\\debug_images\\debug_omr-2.jpg"
}
```

---

## Running the Automated Test Suite

```bash
python -m unittest tests/test_omr_pipeline.py
```
Verifies alignment geometry, JSON schema compliance on `omr-2.jpg`, black pen recognition, scoring logic, SQLite persistence, and multi-core batch execution.


---

## 📡 API Documentation

### 1. OMR Sheet Scanner API (`POST /api/extract`)

Extracts Roll Number, Exam Details, and Question Answers. Returns structured JSON **plus the annotated OMR sheet** both as URL and as a Base64 data URI (`data:image/jpeg;base64,...`).

#### A. Multipart File Upload:
```bash
curl -X POST https://extractor.rusoft.in/api/extract \
  -F "file=@sample_omr.jpg"
```

#### B. Base64 JSON Request (`POST /api/omr/extract-base64`):
```bash
curl -X POST https://extractor.rusoft.in/api/omr/extract-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...", "filename": "omr.jpg"}'
```

#### Response:
```json
{
  "success": true,
  "filename": "sample_omr.jpg",
  "status": "COMPLETED",
  "confidence": 0.98,
  "organization": "RUSOFT EDUCATION POINT",
  "exam_name": "SSC-SAMPLE-TEST",
  "exam_code": "EXAM-TOWFRA",
  "roll_number": "102580",
  "summary": {
    "total_questions": 50,
    "attempted": 13,
    "unattempted": 37,
    "multiple_marked": 0
  },
  "answers": {
    "1": "C",
    "2": null,
    "3": "B",
    "50": null
  },
  "annotated_image_url": "/debug_images/annotated_sample_omr.jpg",
  "annotated_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

---

### 2. Intelligent Aadhaar Card Extractor API (`POST /api/aadhaar/extract`)

Intelligently extracts Name (English & Hindi), DOB, YOB, Gender, 12-digit Aadhaar Number, VID, Care of (Father/Husband), Full Address, District, State, and PIN code.

#### A. Dual-Side Upload (Front + Back):
```bash
curl -X POST https://extractor.rusoft.in/api/aadhaar/extract \
  -F "front=@aadhaar_front.jpg" \
  -F "back=@aadhaar_back.jpg"
```

#### B. Single Card Upload (Front, Back, or Combined):
```bash
curl -X POST https://extractor.rusoft.in/api/aadhaar/extract \
  -F "file=@aadhaar_card.jpg"
```

#### C. Base64 JSON Request (`POST /api/aadhaar/extract-base64`):
```bash
curl -X POST https://extractor.rusoft.in/api/aadhaar/extract-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "front": "data:image/jpeg;base64,...",
    "back": "data:image/jpeg;base64,..."
  }'
```

#### Response:
```json
{
  "success": true,
  "card_side": "BOTH",
  "confidence": 0.94,
  "data": {
    "name": "Tanuja Sharma",
    "name_hindi": null,
    "dob": "23/12/2011",
    "gender": "FEMALE",
    "aadhaar_number": "8215 6872 1541",
    "address": "A-908, Manglam Aadhar, Gandhi Path West, Jaipur, PO: Vaishali Nagar, Jaipur, Rajasthan, 302021",
    "pincode": "302021"
  },
  "front_status": true,
  "back_status": true
}
```
