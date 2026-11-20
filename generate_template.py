"""
Generates standard 1200x1600 OMR blank template and realistic sample sheets.
Creates samples/omr-2.jpg with realistic marks (Roll No: 053826, 50 questions)
and synthetic variations for mobile photos and test batches.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    ANSWERS_ROW_STEP_Y,
    ANSWERS_START_Y,
    BUBBLE_RADIUS,
    LEFT_BLOCK_OPTIONS_X,
    LEFT_BLOCK_Q_LABEL_X,
    MARKER_SIZE,
    MARKER_TARGET_CENTERS,
    OPTIONS,
    RIGHT_BLOCK_OPTIONS_X,
    RIGHT_BLOCK_Q_LABEL_X,
    ROLL_COL_X,
    ROLL_COLUMNS,
    ROLL_DIGITS,
    ROLL_ROW_STEP_Y,
    ROLL_START_Y,
    SAMPLES_DIR,
    SHEET_HEIGHT,
    SHEET_WIDTH,
    TOTAL_QUESTIONS,
)


def create_blank_sheet() -> np.ndarray:
    """Generate the standardized 1200x1600 master OMR sheet canvas."""
    canvas = np.full((SHEET_HEIGHT, SHEET_WIDTH, 3), 255, dtype=np.uint8)

    # 1. Draw 4 Solid Black Fiducial Corner Squares (isolated with clean white margins)
    for center_x, center_y in MARKER_TARGET_CENTERS:
        cx, cy = int(center_x), int(center_y)
        half = MARKER_SIZE // 2
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), (0, 0, 0), -1)

    # 2. Header and Title Area
    cv2.putText(canvas, "OMR EVALUATION SHEET - STANDARDIZED TEMPLATE", (240, 95),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Instructions: Fill bubbles completely with Blue or Black ballpoint pen. Do not fold sheet.",
                (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)

    # Candidate Name & Exam info boxes
    cv2.rectangle(canvas, (130, 160), (600, 240), (120, 120, 120), 1)
    cv2.putText(canvas, "CANDIDATE NAME:", (145, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.line(canvas, (145, 220), (580, 220), (190, 190, 190), 1)

    cv2.rectangle(canvas, (620, 160), (1070, 240), (120, 120, 120), 1)
    cv2.putText(canvas, "EXAM CENTER / DATE:", (635, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.line(canvas, (635, 220), (1050, 220), (190, 190, 190), 1)

    # 3. Roll Number Section (Upper-Middle)
    cv2.rectangle(canvas, (320, 260), (880, 650), (40, 40, 40), 2)
    cv2.rectangle(canvas, (320, 260), (880, 295), (230, 230, 230), -1)
    cv2.putText(canvas, "ROLL NUMBER (6 DIGITS)", (480, 285),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 10), 1, cv2.LINE_AA)

    # Column labels (Digit 1 to Digit 6)
    for col_idx, col_x in enumerate(ROLL_COL_X):
        cv2.putText(canvas, f"D{col_idx + 1}", (col_x - 12, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        # Vertical divider guide lines
        if col_idx > 0:
            divider_x = (ROLL_COL_X[col_idx - 1] + col_x) // 2
            cv2.line(canvas, (divider_x, 300), (divider_x, 640), (220, 220, 220), 1)

        # Draw bubbles 0 to 9
        for digit in range(ROLL_DIGITS):
            bubble_y = ROLL_START_Y + digit * ROLL_ROW_STEP_Y
            cv2.circle(canvas, (col_x, bubble_y), BUBBLE_RADIUS, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(digit), (col_x - 4, bubble_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    # Section separator
    cv2.line(canvas, (125, 675), (1075, 675), (50, 50, 50), 2)
    cv2.putText(canvas, "ANSWERS SECTION (50 QUESTIONS)", (420, 705),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2, cv2.LINE_AA)

    # 4. Answers Section - Left Block (Q1 to Q25)
    cv2.rectangle(canvas, (125, 725), (475, 1490), (80, 80, 80), 1)
    # Header row for Left Block
    cv2.rectangle(canvas, (125, 725), (475, 755), (240, 240, 240), -1)
    cv2.putText(canvas, "Q#", (LEFT_BLOCK_Q_LABEL_X - 15, 746), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    for opt_idx, opt_char in enumerate(OPTIONS):
        cv2.putText(canvas, opt_char, (LEFT_BLOCK_OPTIONS_X[opt_idx] - 6, 746),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # 5. Answers Section - Right Block (Q26 to Q50)
    cv2.rectangle(canvas, (700, 725), (1050, 1490), (80, 80, 80), 1)
    # Header row for Right Block
    cv2.rectangle(canvas, (700, 725), (1050, 755), (240, 240, 240), -1)
    cv2.putText(canvas, "Q#", (RIGHT_BLOCK_Q_LABEL_X - 15, 746), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    for opt_idx, opt_char in enumerate(OPTIONS):
        cv2.putText(canvas, opt_char, (RIGHT_BLOCK_OPTIONS_X[opt_idx] - 6, 746),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Draw Question rows
    for q_idx in range(TOTAL_QUESTIONS):
        if q_idx < 25:
            # Left block
            row = q_idx
            q_num = q_idx + 1
            lbl_x = LEFT_BLOCK_Q_LABEL_X
            opt_xs = LEFT_BLOCK_OPTIONS_X
        else:
            # Right block
            row = q_idx - 25
            q_num = q_idx + 1
            lbl_x = RIGHT_BLOCK_Q_LABEL_X
            opt_xs = RIGHT_BLOCK_OPTIONS_X

        qy = ANSWERS_START_Y + 30 + row * ANSWERS_ROW_STEP_Y

        # Subtle row zebra or line
        if row % 2 == 0:
            box_x = 111 if q_idx < 25 else 701
            cv2.rectangle(canvas, (box_x, qy - 12), (box_x + 358, qy + 14), (250, 250, 252), -1)

        cv2.putText(canvas, f"{q_num:02d}", (lbl_x - 18, qy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)

        for opt_idx, ox in enumerate(opt_xs):
            cv2.circle(canvas, (ox, qy), BUBBLE_RADIUS, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(canvas, OPTIONS[opt_idx], (ox - 4, qy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

    # Footer instructions
    cv2.putText(canvas, "Google Antigravity High-Volume OMR Scanner | Self-Hosted Production Engine",
                (360, 1565), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    return canvas


def simulate_bubble_fill(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    ink_color: str = "blue",
    fill_degree: float = 0.95,
    imperfect: bool = True
) -> None:
    """
    Renders realistic pen marks (blue ballpoint, black ink, imperfect edges, slight center offsets).
    """
    if fill_degree <= 0.05:
        return

    # Random jitter in pen placement
    dx = random.randint(-1, 1) if imperfect else 0
    dy = random.randint(-1, 1) if imperfect else 0
    cx = center_x + dx
    cy = center_y + dy

    # Determine ink color with realistic texture
    if ink_color == "blue":
        # Blue ballpoint pen: High B, low R, medium G
        base_color = (random.randint(180, 230), random.randint(50, 90), random.randint(10, 45))
    else:
        # Dark black / charcoal ballpoint
        val = random.randint(15, 45)
        base_color = (val, val, val)

    fill_radius = int(round(BUBBLE_RADIUS * min(1.0, fill_degree + 0.08)))

    # Draw solid inner fill
    cv2.circle(image, (cx, cy), fill_radius, base_color, -1, cv2.LINE_AA)

    # Simulate realistic pen stroke texture (overlapping swirls or strokes)
    if imperfect:
        for _ in range(4):
            sx = cx + random.randint(-fill_radius // 2, fill_radius // 2)
            sy = cy + random.randint(-fill_radius // 2, fill_radius // 2)
            stroke_rad = max(2, fill_radius // 2)
            shade_var = random.randint(-15, 15)
            stroke_col = tuple(max(0, min(255, c + shade_var)) for c in base_color)
            cv2.circle(image, (sx, sy), stroke_rad, stroke_col, -1, cv2.LINE_AA)


def generate_sample_sheet(
    roll_number: str = "053826",
    answers: Optional[Dict[int, str]] = None,
    ink_color: str = "blue",
    add_perspective_noise: bool = False
) -> np.ndarray:
    """
    Generates a filled OMR sheet matching specific roll number and answers.
    Can add simulated smartphone tilt, perspective distortion, and shadows.
    """
    sheet = create_blank_sheet()

    # Fill Roll Number
    assert len(roll_number) == ROLL_COLUMNS, "Roll number must have 6 digits"
    for col_idx, char in enumerate(roll_number):
        digit = int(char)
        col_x = ROLL_COL_X[col_idx]
        bubble_y = ROLL_START_Y + digit * ROLL_ROW_STEP_Y
        simulate_bubble_fill(sheet, col_x, bubble_y, ink_color=ink_color, fill_degree=0.92)

    # Fill Answers
    if answers:
        for q_num, ans in answers.items():
            if ans is None or ans not in OPTIONS:
                continue
            opt_idx = OPTIONS.index(ans)
            if q_num <= 25:
                row = q_num - 1
                ox = LEFT_BLOCK_OPTIONS_X[opt_idx]
            else:
                row = q_num - 26
                ox = RIGHT_BLOCK_OPTIONS_X[opt_idx]

            qy = ANSWERS_START_Y + 30 + row * ANSWERS_ROW_STEP_Y
            simulate_bubble_fill(sheet, ox, qy, ink_color=ink_color, fill_degree=0.90)

    # Simulate smartphone photo / physical capture if requested
    if add_perspective_noise:
        h, w = sheet.shape[:2]
        # Realistic slight camera angle / perspective distortion
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([
            [random.randint(25, 45), random.randint(20, 40)],
            [w - random.randint(20, 50), random.randint(15, 35)],
            [random.randint(15, 40), h - random.randint(20, 45)],
            [w - random.randint(25, 55), h - random.randint(20, 50)]
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Pad with desk / background surface
        background = np.full((h + 100, w + 100, 3), (210, 215, 220), dtype=np.uint8)
        warped = cv2.warpPerspective(sheet, matrix, (w + 100, h + 100),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(220, 220, 225))
        # Add slight shadow gradient
        y_coords = np.linspace(0.85, 1.05, h + 100).reshape(-1, 1, 1)
        shaded = np.clip(warped.astype(np.float32) * y_coords, 0, 255).astype(np.uint8)
        return shaded

    return sheet


def generate_all_samples():
    """Generates standard template and sample sheets into samples/ folder."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Blank Template
    blank = create_blank_sheet()
    blank_path = SAMPLES_DIR / "omr_template.png"
    cv2.imwrite(str(blank_path), blank)
    print(f"[OK] Generated blank template: {blank_path}")

    # 2. samples/omr-2.jpg (Roll: 053826, 5 attempted questions as specified in prompt schema)
    # Prompt schema:
    # roll_number: "053826", attempted: 5, unattempted: 45, Q1: A, Q2: null
    sample_answers_omr2 = {
        1: "A",
        2: None,
        3: "C",
        7: "B",
        15: "D",
        28: "A"
    }
    # Note: exactly 5 attempted answers (1, 3, 7, 15, 28)
    omr2_img = generate_sample_sheet(
        roll_number="053826",
        answers=sample_answers_omr2,
        ink_color="blue",
        add_perspective_noise=False
    )
    omr2_path = SAMPLES_DIR / "omr-2.jpg"
    cv2.imwrite(str(omr2_path), omr2_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Generated reference sample: {omr2_path}")

    # 3. samples/omr-mobile-tilt.jpg (Simulating smartphone photo with perspective tilt)
    full_answers = {q: random.choice(["A", "B", "C", "D"]) for q in range(1, 46)}
    tilted_img = generate_sample_sheet(
        roll_number="184920",
        answers=full_answers,
        ink_color="blue",
        add_perspective_noise=True
    )
    tilted_path = SAMPLES_DIR / "omr-mobile-tilt.jpg"
    cv2.imwrite(str(tilted_path), tilted_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[OK] Generated mobile photo sample: {tilted_path}")

    # 4. samples/omr-black-pen.jpg (Black ink ballpoint pen with 48 attempted questions)
    black_answers = {q: random.choice(["A", "B", "C", "D"]) for q in range(1, 49)}
    black_img = generate_sample_sheet(
        roll_number="927401",
        answers=black_answers,
        ink_color="black",
        add_perspective_noise=False
    )
    black_path = SAMPLES_DIR / "omr-black-pen.jpg"
    cv2.imwrite(str(black_path), black_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Generated black pen sample: {black_path}")

    # 5. Create a sample batch directory with 10 images for batch processing tests
    batch_dir = SAMPLES_DIR / "sample_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        rand_roll = f"{random.randint(100000, 999999)}"
        rand_answers = {q: random.choice(["A", "B", "C", "D"]) for q in random.sample(range(1, 51), k=random.randint(35, 50))}
        pen = "blue" if i % 2 == 0 else "black"
        noise = (i % 3 == 0)
        img = generate_sample_sheet(rand_roll, rand_answers, ink_color=pen, add_perspective_noise=noise)
        cv2.imwrite(str(batch_dir / f"sheet_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"[OK] Generated 10 test sheets in: {batch_dir}")


if __name__ == "__main__":
    generate_all_samples()
