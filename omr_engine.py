"""
High-Performance OMR Engine using OpenCV and NumPy.
Handles corner fiducial detection, 4-point perspective warp,
blue/black ballpoint ink evaluation, edge cases, grading, and debug image generation.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    AMBIGUOUS_MIN,
    ANSWERS_ROW_STEP_Y,
    ANSWERS_START_Y,
    BUBBLE_RADIUS,
    DEBUG_DIR,
    DEFAULT_NEGATIVE_MARK,
    DEFAULT_POSITIVE_MARK,
    EMPTY_THRESHOLD,
    FILLED_THRESHOLD,
    LEFT_BLOCK_OPTIONS_X,
    LEFT_BLOCK_Q_LABEL_X,
    MARKER_SIZE,
    MARKER_TARGET_CENTERS,
    MIN_CONFIDENCE_DIFF,
    OPTIONS,
    RIGHT_BLOCK_OPTIONS_X,
    RIGHT_BLOCK_Q_LABEL_X,
    ROLL_COL_X,
    ROLL_COLUMNS,
    ROLL_DIGITS,
    ROLL_ROW_STEP_Y,
    ROLL_START_Y,
    SEARCH_JITTER,
    SHEET_HEIGHT,
    SHEET_WIDTH,
    TOTAL_QUESTIONS,
)


@dataclass
class QuestionEvaluation:
    question: int
    answer: Optional[str]      # "A", "B", "C", "D", "MULTIPLE", or None
    status: str                # "DETECTED", "BLANK", "MULTIPLE", "AMBIGUOUS"
    confidence: float
    scores: Dict[str, float]
    selected_option_idx: Optional[int] = None


@dataclass
class RollEvaluation:
    roll_number: str
    is_valid: bool
    status: str
    column_digits: List[Optional[str]]
    confidence: float
    reasons: List[str]


@dataclass
class SheetResult:
    filename: str
    status: str                # "COMPLETED", "NEEDS_REVIEW", "FAILED"
    confidence: float
    roll_number: str
    summary: Dict[str, int]
    answers: Dict[str, Optional[str]]
    review_reasons: List[str]
    score_details: Optional[Dict[str, Any]] = None
    debug_image_path: Optional[str] = None


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into OpenCV BGR numpy array."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Format not supported or corrupt data.")
    return image


def find_fiducial_markers(image: np.ndarray) -> np.ndarray:
    """
    Locates the 4 solid black corner square markers on the sheet.
    Robust to tilt, rotation, illumination variations, and desk backgrounds.
    Returns sorted 4x2 array of corner centers: [TL, TR, BL, BR].
    """
    h_orig, w_orig = image.shape[:2]
    # Resize temporarily for fast and consistent contour detection
    scale = min(1.0, 1600.0 / max(h_orig, w_orig))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu binarization + adaptive threshold fallback
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 8)
    # Combine both to handle uneven lighting and shadows
    binary = cv2.bitwise_or(binary_otsu, binary_adapt)

    # Morphological closing to seal square interiors
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_curr, w_curr = gray.shape
    total_area = h_curr * w_curr

    candidates: List[Tuple[float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Expected fiducial marker area is around (60*scale)^2 (~3600 on 1200x1600 canvas)
        if not (0.0006 * total_area <= area <= 0.035 * total_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / max(1, h)
        if not (0.65 <= aspect_ratio <= 1.50):
            continue

        extent = area / float(w * h)
        if extent < 0.65:
            continue

        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"]) / scale
            cy = (M["m01"] / M["m00"]) / scale
            candidates.append((cx, cy, area))

    if len(candidates) < 4:
        raise ValueError(f"Found only {len(candidates)} marker candidates. At least 4 required.")

    # Partition candidates into 4 quadrants relative to image center
    mid_x = w_orig / 2.0
    mid_y = h_orig / 2.0

    quadrants = {
        "TL": [c for c in candidates if c[0] < mid_x and c[1] < mid_y],
        "TR": [c for c in candidates if c[0] >= mid_x and c[1] < mid_y],
        "BL": [c for c in candidates if c[0] < mid_x and c[1] >= mid_y],
        "BR": [c for c in candidates if c[0] >= mid_x and c[1] >= mid_y],
    }

    selected = []
    # Anchors: exact image boundaries
    anchors = {
        "TL": (0.0, 0.0),
        "TR": (float(w_orig), 0.0),
        "BL": (0.0, float(h_orig)),
        "BR": (float(w_orig), float(h_orig))
    }

    for quad_name in ["TL", "TR", "BL", "BR"]:
        quad_items = quadrants[quad_name]
        if not quad_items:
            raise ValueError(f"Fiducial marker missing in quadrant {quad_name}.")
        ax, ay = anchors[quad_name]
        # Choose candidate closest to the respective corner anchor
        best = min(quad_items, key=lambda it: (it[0] - ax) ** 2 + (it[1] - ay) ** 2)
        selected.append([best[0], best[1]])

    points = np.float32(selected)

    # Geometric consistency check
    w_top = np.linalg.norm(points[1] - points[0])
    w_bot = np.linalg.norm(points[3] - points[2])
    h_left = np.linalg.norm(points[2] - points[0])
    h_right = np.linalg.norm(points[3] - points[1])

    if w_top < w_orig * 0.40 or h_left < h_orig * 0.45:
        raise ValueError("Detected fiducial markers do not span the full sheet.")

    return points


def align_sheet(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies 4-point perspective transform to align the sheet to standardized 1200x1600.
    Returns (aligned_image, detected_corners).
    """
    # If the image is already exactly 1200x1600 with markers in place, test corners first
    h, w = image.shape[:2]
    if (w, h) == (SHEET_WIDTH, SHEET_HEIGHT):
        # Quick check if corners at target locations are black
        tl_check = np.mean(image[60:100, 60:100])
        tr_check = np.mean(image[60:100, 1100:1140])
        if tl_check < 60 and tr_check < 60:
            return image.copy(), MARKER_TARGET_CENTERS.copy()

    detected_pts = find_fiducial_markers(image)
    M = cv2.getPerspectiveTransform(detected_pts, MARKER_TARGET_CENTERS)
    aligned = cv2.warpPerspective(
        image, M, (SHEET_WIDTH, SHEET_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return aligned, detected_pts


def evaluate_bubble_fill(
    image: np.ndarray,
    gray: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int = BUBBLE_RADIUS
) -> Tuple[float, float, float]:
    """
    Evaluates bubble darkness and ink presence.
    Supports both Blue ballpoint ink and Black/Dark ink.
    Returns (combined_score, blue_score, dark_score).
    """
    h, w = gray.shape
    r_inner = radius - 1
    r_outer = radius + 6

    # Coordinate boundaries
    if (center_y - r_outer < 0 or center_y + r_outer >= h or
            center_x - r_outer < 0 or center_x + r_outer >= w):
        return 0.0, 0.0, 0.0

    y_grid, x_grid = np.ogrid[-r_outer:r_outer + 1, -r_outer:r_outer + 1]
    dist_sq = x_grid * x_grid + y_grid * y_grid

    inner_mask = dist_sq <= (r_inner * r_inner)
    annulus_mask = (dist_sq > (radius + 2) ** 2) & (dist_sq <= (r_outer * r_outer))

    patch_bgr = image[center_y - r_outer:center_y + r_outer + 1,
                      center_x - r_outer:center_x + r_outer + 1]
    patch_gray = gray[center_y - r_outer:center_y + r_outer + 1,
                      center_x - r_outer:center_x + r_outer + 1]

    # Background brightness estimation from annulus
    bg_pixels = patch_gray[annulus_mask]
    bg_mean = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 240.0

    # Inner bubble pixels
    inner_gray = patch_gray[inner_mask]
    inner_bgr = patch_bgr[inner_mask]
    if len(inner_gray) == 0:
        return 0.0, 0.0, 0.0

    b = inner_bgr[:, 0].astype(np.float32)
    g = inner_bgr[:, 1].astype(np.float32)
    r = inner_bgr[:, 2].astype(np.float32)

    # 1. Blue Ink Detection
    # Blue ink has significantly higher Blue than Red, and modest Green
    blue_pen = (b - r > 16.0) & (b - g > 8.0) & (inner_gray < 200)
    blue_score = float(np.mean(blue_pen))

    # 2. Black / Dark Ink Detection
    # Dark pixels are significantly darker than local background
    dark_pen = (inner_gray < min(135.0, bg_mean - 35.0))
    dark_score = float(np.mean(dark_pen))

    # Combined robust fill score
    if blue_score >= 0.28:
        combined_score = blue_score * 1.15
    elif dark_score >= 0.35:
        combined_score = dark_score
    else:
        # Subtle or partial mark
        combined_score = max(blue_score, dark_score * 0.9)

    return float(np.clip(combined_score, 0.0, 1.0)), blue_score, dark_score


def sample_with_jitter(
    image: np.ndarray,
    gray: np.ndarray,
    center_x: int,
    center_y: int,
    jitter: int = SEARCH_JITTER
) -> float:
    """Samples bubble score with jitter window to account for small paper distortions."""
    best_score = 0.0
    for dy in range(-jitter, jitter + 1, 1):
        for dx in range(-jitter, jitter + 1, 1):
            score, _, _ = evaluate_bubble_fill(image, gray, center_x + dx, center_y + dy)
            if score > best_score:
                best_score = score
    return best_score


def evaluate_question(
    image: np.ndarray,
    gray: np.ndarray,
    question_num: int,
    opt_centers_x: List[int],
    center_y: int
) -> QuestionEvaluation:
    """Evaluates the 4 option bubbles (A, B, C, D) for a single question."""
    scores: Dict[str, float] = {}
    score_list: List[float] = []

    for idx, opt_char in enumerate(OPTIONS):
        cx = opt_centers_x[idx]
        sc = sample_with_jitter(image, gray, cx, center_y)
        scores[opt_char] = round(sc, 4)
        score_list.append(sc)

    sorted_indices = np.argsort(score_list)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    top_score = score_list[top_idx]
    second_score = score_list[second_idx]
    diff = top_score - second_score

    # 1. Unattempted (Blank)
    if top_score < EMPTY_THRESHOLD:
        return QuestionEvaluation(
            question=question_num,
            answer=None,
            status="BLANK",
            confidence=round(1.0 - top_score, 4),
            scores=scores,
            selected_option_idx=None
        )

    # 2. Multiple Marked
    if (top_score >= FILLED_THRESHOLD and second_score >= FILLED_THRESHOLD - 0.05 and
            diff < MIN_CONFIDENCE_DIFF):
        return QuestionEvaluation(
            question=question_num,
            answer="MULTIPLE",
            status="MULTIPLE",
            confidence=round(diff, 4),
            scores=scores,
            selected_option_idx=None
        )

    # 3. Ambiguous / Half-filled
    if AMBIGUOUS_MIN <= top_score < FILLED_THRESHOLD:
        return QuestionEvaluation(
            question=question_num,
            answer=None,
            status="AMBIGUOUS",
            confidence=round(top_score, 4),
            scores=scores,
            selected_option_idx=top_idx
        )

    # 4. Clear Detection
    if top_score >= FILLED_THRESHOLD and diff >= MIN_CONFIDENCE_DIFF:
        return QuestionEvaluation(
            question=question_num,
            answer=OPTIONS[top_idx],
            status="DETECTED",
            confidence=round(min(1.0, top_score + diff * 0.4), 4),
            scores=scores,
            selected_option_idx=top_idx
        )

    # Fallback to Ambiguous if confidence margin is narrow
    return QuestionEvaluation(
        question=question_num,
        answer=None,
        status="AMBIGUOUS",
        confidence=round(diff, 4),
        scores=scores,
        selected_option_idx=top_idx
    )


def evaluate_roll_number(
    image: np.ndarray,
    gray: np.ndarray
) -> RollEvaluation:
    """Evaluates the 6-digit roll number section."""
    detected_digits: List[Optional[str]] = []
    column_confidences: List[float] = []
    reasons: List[str] = []

    for col_idx, col_x in enumerate(ROLL_COL_X):
        scores = []
        for digit in range(ROLL_DIGITS):
            by = ROLL_START_Y + digit * ROLL_ROW_STEP_Y
            sc = sample_with_jitter(image, gray, col_x, by)
            scores.append(sc)

        order = np.argsort(scores)[::-1]
        top_digit = int(order[0])
        second_digit = int(order[1])
        top_sc = scores[top_digit]
        sec_sc = scores[second_digit]
        diff = top_sc - sec_sc

        if top_sc < EMPTY_THRESHOLD:
            detected_digits.append("?")
            column_confidences.append(0.0)
            reasons.append(f"Roll number Digit {col_idx + 1} is empty")
        elif top_sc >= FILLED_THRESHOLD and sec_sc >= FILLED_THRESHOLD - 0.05 and diff < MIN_CONFIDENCE_DIFF:
            detected_digits.append("X")
            column_confidences.append(0.2)
            reasons.append(f"Roll number Digit {col_idx + 1} has multiple bubbles filled")
        elif AMBIGUOUS_MIN <= top_sc < FILLED_THRESHOLD:
            detected_digits.append(str(top_digit))
            column_confidences.append(round(top_sc, 3))
            reasons.append(f"Roll number Digit {col_idx + 1} is ambiguous (fill: {top_sc:.2f})")
        else:
            detected_digits.append(str(top_digit))
            column_confidences.append(round(min(1.0, top_sc + diff * 0.4), 4))

    roll_str = "".join(str(d) for d in detected_digits)
    is_valid = ("?" not in roll_str and "X" not in roll_str and len(reasons) == 0)
    avg_conf = float(np.mean(column_confidences)) if column_confidences else 0.0

    status = "VALID" if is_valid else "INVALID"
    return RollEvaluation(
        roll_number=roll_str,
        is_valid=is_valid,
        status=status,
        column_digits=detected_digits,
        confidence=round(avg_conf, 4),
        reasons=reasons
    )


def grade_answers(
    user_answers: Dict[str, Optional[str]],
    answer_key: Dict[str, str],
    positive_mark: float = DEFAULT_POSITIVE_MARK,
    negative_mark: float = DEFAULT_NEGATIVE_MARK
) -> Dict[str, Any]:
    """
    Grades user answers against an answer key.
    Calculates correct, incorrect, unattempted, score, and percentage.
    """
    correct_count = 0
    incorrect_count = 0
    unattempted_count = 0
    multiple_count = 0

    per_question_result: Dict[str, Dict[str, Any]] = {}

    for q_idx in range(1, TOTAL_QUESTIONS + 1):
        q_key = str(q_idx)
        user_ans = user_answers.get(q_key)
        correct_ans = answer_key.get(q_key)

        if user_ans is None:
            unattempted_count += 1
            verdict = "UNATTEMPTED"
            awarded = 0.0
        elif user_ans == "MULTIPLE":
            multiple_count += 1
            incorrect_count += 1
            verdict = "MULTIPLE_WRONG"
            awarded = negative_mark
        elif correct_ans and user_ans.upper() == correct_ans.upper():
            correct_count += 1
            verdict = "CORRECT"
            awarded = positive_mark
        else:
            incorrect_count += 1
            verdict = "INCORRECT"
            awarded = negative_mark

        per_question_result[q_key] = {
            "user_answer": user_ans,
            "correct_answer": correct_ans,
            "verdict": verdict,
            "awarded_marks": awarded
        }

    total_score = (correct_count * positive_mark) + (incorrect_count * negative_mark)
    max_score = TOTAL_QUESTIONS * positive_mark
    percentage = round((total_score / max_score) * 100, 2) if max_score > 0 else 0.0

    return {
        "total_score": round(total_score, 2),
        "max_score": max_score,
        "percentage": percentage,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unattempted_count": unattempted_count,
        "multiple_count": multiple_count,
        "details": per_question_result
    }


def generate_annotated_debug_image(
    aligned_image: np.ndarray,
    q_evaluations: List[QuestionEvaluation],
    roll_eval: RollEvaluation,
    sheet_status: str,
    score_details: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Generates a high-resolution annotated image highlighting detected marks,
    multiple marks, and review warnings.
    """
    annotated = aligned_image.copy()

    # 1. Annotate Roll Number
    for col_idx, col_x in enumerate(ROLL_COL_X):
        digit_val = roll_eval.column_digits[col_idx]
        if digit_val and digit_val.isdigit():
            target_digit = int(digit_val)
            by = ROLL_START_Y + target_digit * ROLL_ROW_STEP_Y
            cv2.circle(annotated, (col_x, by), BUBBLE_RADIUS + 3, (0, 200, 0), 2, cv2.LINE_AA)
        elif digit_val in ["?", "X"]:
            # Red box for missing or multiple roll mark
            cv2.rectangle(annotated, (col_x - 18, ROLL_START_Y - 10),
                          (col_x + 18, ROLL_START_Y + 9 * ROLL_ROW_STEP_Y + 10),
                          (0, 0, 255), 2, cv2.LINE_AA)

    # 2. Annotate Questions
    for q_eval in q_evaluations:
        q_idx = q_eval.question - 1
        if q_idx < 25:
            row = q_idx
            opt_xs = LEFT_BLOCK_OPTIONS_X
        else:
            row = q_idx - 25
            opt_xs = RIGHT_BLOCK_OPTIONS_X
        qy = ANSWERS_START_Y + 30 + row * ANSWERS_ROW_STEP_Y

        if q_eval.status == "DETECTED" and q_eval.selected_option_idx is not None:
            cx = opt_xs[q_eval.selected_option_idx]
            # Solid green circle around recognized bubble
            cv2.circle(annotated, (cx, qy), BUBBLE_RADIUS + 3, (0, 210, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, q_eval.answer or "", (opt_xs[-1] + 16, qy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 0), 1, cv2.LINE_AA)

        elif q_eval.status == "MULTIPLE":
            # Red box enclosing the entire multiple-marked question row
            cv2.rectangle(annotated, (opt_xs[0] - 15, qy - 12), (opt_xs[-1] + 15, qy + 12),
                          (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "MULT", (opt_xs[-1] + 16, qy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

        elif q_eval.status == "AMBIGUOUS":
            # Amber/Orange warning box
            cv2.rectangle(annotated, (opt_xs[0] - 15, qy - 12), (opt_xs[-1] + 15, qy + 12),
                          (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "?", (opt_xs[-1] + 16, qy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

    # 3. Header Status Banner
    banner_bg = (30, 30, 30)
    badge_color = (0, 180, 0) if sheet_status == "COMPLETED" else ((0, 140, 255) if sheet_status == "NEEDS_REVIEW" else (0, 0, 230))

    cv2.rectangle(annotated, (0, 0), (SHEET_WIDTH, 45), banner_bg, -1)
    cv2.rectangle(annotated, (15, 8), (170, 37), badge_color, -1)
    cv2.putText(annotated, sheet_status, (25, 29),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    info_text = f"Roll No: {roll_eval.roll_number} | Conf: {roll_eval.confidence:.2f}"
    if score_details:
        info_text += f" | Score: {score_details['total_score']}/{score_details['max_score']} ({score_details['percentage']}%)"
    cv2.putText(annotated, info_text, (190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return annotated


def process_omr_sheet(
    image: np.ndarray,
    filename: str = "sheet.jpg",
    answer_key: Optional[Dict[str, str]] = None,
    save_debug_image: bool = True,
    debug_dir: Path = DEBUG_DIR,
    positive_mark: float = DEFAULT_POSITIVE_MARK,
    negative_mark: float = DEFAULT_NEGATIVE_MARK
) -> SheetResult:
    """
    Main processing pipeline for a single OMR sheet.
    Aligns, extracts Roll Number and 50 Answers, checks edge cases,
    grades if answer key is provided, and saves annotated debug image.
    """
    review_reasons: List[str] = []

    # Step 1: Alignment via Fiducial Markers
    try:
        aligned, _ = align_sheet(image)
    except Exception as e:
        # Failed corner detection -> flag as NEEDS_REVIEW / FAILED
        reason = f"Corner marker detection failed: {str(e)}"
        review_reasons.append(reason)
        # Fallback to direct resize for debug visualization
        aligned = cv2.resize(image, (SHEET_WIDTH, SHEET_HEIGHT))
        debug_path = debug_dir / f"debug_{Path(filename).stem}.jpg" if save_debug_image else None
        if debug_path:
            cv2.putText(aligned, "ALIGNMENT FAILED", (400, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imwrite(str(debug_path), aligned)

        return SheetResult(
            filename=filename,
            status="NEEDS_REVIEW",
            confidence=0.0,
            roll_number="UNKNOWN",
            summary={"total_questions": TOTAL_QUESTIONS, "attempted": 0, "unattempted": TOTAL_QUESTIONS, "multiple_marked": 0},
            answers={str(i): None for i in range(1, TOTAL_QUESTIONS + 1)},
            review_reasons=review_reasons,
            score_details=None,
            debug_image_path=str(debug_path) if debug_path else None
        )

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Step 2: Roll Number Extraction
    roll_eval = evaluate_roll_number(aligned, gray)
    if not roll_eval.is_valid:
        review_reasons.extend(roll_eval.reasons)

    # Step 3: 50 Questions Extraction
    q_evaluations: List[QuestionEvaluation] = []
    answers_dict: Dict[str, Optional[str]] = {}
    attempted_count = 0
    unattempted_count = 0
    multiple_count = 0
    q_confidences: List[float] = []

    for q_idx in range(TOTAL_QUESTIONS):
        q_num = q_idx + 1
        if q_idx < 25:
            row = q_idx
            opt_xs = LEFT_BLOCK_OPTIONS_X
        else:
            row = q_idx - 25
            opt_xs = RIGHT_BLOCK_OPTIONS_X

        qy = ANSWERS_START_Y + 30 + row * ANSWERS_ROW_STEP_Y
        q_eval = evaluate_question(aligned, gray, q_num, opt_xs, qy)
        q_evaluations.append(q_eval)

        answers_dict[str(q_num)] = q_eval.answer
        q_confidences.append(q_eval.confidence)

        if q_eval.status == "DETECTED":
            attempted_count += 1
        elif q_eval.status == "MULTIPLE":
            multiple_count += 1
            review_reasons.append(f"Multiple bubbles marked on Question {q_num}")
        elif q_eval.status == "AMBIGUOUS":
            review_reasons.append(f"Ambiguous bubble mark on Question {q_num}")
        else:
            unattempted_count += 1

    # Step 4: Overall Status & Confidence
    overall_confidence = float(np.mean(q_confidences + [roll_eval.confidence]))
    overall_confidence = round(max(0.1, min(1.0, overall_confidence)), 4)

    status = "COMPLETED"
    if review_reasons:
        status = "NEEDS_REVIEW"

    # Step 5: Optional Scoring
    score_details = None
    if answer_key:
        score_details = grade_answers(
            answers_dict, answer_key,
            positive_mark=positive_mark,
            negative_mark=negative_mark
        )

    # Step 6: Annotated Debug Image
    debug_path = None
    if save_debug_image:
        debug_path = debug_dir / f"debug_{Path(filename).stem}.jpg"
        generate_annotated_debug_image(
            aligned, q_evaluations, roll_eval, status, score_details, debug_path
        )

    return SheetResult(
        filename=filename,
        status=status,
        confidence=overall_confidence,
        roll_number=roll_eval.roll_number,
        summary={
            "total_questions": TOTAL_QUESTIONS,
            "attempted": attempted_count,
            "unattempted": unattempted_count,
            "multiple_marked": multiple_count
        },
        answers=answers_dict,
        review_reasons=review_reasons,
        score_details=score_details,
        debug_image_path=str(debug_path) if debug_path else None
    )
