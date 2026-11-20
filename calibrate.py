"""
Visual Calibration Tool for OMR Grid Overlay.
Draws configured ROI bounding boxes, fiducial markers, roll number grid,
and answer bubble circles on samples/omr-2.jpg and saves debug_grid.jpg.
Allows rapid verification and coordinate tuning in seconds.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from config import (
    ANSWERS_ROW_STEP_Y,
    ANSWERS_START_Y,
    BASE_DIR,
    BUBBLE_RADIUS,
    DEBUG_DIR,
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
    SEARCH_JITTER,
    SHEET_HEIGHT,
    SHEET_WIDTH,
    TOTAL_QUESTIONS,
)


def draw_calibration_grid(
    image: np.ndarray,
    show_jitter_bounds: bool = True
) -> np.ndarray:
    """
    Overlays colored coordinate guide lines, bounding boxes, and bubble targets
    onto an aligned 1200x1600 OMR sheet.
    """
    canvas = image.copy()
    if canvas.shape[:2] != (SHEET_HEIGHT, SHEET_WIDTH):
        canvas = cv2.resize(canvas, (SHEET_WIDTH, SHEET_HEIGHT))

    # Color palette (BGR)
    COLOR_MARKER = (255, 100, 0)      # Bright Blue
    COLOR_ROLL = (0, 165, 255)        # Orange
    COLOR_BUBBLE = (0, 200, 0)        # Green
    COLOR_JITTER = (180, 180, 255)    # Light Pink
    COLOR_GRID_LINE = (230, 230, 230) # Very faint gray

    # 1. Overlay 4 Corner Marker Target Boxes
    marker_labels = ["TL Marker (80,80)", "TR Marker (1120,80)",
                     "BL Marker (80,1520)", "BR Marker (1120,1520)"]
    half_m = MARKER_SIZE // 2
    for idx, (cx, cy) in enumerate(MARKER_TARGET_CENTERS):
        x1, y1 = int(cx - half_m), int(cy - half_m)
        x2, y2 = int(cx + half_m), int(cy + half_m)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_MARKER, 2)
        cv2.drawMarker(canvas, (int(cx), int(cy)), COLOR_MARKER, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(canvas, marker_labels[idx], (x1 - 10, y1 - 8 if cy < 500 else y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MARKER, 1, cv2.LINE_AA)

    # 2. Roll Number Section Overlay
    cv2.rectangle(canvas, (320, 260), (880, 650), COLOR_ROLL, 2)
    cv2.putText(canvas, "ROI: Roll Number (6 Cols x 10 Digits)", (330, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROLL, 1, cv2.LINE_AA)

    for col_idx, col_x in enumerate(ROLL_COL_X):
        cv2.putText(canvas, f"C{col_idx + 1}:{col_x}", (col_x - 22, 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ROLL, 1, cv2.LINE_AA)
        for digit in range(ROLL_DIGITS):
            by = ROLL_START_Y + digit * ROLL_ROW_STEP_Y
            # Target bubble circle
            cv2.circle(canvas, (col_x, by), BUBBLE_RADIUS, COLOR_BUBBLE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (col_x, by), 1, COLOR_BUBBLE, -1)
            if show_jitter_bounds:
                cv2.rectangle(canvas,
                              (col_x - BUBBLE_RADIUS - SEARCH_JITTER, by - BUBBLE_RADIUS - SEARCH_JITTER),
                              (col_x + BUBBLE_RADIUS + SEARCH_JITTER, by + BUBBLE_RADIUS + SEARCH_JITTER),
                              COLOR_JITTER, 1)

    # 3. Answers Section Overlay
    # Left Block (Q1-25)
    cv2.rectangle(canvas, (110, 725), (470, 1500), COLOR_BUBBLE, 2)
    cv2.putText(canvas, "ROI: Left Block (Q1-25)", (115, 718),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BUBBLE, 1, cv2.LINE_AA)

    # Right Block (Q26-50)
    cv2.rectangle(canvas, (700, 725), (1060, 1500), COLOR_BUBBLE, 2)
    cv2.putText(canvas, "ROI: Right Block (Q26-50)", (705, 718),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BUBBLE, 1, cv2.LINE_AA)

    for q_idx in range(TOTAL_QUESTIONS):
        if q_idx < 25:
            row = q_idx
            opt_xs = LEFT_BLOCK_OPTIONS_X
            lbl_x = LEFT_BLOCK_Q_LABEL_X
        else:
            row = q_idx - 25
            opt_xs = RIGHT_BLOCK_OPTIONS_X
            lbl_x = RIGHT_BLOCK_Q_LABEL_X

        qy = ANSWERS_START_Y + 30 + row * ANSWERS_ROW_STEP_Y

        # Row indicator line
        cv2.line(canvas, (lbl_x - 35, qy), (opt_xs[-1] + 25, qy), COLOR_GRID_LINE, 1)

        for opt_idx, ox in enumerate(opt_xs):
            cv2.circle(canvas, (ox, qy), BUBBLE_RADIUS, COLOR_BUBBLE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (ox, qy), 1, COLOR_BUBBLE, -1)
            if show_jitter_bounds:
                cv2.rectangle(canvas,
                              (ox - BUBBLE_RADIUS - SEARCH_JITTER, qy - BUBBLE_RADIUS - SEARCH_JITTER),
                              (ox + BUBBLE_RADIUS + SEARCH_JITTER, qy + BUBBLE_RADIUS + SEARCH_JITTER),
                              COLOR_JITTER, 1)

    # Legend at bottom
    legend_y = 1585
    cv2.rectangle(canvas, (30, legend_y - 20), (1170, legend_y + 10), (240, 240, 240), -1)
    cv2.rectangle(canvas, (30, legend_y - 20), (1170, legend_y + 10), (180, 180, 180), 1)
    cv2.putText(canvas, "LEGEND: Blue Box = Fiducials | Orange = Roll ROI | Green Circles = Bubble Targets | Pink = Jitter Search Window",
                (45, legend_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)

    return canvas


def run_calibration(sample_image_path: Path, output_image_path: Path) -> None:
    """Loads sample image, runs auto-alignment if needed, and draws calibration grid."""
    print(f"[*] Reading image from: {sample_image_path}")
    if not sample_image_path.exists():
        raise FileNotFoundError(f"Input file not found: {sample_image_path}")

    img = cv2.imread(str(sample_image_path))
    if img is None:
        raise ValueError(f"Failed to decode image from: {sample_image_path}")

    # If omr_engine is available, align first; otherwise direct overlay
    try:
        from omr_engine import align_sheet
        aligned, _ = align_sheet(img)
        print("[*] Successfully applied perspective transform using detected fiducial markers.")
    except Exception as e:
        print(f"[!] Warning: Auto-alignment skipped ({e}), resizing directly to 1200x1600.")
        aligned = cv2.resize(img, (SHEET_WIDTH, SHEET_HEIGHT))

    grid = draw_calibration_grid(aligned)
    cv2.imwrite(str(output_image_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SUCCESS] Calibration grid saved to: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Grid Calibration and Visualizer")
    parser.add_argument("--input", type=str, default=str(SAMPLES_DIR / "omr-2.jpg"),
                        help="Path to sample image (default: samples/omr-2.jpg)")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "debug_grid.jpg"),
                        help="Path to output debug grid (default: debug_grid.jpg)")
    args = parser.parse_args()

    run_calibration(Path(args.input), Path(args.output))
