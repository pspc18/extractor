"""
Configuration file for High-Volume OMR Scanner.
Defines template dimensions, ROI coordinates, fiducial markers, and classification thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
DEBUG_DIR = BASE_DIR / "debug_images"
DB_PATH = DATA_DIR / "omr_scanner.db"

# Ensure runtime directories exist
for path in [DATA_DIR, SAMPLES_DIR, UPLOADS_DIR, RESULTS_DIR, DEBUG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Standardized Canvas Geometry
SHEET_WIDTH: int = 1200
SHEET_HEIGHT: int = 1600

# 4 Corner Fiducial Markers (solid black squares)
# Center coordinates on standardized 1200x1600 canvas:
# Top-Left, Top-Right, Bottom-Left, Bottom-Right
MARKER_SIZE: int = 60
MARKER_TARGET_CENTERS: np.ndarray = np.float32([
    [80, 80],      # Top-Left
    [1120, 80],    # Top-Right
    [80, 1520],    # Bottom-Left
    [1120, 1520]   # Bottom-Right
])

# Roll Number Section Specifications
# 6 vertical columns, each with 10 bubbles (digits 0 to 9)
ROLL_COLUMNS: int = 6
ROLL_DIGITS: int = 10
ROLL_COL_X: List[int] = [380, 460, 540, 620, 700, 780]
ROLL_START_Y: int = 340
ROLL_ROW_STEP_Y: int = 28

# Answers Section Specifications
# 50 Questions split into 2 side-by-side blocks of 25 questions each
TOTAL_QUESTIONS: int = 50
QUESTIONS_PER_BLOCK: int = 25
OPTIONS: List[str] = ["A", "B", "C", "D"]

ANSWERS_START_Y: int = 740
ANSWERS_ROW_STEP_Y: int = 30

# Left Block: Questions 1 to 25
LEFT_BLOCK_Q_LABEL_X: int = 150
LEFT_BLOCK_OPTIONS_X: List[int] = [230, 290, 350, 410]

# Right Block: Questions 26 to 50
RIGHT_BLOCK_Q_LABEL_X: int = 740
RIGHT_BLOCK_OPTIONS_X: List[int] = [820, 880, 940, 1000]

# Bubble Evaluation Parameters
BUBBLE_RADIUS: int = 12
SEARCH_JITTER: int = 3  # +/- pixels jitter search to absorb slight printing shifts

# Thresholds for classification
# Pixel fill-ratio thresholds:
EMPTY_THRESHOLD: float = 0.20       # Below this is considered unattempted (blank)
FILLED_THRESHOLD: float = 0.42      # Above this is definitively marked
MIN_CONFIDENCE_DIFF: float = 0.16   # Minimum difference between top and second choice
AMBIGUOUS_MIN: float = 0.24         # Between 0.24 and 0.42 is flagged as ambiguous mark

# Scoring default parameters (can be overridden per batch/job)
DEFAULT_POSITIVE_MARK: float = 4.0
DEFAULT_NEGATIVE_MARK: float = -1.0
