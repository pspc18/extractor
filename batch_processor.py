"""
High-Volume Parallel OMR Batch Processing Engine.
Utilizes Python's ProcessPoolExecutor to distribute sheet scanning across
all available CPU cores for maximum throughput (>3,000 sheets/hour).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2

from config import (
    DEBUG_DIR,
    DEFAULT_NEGATIVE_MARK,
    DEFAULT_POSITIVE_MARK,
)
from database import create_job, save_sheet_record
from omr_engine import process_omr_sheet


def process_single_file_worker(
    task_args: Tuple[str, str, Optional[Dict[str, str]], float, float, bool, str]
) -> Dict[str, Any]:
    """
    Top-level picklable worker function executed inside process pool.
    Loads image, executes computer vision pipeline, and returns serializable dict.
    """
    file_path_str, filename, answer_key, pos_mark, neg_mark, save_debug, debug_dir_str = task_args
    file_path = Path(file_path_str)
    debug_dir = Path(debug_dir_str)

    try:
        img = cv2.imread(str(file_path))
        if img is None:
            return {
                "filename": filename,
                "status": "FAILED",
                "confidence": 0.0,
                "roll_number": "UNKNOWN",
                "summary": {"total_questions": 50, "attempted": 0, "unattempted": 50, "multiple_marked": 0},
                "answers": {str(i): None for i in range(1, 51)},
                "review_reasons": ["Corrupt or unreadable image file"],
                "score_details": None,
                "debug_image_path": None
            }

        res = process_omr_sheet(
            image=img,
            filename=filename,
            answer_key=answer_key,
            save_debug_image=save_debug,
            debug_dir=debug_dir,
            positive_mark=pos_mark,
            negative_mark=neg_mark
        )
        return asdict(res)

    except Exception as e:
        return {
            "filename": filename,
            "status": "FAILED",
            "confidence": 0.0,
            "roll_number": "UNKNOWN",
            "summary": {"total_questions": 50, "attempted": 0, "unattempted": 50, "multiple_marked": 0},
            "answers": {str(i): None for i in range(1, 51)},
            "review_reasons": [f"Processing exception: {str(e)}"],
            "score_details": None,
            "debug_image_path": None
        }


class BatchProcessor:
    """Manages parallel multi-core batch processing of OMR sheets."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4))

    def run_batch(
        self,
        job_id: str,
        files: List[Tuple[Path, str]],  # (absolute_file_path, original_filename)
        answer_key: Optional[Dict[str, str]] = None,
        positive_mark: float = DEFAULT_POSITIVE_MARK,
        negative_mark: float = DEFAULT_NEGATIVE_MARK,
        save_debug: bool = True,
        on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel processing of a batch of files.
        Logs to SQLite as each sheet completes and returns the full results list.
        """
        total = len(files)
        create_job(
            job_id=job_id,
            total_sheets=total,
            answer_key=answer_key,
            positive_mark=positive_mark,
            negative_mark=negative_mark
        )

        tasks = [
            (str(fpath), fname, answer_key, positive_mark, negative_mark, save_debug, str(DEBUG_DIR))
            for fpath, fname in files
        ]

        results = []
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file_worker, task): task[1]
                for task in tasks
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    res_dict = future.result()
                except Exception as exc:
                    res_dict = {
                        "filename": filename,
                        "status": "FAILED",
                        "confidence": 0.0,
                        "roll_number": "UNKNOWN",
                        "summary": {"total_questions": 50, "attempted": 0, "unattempted": 50, "multiple_marked": 0},
                        "answers": {str(i): None for i in range(1, 51)},
                        "review_reasons": [f"Worker error: {str(exc)}"],
                        "score_details": None,
                        "debug_image_path": None
                    }

                # Save record to SQLite
                save_sheet_record(job_id, res_dict)
                results.append(res_dict)
                completed_count += 1

                if on_progress:
                    on_progress(completed_count, total, res_dict)

        return results
