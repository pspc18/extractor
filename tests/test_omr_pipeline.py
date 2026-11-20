"""
Comprehensive Automated Test Suite for OMR Scanner.
Verifies CV alignment, bubble evaluation, JSON output schemas,
scoring calculations, SQLite persistence, and batch processing.
"""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from batch_processor import BatchProcessor
from config import (
    BASE_DIR,
    SAMPLES_DIR,
    SHEET_HEIGHT,
    SHEET_WIDTH,
    TOTAL_QUESTIONS,
)
from database import (
    create_job,
    get_global_stats,
    get_job_details,
    get_sheets_for_job,
    init_db,
    save_sheet_record,
    update_sheet_review,
)
from omr_engine import (
    align_sheet,
    find_fiducial_markers,
    grade_answers,
    process_omr_sheet,
)


class TestOMRPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        # Verify samples exist
        cls.omr2_path = SAMPLES_DIR / "omr-2.jpg"
        if not cls.omr2_path.exists():
            from generate_template import generate_all_samples
            generate_all_samples()

    def test_01_alignment_geometry(self):
        """Checks fiducial marker identification and 1200x1600 perspective warp."""
        img = cv2.imread(str(self.omr2_path))
        self.assertIsNotNone(img, "Could not load omr-2.jpg")

        corners = find_fiducial_markers(img)
        self.assertEqual(len(corners), 4, "Must detect exactly 4 fiducials")

        aligned, _ = align_sheet(img)
        self.assertEqual(aligned.shape[0], SHEET_HEIGHT)
        self.assertEqual(aligned.shape[1], SHEET_WIDTH)

    def test_02_omr_schema_and_values(self):
        """Verifies exact schema and values for samples/omr-2.jpg."""
        img = cv2.imread(str(self.omr2_path))
        result = process_omr_sheet(img, filename="omr-2.jpg")
        res_dict = asdict(result)

        # Expected prompt specifications:
        # roll_number: "053826", attempted: 5, unattempted: 45, Q1: A, Q2: null
        self.assertEqual(res_dict["filename"], "omr-2.jpg")
        self.assertEqual(res_dict["status"], "COMPLETED")
        self.assertEqual(res_dict["roll_number"], "053826")
        self.assertGreaterEqual(res_dict["confidence"], 0.85)

        summary = res_dict["summary"]
        self.assertEqual(summary["total_questions"], TOTAL_QUESTIONS)
        self.assertEqual(summary["attempted"], 5)
        self.assertEqual(summary["unattempted"], 45)
        self.assertEqual(summary["multiple_marked"], 0)

        answers = res_dict["answers"]
        self.assertEqual(answers["1"], "A")
        self.assertIsNone(answers["2"])

    def test_03_black_pen_detection(self):
        """Checks that dark black ballpoint pen / pencil marks are correctly classified."""
        black_path = SAMPLES_DIR / "omr-black-pen.jpg"
        if black_path.exists():
            img = cv2.imread(str(black_path))
            result = process_omr_sheet(img, filename="omr-black-pen.jpg")
            self.assertEqual(result.roll_number, "927401")
            self.assertGreaterEqual(result.summary["attempted"], 40)

    def test_04_scoring_engine(self):
        """Checks positive and negative marking calculation."""
        user_answers = {
            "1": "A",   # Correct (+4)
            "2": "C",   # Incorrect (-1)
            "3": None,  # Unattempted (0)
            "4": "MULTIPLE"  # Multiple/Wrong (-1)
        }
        answer_key = {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D"
        }
        score = grade_answers(user_answers, answer_key, positive_mark=4.0, negative_mark=-1.0)
        self.assertEqual(score["correct_count"], 1)
        # Q2 is incorrect, Q4 is multiple (-1 each) -> 4 - 2 = 2.0
        # Q5-50 are not in user_answers so unattempted
        self.assertAlmostEqual(score["total_score"], 2.0)

    def test_05_database_and_review_workflow(self):
        """Tests job creation, sheet saving, and manual operator approval."""
        import uuid
        test_job_id = f"test_job_{uuid.uuid4().hex[:6]}"
        create_job(test_job_id, total_sheets=1)

        sheet_payload = {
            "filename": "flagged_sheet.jpg",
            "status": "NEEDS_REVIEW",
            "confidence": 0.65,
            "roll_number": "05?826",
            "summary": {"total_questions": 50, "attempted": 4, "unattempted": 46, "multiple_marked": 0},
            "answers": {"1": "A", "2": None, "3": "B"},
            "review_reasons": ["Roll number Digit 3 is ambiguous"]
        }
        sheet_id = save_sheet_record(test_job_id, sheet_payload)
        self.assertGreater(sheet_id, 0)

        # Check in review queue
        sheets = get_sheets_for_job(job_id=test_job_id, status="NEEDS_REVIEW")
        self.assertEqual(len(sheets), 1)

        # Approve and fix
        fixed_answers = {"1": "A", "2": "B", "3": "B"}
        success = update_sheet_review(sheet_id, roll_number="053826", answers=fixed_answers)
        self.assertTrue(success)

        # Verify status is now COMPLETED
        updated_sheets = get_sheets_for_job(job_id=test_job_id)
        self.assertEqual(updated_sheets[0]["status"], "COMPLETED")
        self.assertEqual(updated_sheets[0]["roll_number"], "053826")

    def test_06_batch_processor_parallel(self):
        """Verifies multi-core ProcessPoolExecutor batch processing."""
        import uuid
        processor = BatchProcessor(max_workers=2)
        batch_files = [(self.omr2_path, "omr-2.jpg")]
        job_id = f"test_batch_{uuid.uuid4().hex[:6]}"

        results = processor.run_batch(job_id=job_id, files=batch_files)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "COMPLETED")
        self.assertEqual(results[0]["roll_number"], "053826")


if __name__ == "__main__":
    unittest.main()
