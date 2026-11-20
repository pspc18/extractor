"""
High-Speed Offline Batch CLI for OMR Processing.
Processes an entire folder of OMR sheets across multi-core CPUs
and exports final scores and answers to JSON and CSV.

Usage:
    python batch_cli.py --input ./input_folder --output ./results
    python batch_cli.py --input ./samples/sample_batch --key ./samples/answer_key.json
"""

import argparse
import csv
import json
from pathlib import Path
import time
import uuid
from typing import Dict, Optional

from batch_processor import BatchProcessor
from config import DEFAULT_NEGATIVE_MARK, DEFAULT_POSITIVE_MARK, RESULTS_DIR


def run_cli_batch(
    input_dir: Path,
    output_dir: Path,
    key_file: Optional[Path] = None,
    workers: Optional[int] = None,
    positive_mark: float = DEFAULT_POSITIVE_MARK,
    negative_mark: float = DEFAULT_NEGATIVE_MARK
) -> None:
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather image files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [
        (p, p.name) for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    if not files:
        print(f"[WARNING] No valid image files found in {input_dir}")
        return

    # Load Answer Key if provided
    answer_key = None
    if key_file and key_file.exists():
        with open(key_file, "r", encoding="utf-8") as kf:
            answer_key = json.load(kf)
        print(f"[*] Loaded Answer Key from {key_file} ({len(answer_key)} questions)")

    job_id = f"cli_{uuid.uuid4().hex[:8]}"
    processor = BatchProcessor(max_workers=workers)

    print(f"\n=======================================================")
    print(f"  HIGH-VOLUME OMR BATCH SCANNER (OFFLINE CLI)")
    print(f"=======================================================")
    print(f"  Input Directory : {input_dir.resolve()}")
    print(f"  Output Directory: {output_dir.resolve()}")
    print(f"  Total Sheets    : {len(files)}")
    print(f"  Workers (Cores) : {processor.max_workers}")
    print(f"  Job ID          : {job_id}")
    print(f"=======================================================\n")

    start_time = time.time()

    def progress_callback(completed, total, latest):
        status_symbol = "[✓]" if latest["status"] == "COMPLETED" else "[!]"
        roll = latest.get("roll_number", "UNKNOWN")
        score_str = ""
        if latest.get("score_details"):
            score_str = f" | Score: {latest['score_details']['total_score']}"
        print(f"  {status_symbol} [{completed:03d}/{total:03d}] {latest['filename']} | Roll: {roll} | {latest['status']}{score_str}")

    results = processor.run_batch(
        job_id=job_id,
        files=files,
        answer_key=answer_key,
        positive_mark=positive_mark,
        negative_mark=negative_mark,
        save_debug=True,
        on_progress=progress_callback
    )

    elapsed = max(0.001, time.time() - start_time)
    speed = len(files) / elapsed
    daily_capacity = int(speed * 86400)

    # Export results to JSON
    json_path = output_dir / f"omr_results_{job_id}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    # Export results to CSV
    csv_path = output_dir / f"omr_results_{job_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        headers = [
            "Filename", "Status", "Roll Number", "Confidence",
            "Attempted", "Unattempted", "Multiple Marked",
            "Total Score", "Max Score", "Percentage", "Review Reasons"
        ] + [f"Q{i}" for i in range(1, 51)]
        writer.writerow(headers)

        for item in results:
            summary = item.get("summary", {})
            score = item.get("score_details") or {}
            reasons = "; ".join(item.get("review_reasons", []))
            answers = item.get("answers", {})

            row = [
                item.get("filename"),
                item.get("status"),
                item.get("roll_number"),
                item.get("confidence"),
                summary.get("attempted", 0),
                summary.get("unattempted", 0),
                summary.get("multiple_marked", 0),
                score.get("total_score", "N/A"),
                score.get("max_score", "N/A"),
                score.get("percentage", "N/A"),
                reasons
            ]
            for i in range(1, 51):
                ans = answers.get(str(i))
                row.append(ans if ans is not None else "-")
            writer.writerow(row)

    comp_count = sum(1 for r in results if r["status"] == "COMPLETED")
    rev_count = sum(1 for r in results if r["status"] == "NEEDS_REVIEW")
    fail_count = sum(1 for r in results if r["status"] == "FAILED")

    print(f"\n=======================================================")
    print(f"  BATCH COMPLETED IN {elapsed:.2f}s ({speed:.1f} sheets/sec)")
    print(f"  Projected Daily Capacity : {daily_capacity:,} sheets/day")
    print(f"  Summary: {comp_count} Completed | {rev_count} Needs Review | {fail_count} Failed")
    print(f"  JSON Export: {json_path}")
    print(f"  CSV Export : {csv_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Volume OMR Scanner Batch CLI")
    parser.add_argument("--input", "-i", type=str, default="./samples/sample_batch",
                        help="Path to input folder with OMR sheet images")
    parser.add_argument("--output", "-o", type=str, default="./results",
                        help="Path to output folder for results JSON and CSV")
    parser.add_argument("--key", "-k", type=str, default=None,
                        help="Optional path to answer key JSON file")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of CPU worker processes (defaults to all cores)")
    parser.add_argument("--positive", "-p", type=float, default=DEFAULT_POSITIVE_MARK,
                        help="Marks awarded for correct answers (default: 4.0)")
    parser.add_argument("--negative", "-n", type=float, default=DEFAULT_NEGATIVE_MARK,
                        help="Marks deducted for incorrect answers (default: -1.0)")
    args = parser.parse_args()

    run_cli_batch(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        key_file=Path(args.key) if args.key else None,
        workers=args.workers,
        positive_mark=args.positive,
        negative_mark=args.negative
    )
