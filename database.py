"""
SQLite Database Layer for High-Volume OMR Scanner.
Tracks batch processing jobs, individual sheets, extracted answers, review statuses, and scores.
"""

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from config import DB_PATH


def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes SQLite database schema and indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                total_sheets INTEGER DEFAULT 0,
                completed_count INTEGER DEFAULT 0,
                needs_review_count INTEGER DEFAULT 0,
                failed_count INTEGER DEFAULT 0,
                answer_key_json TEXT,
                positive_mark REAL DEFAULT 4.0,
                negative_mark REAL DEFAULT -1.0
            );
        """)

        # Sheets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sheets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                roll_number TEXT DEFAULT 'UNKNOWN',
                summary_json TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                review_reasons_json TEXT DEFAULT '[]',
                score_details_json TEXT,
                debug_image_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );
        """)

        # Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sheets_job_id ON sheets(job_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sheets_status ON sheets(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sheets_roll ON sheets(roll_number);")
        conn.commit()


@contextmanager
def get_db(db_path: Path = DB_PATH):
    """Context manager for thread-safe SQLite connections."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_job(
    job_id: str,
    total_sheets: int,
    answer_key: Optional[Dict[str, str]] = None,
    positive_mark: float = 4.0,
    negative_mark: float = -1.0
) -> None:
    """Registers a new batch processing job."""
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO jobs (
                id, created_at, status, total_sheets, completed_count,
                needs_review_count, failed_count, answer_key_json, positive_mark, negative_mark
            ) VALUES (?, ?, ?, ?, 0, 0, 0, ?, ?, ?)
        """, (
            job_id, now, "PROCESSING", total_sheets,
            json.dumps(answer_key) if answer_key else None,
            positive_mark, negative_mark
        ))
        conn.commit()


def save_sheet_record(job_id: str, sheet_data: Dict[str, Any]) -> int:
    """Saves a processed sheet result and increments job counts atomically."""
    now = datetime.now().isoformat()
    status = sheet_data.get("status", "COMPLETED")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sheets (
                job_id, filename, status, confidence, roll_number,
                summary_json, answers_json, review_reasons_json,
                score_details_json, debug_image_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            sheet_data.get("filename", "unknown.jpg"),
            status,
            float(sheet_data.get("confidence", 0.0)),
            sheet_data.get("roll_number", "UNKNOWN"),
            json.dumps(sheet_data.get("summary", {})),
            json.dumps(sheet_data.get("answers", {})),
            json.dumps(sheet_data.get("review_reasons", [])),
            json.dumps(sheet_data.get("score_details")) if sheet_data.get("score_details") else None,
            sheet_data.get("debug_image_path"),
            now, now
        ))
        sheet_id = cursor.lastrowid

        # Update job stats
        col_to_increment = "completed_count" if status == "COMPLETED" else (
            "needs_review_count" if status == "NEEDS_REVIEW" else "failed_count"
        )
        cursor.execute(f"""
            UPDATE jobs
            SET {col_to_increment} = {col_to_increment} + 1
            WHERE id = ?
        """, (job_id,))

        # Check if job is completed
        cursor.execute("SELECT total_sheets, completed_count, needs_review_count, failed_count FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            total, comp, rev, fail = row["total_sheets"], row["completed_count"], row["needs_review_count"], row["failed_count"]
            if (comp + rev + fail) >= total and total > 0:
                final_status = "COMPLETED" if rev == 0 and fail == 0 else "PARTIAL_REVIEW"
                cursor.execute("UPDATE jobs SET status = ? WHERE id = ?", (final_status, job_id))

        conn.commit()
        return sheet_id


def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches status and summary of a specific job."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_all_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Lists recent batch jobs."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cursor.fetchall()]


def get_sheets_for_job(job_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves sheets filtered by job_id and/or status."""
    query = "SELECT * FROM sheets WHERE 1=1"
    params: List[Any] = []
    if job_id:
        query += " AND job_id = ?"
        params.append(job_id)
    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY id DESC"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            item = dict(r)
            item["summary"] = json.loads(item["summary_json"])
            item["answers"] = json.loads(item["answers_json"])
            item["review_reasons"] = json.loads(item["review_reasons_json"])
            item["score_details"] = json.loads(item["score_details_json"]) if item["score_details_json"] else None
            results.append(item)
        return results


def update_sheet_review(
    sheet_id: int,
    roll_number: str,
    answers: Dict[str, Optional[str]],
    answer_key: Optional[Dict[str, str]] = None,
    positive_mark: float = 4.0,
    negative_mark: float = -1.0
) -> bool:
    """Allows manual human operator review: approves and recalculates score."""
    from omr_engine import grade_answers

    now = datetime.now().isoformat()
    score_details = None
    if answer_key:
        score_details = grade_answers(answers, answer_key, positive_mark, negative_mark)

    # Recalculate summary
    attempted = sum(1 for a in answers.values() if a in ["A", "B", "C", "D"])
    multiple = sum(1 for a in answers.values() if a == "MULTIPLE")
    unattempted = len(answers) - attempted - multiple

    summary = {
        "total_questions": len(answers),
        "attempted": attempted,
        "unattempted": unattempted,
        "multiple_marked": multiple
    }

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, status FROM sheets WHERE id = ?", (sheet_id,))
        row = cursor.fetchone()
        if not row:
            return False

        old_status = row["status"]
        job_id = row["job_id"]

        cursor.execute("""
            UPDATE sheets
            SET roll_number = ?,
                answers_json = ?,
                summary_json = ?,
                score_details_json = ?,
                status = 'COMPLETED',
                review_reasons_json = '[]',
                updated_at = ?
            WHERE id = ?
        """, (
            roll_number,
            json.dumps(answers),
            json.dumps(summary),
            json.dumps(score_details) if score_details else None,
            now,
            sheet_id
        ))

        # Adjust counts in jobs table if status changed from NEEDS_REVIEW
        if old_status == "NEEDS_REVIEW":
            cursor.execute("""
                UPDATE jobs
                SET needs_review_count = max(0, needs_review_count - 1),
                    completed_count = completed_count + 1
                WHERE id = ?
            """, (job_id,))

        conn.commit()
        return True


def get_global_stats() -> Dict[str, Any]:
    """Retrieves high-level summary stats for the dashboard."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total_jobs FROM jobs")
        jobs_count = cursor.fetchone()["total_jobs"]

        cursor.execute("""
            SELECT 
                COUNT(*) as total_sheets,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'NEEDS_REVIEW' THEN 1 ELSE 0 END) as needs_review,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                AVG(confidence) as avg_confidence
            FROM sheets
        """)
        row = cursor.fetchone()

        return {
            "total_jobs": jobs_count,
            "total_sheets": row["total_sheets"] or 0,
            "completed": row["completed"] or 0,
            "needs_review": row["needs_review"] or 0,
            "failed": row["failed"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0.0, 3)
        }


# Automatically create tables on import
init_db()
