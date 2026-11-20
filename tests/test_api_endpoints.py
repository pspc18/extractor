"""
Integration test for FastAPI endpoints over HTTP.
"""

import json
from pathlib import Path
import urllib.request
import uuid

BASE_URL = "http://127.0.0.1:8000"


def test_http_api():
    # 1. Health
    req = urllib.request.urlopen(f"{BASE_URL}/health")
    data = json.loads(req.read().decode())
    assert data["status"] == "healthy"
    print("[PASS] GET /health")

    # 2. Stats
    req = urllib.request.urlopen(f"{BASE_URL}/api/stats")
    stats = json.loads(req.read().decode())
    assert "total_sheets" in stats
    print(f"[PASS] GET /api/stats: {stats}")

    # 3. Multipart Upload for /api/scan-single
    img_path = Path("samples/omr-2.jpg")
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="image"; filename="omr-2.jpg"\r\n')
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(img_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    upload_req = urllib.request.Request(
        f"{BASE_URL}/api/scan-single",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    resp = urllib.request.urlopen(upload_req)
    res_json = json.loads(resp.read().decode())

    print(f"[PASS] POST /api/scan-single:")
    print(f"       Filename: {res_json['filename']}")
    print(f"       Status: {res_json['status']}")
    print(f"       Roll Number: {res_json['roll_number']}")
    print(f"       Attempted: {res_json['summary']['attempted']}")
    print(f"       Q1: {res_json['answers']['1']}, Q2: {res_json['answers']['2']}")

    assert res_json["status"] == "COMPLETED"
    assert res_json["roll_number"] == "053826"
    assert res_json["summary"]["attempted"] == 5
    assert res_json["answers"]["1"] == "A"
    assert res_json["answers"]["2"] is None
    print("[ALL HTTP ENDPOINT TESTS PASSED]")


if __name__ == "__main__":
    test_http_api()
