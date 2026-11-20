import json
from pathlib import Path
import urllib.request
import uuid

def upload_and_test(file_path):
    p = Path(file_path)
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(p.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/extract",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    print(f"[HTTP OK] {data['filename']:<36} | Roll: {data['roll_number']:<8} | Questions: {data['summary']['total_questions']:<3} | Attempted: {data['summary']['attempted']:<3}")

if __name__ == "__main__":
    upload_and_test("samples/omr-2.jpg")
    upload_and_test("samples/omr-mobile-tilt.jpg")
    upload_and_test("samples/omr-black-pen.jpg")
    upload_and_test(r"d:\omr_extractor\sample-omr-filled-50.png")
    upload_and_test(r"d:\omr_extractor\sample-omr-random-10-two-skipped.png")
