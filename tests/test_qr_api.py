"""
Integration test for Aadhaar QR API endpoint functions directly
"""

import asyncio
import gzip
from pathlib import Path
import sys
import numpy as np
import zxingcpp

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import AadhaarDecodeQRRequest, decode_aadhaar_qr_api, extract_aadhaar_qr_file
from fastapi import UploadFile
import io


def test_decode_xml_qr_api():
    xml_data = '<PrintLetterBarcodeData uid="123456789012" name="Deepak Joshi" gender="M" dob="10/10/1990" co="S/O Mohan Joshi" house="14" street="Station Road" vtc="Kota" state="Rajasthan" pc="324002"/>'
    req = AadhaarDecodeQRRequest(qr_data=xml_data)
    resp = asyncio.run(decode_aadhaar_qr_api(req))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    assert data["success"] is True
    assert data["extraction_method"] == "QR_CODE"
    assert data["data"]["name"] == "Deepak Joshi"
    assert data["data"]["aadhaar_number"] == "1234 5678 9012"
    assert data["data"]["dob"] == "10/10/1990"
    assert data["data"]["gender"] == "MALE"
    assert data["data"]["pincode"] == "324002"
    print("✅ XML QR API Endpoint Test Passed!")


def test_decode_secure_qr_api():
    fields = [
        b"3", b"5555202102141530", b"Sunita Devi", b"05-04-1988", b"F",
        b"W/O Rajendra Prasad", b"Jaipur", b"Near Post Office", b"C-12",
        b"Jagatpura", b"302017", b"Jagatpura SO", b"Rajasthan", b"Mahal Road",
        b"Sanganer", b"Jaipur"
    ]
    payload = b"\xff".join(fields)
    fake_photo = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    full_data = payload + b"\xff" + fake_photo + (b"\x00" * 256)
    compressed = gzip.compress(full_data)
    big_int = int.from_bytes(compressed, byteorder="big")

    req = AadhaarDecodeQRRequest(qr_data=str(big_int))
    resp = asyncio.run(decode_aadhaar_qr_api(req))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    assert data["success"] is True
    assert data["data"]["name"] == "Sunita Devi"
    assert data["data"]["dob"] == "05/04/1988"
    assert data["data"]["gender"] == "FEMALE"
    assert data["data"]["pincode"] == "302017"
    assert "C-12" in data["data"]["address"]
    assert "Jagatpura" in data["data"]["address"]
    print("✅ Secure QR API Endpoint Test Passed!")


def test_extract_qr_file_api():
    xml_data = '<PrintLetterBarcodeData uid="888877776666" name="Manish Sharma" gender="M" dob="12/12/1993" co="S/O R.K. Sharma" house="Plot 10" street="Sirsi Road" vtc="Jaipur" state="Rajasthan" pc="302012"/>'
    barcode = zxingcpp.create_barcode(xml_data, zxingcpp.BarcodeFormat.QRCode)
    qr_img = np.array(zxingcpp.write_barcode_to_image(barcode, scale=4))

    # Encode as JPEG
    import cv2
    _, img_bytes = cv2.imencode(".jpg", qr_img)

    upload_file = UploadFile(
        file=io.BytesIO(img_bytes.tobytes()),
        filename="qr_card.jpg"
    )

    resp = asyncio.run(extract_aadhaar_qr_file(upload_file))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    assert data["success"] is True
    assert data["data"]["name"] == "Manish Sharma"
    assert data["data"]["aadhaar_number"] == "8888 7777 6666"
    assert data["data"]["dob"] == "12/12/1993"
    print("✅ Extract QR File API Endpoint Test Passed!")


if __name__ == "__main__":
    test_decode_xml_qr_api()
    test_decode_secure_qr_api()
    test_extract_qr_file_api()
    print("\n🎉 ALL FASTAPI AADHAAR QR API TESTS PASSED!")
