"""
Comprehensive Unit Tests for Aadhaar QR Extraction Engine
"""

import gzip
import json
from pathlib import Path
import sys
import numpy as np
import zxingcpp

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aadhaar_qr import (
    decode_aadhaar_qr,
    detect_and_decode_qr_from_image,
    parse_secure_aadhaar_qr,
    parse_xml_aadhaar
)


def test_xml_qr_decoding():
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
    <PrintLetterBarcodeData uid="482910482910" name="Kavita Meena" gender="F" dob="12/06/1998" co="W/O Suresh Meena" house="Plot 45" street="Tonk Road" loc="Mansarovar" vtc="Jaipur" po="Mansarovar SO" dist="Jaipur" state="Rajasthan" pc="302020"/>"""

    res = decode_aadhaar_qr(xml_data)
    assert res is not None
    assert res["name"] == "Kavita Meena"
    assert res["dob"] == "12/06/1998"
    assert res["gender"] == "FEMALE"
    assert res["aadhaar_number"] == "4829 1048 2910"
    assert res["pincode"] == "302020"
    assert "Plot 45" in res["address"]
    assert "Mansarovar" in res["address"]
    assert res["qr_type"] == "UIDAI_XML_QR"
    print("✅ XML QR Decoding Test Passed!")


def test_secure_qr_decoding():
    fields = [
        b"3",                  # 0: flag
        b"7829202102141530",   # 1: ref id (starts with 7829)
        b"Pooja Verma",        # 2: name
        b"25-11-1996",         # 3: dob
        b"F",                  # 4: gender
        b"D/O Mohan Verma",    # 5: care of
        b"Jaipur",             # 6: district
        b"Near KV School",     # 7: landmark
        b"B-201",              # 8: house
        b"Malviya Nagar",      # 9: loc
        b"302017",             # 10: pin
        b"Malviya Nagar SO",   # 11: po
        b"Rajasthan",          # 12: state
        b"Calgiri Marg",       # 13: street
        b"Sanganer",           # 14: subdist
        b"Jaipur"              # 15: vtc
    ]
    payload = b"\xff".join(fields)
    fake_photo = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    full_decompressed = payload + b"\xff" + fake_photo + (b"\x00" * 256)

    compressed = gzip.compress(full_decompressed)
    big_int = int.from_bytes(compressed, byteorder="big")
    int_str = str(big_int)

    # Test decoding string
    res = decode_aadhaar_qr(int_str)
    assert res is not None
    assert res["name"] == "Pooja Verma"
    assert res["dob"] == "25/11/1996"
    assert res["gender"] == "FEMALE"
    assert res["pincode"] == "302017"
    assert "7829" in res["aadhaar_number"]
    assert "B-201" in res["address"]
    assert "Malviya Nagar" in res["address"]
    assert res["photo_base64"] is not None
    assert res["photo_base64"].startswith("data:image/jpeg;base64,")
    assert res["qr_type"] == "UIDAI_SECURE_QR"
    print("✅ Secure QR Big-Int Decoding Test Passed!")


def test_image_qr_detection():
    # Create an image containing a real Aadhaar XML QR code
    xml_data = '<PrintLetterBarcodeData uid="999988887777" name="Amit Singh" gender="M" dob="01/01/1992" co="S/O Balwant Singh" house="H-12" street="Ajmer Road" vtc="Jaipur" state="Rajasthan" pc="302006"/>'
    barcode = zxingcpp.create_barcode(xml_data, zxingcpp.BarcodeFormat.QRCode)
    qr_img = np.array(zxingcpp.write_barcode_to_image(barcode))

    # Place QR inside a larger canvas representing an Aadhaar card back side
    card_canvas = np.full((600, 800), 245, dtype=np.uint8)
    qr_resized = np.array(zxingcpp.write_barcode_to_image(barcode, scale=4))
    rh, rw = qr_resized.shape[:2]
    card_canvas[50 : 50 + rh, 50 : 50 + rw] = qr_resized

    # Detect & decode from card canvas
    res = detect_and_decode_qr_from_image(card_canvas)
    assert res is not None
    assert res["name"] == "Amit Singh"
    assert res["dob"] == "01/01/1992"
    assert res["gender"] == "MALE"
    assert res["aadhaar_number"] == "9999 8888 7777"
    assert res["pincode"] == "302006"
    print("✅ Synthetic Card Image QR Detection Test Passed!")


if __name__ == "__main__":
    test_xml_qr_decoding()
    test_secure_qr_decoding()
    test_image_qr_detection()
    print("\n🎉 ALL AADHAAR QR TESTS PASSED!")
