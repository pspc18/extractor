"""
Direct async test for OMR and Aadhaar extraction functions.
"""
import asyncio
import base64
import json
import cv2
import numpy as np

from aadhaar_engine import extract_aadhaar_from_image, extract_aadhaar_dual_side, decode_base64_image
from dynamic_engine import process_dynamic_omr

def test_omr():
    print("\n--- Testing OMR Engine with Base64 Annotation ---")
    img = cv2.imread("samples/omr-2.jpg")
    res = process_dynamic_omr(img, "omr-2.jpg")
    print("OMR Status:", res.get("status"))
    print("Roll Number:", res.get("roll_number"))
    print("Has annotated_image_url:", bool(res.get("annotated_image_url")))
    print("Has annotated_image_base64:", bool(res.get("annotated_image_base64")))
    if res.get("annotated_image_base64"):
        print("Base64 prefix:", res.get("annotated_image_base64")[:45])
    assert "annotated_image_base64" in res
    assert res.get("annotated_image_base64").startswith("data:image/jpeg;base64,")
    print("✅ OMR Base64 Verification Passed!")

def test_aadhaar():
    print("\n--- Testing Aadhaar Dual-Side Engine ---")
    front = np.full((350, 600, 3), 255, dtype=np.uint8)
    cv2.putText(front, 'Government of India', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, 'Rohit Sharma', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.putText(front, 'DOB: 12/04/1995', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, 'Male', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, '4512 8794 1234', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    cv2.putText(front, 'VID: 9123 4567 8901 2345', (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    back = np.full((350, 600, 3), 255, dtype=np.uint8)
    cv2.putText(back, 'Unique Identification Authority of India', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(back, 'Address: S/O Ramesh Kumar', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(back, 'Plot 45, Near Post Office, Chomu', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(back, 'Jaipur, Rajasthan - 303702', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(back, '4512 8794 1234', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    res = extract_aadhaar_dual_side(front, back, "front.jpg", "back.jpg")
    print("Aadhaar Result Data:")
    print(json.dumps(res.get("data"), indent=2))
    assert res.get("data", {}).get("name") == "Rohit Sharma"
    assert res.get("data", {}).get("aadhaar_number") == "4512 8794 1234"
    assert res.get("data", {}).get("pincode") == "303702"
    assert res.get("data", {}).get("care_of") == "Ramesh Kumar"
    assert res.get("data", {}).get("state") == "Rajasthan"
    assert res.get("data", {}).get("district") == "Jaipur"
    print("✅ Aadhaar Verification Passed!")

if __name__ == "__main__":
    test_omr()
    test_aadhaar()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
