"""
Test for updated clean Aadhaar extraction schema.
"""
import json
import cv2
import numpy as np

from aadhaar_engine import extract_aadhaar_dual_side, clean_direct_address

def test_direct_address():
    raw_addr = "C/O:Prashant Sharma, A-908, Manglam Aadhar, , Gandhi Path West, Jaipur, PO:Vaishali Nagar, Di T:, Jaipur, , Rajasthan, 302021"
    cleaned = clean_direct_address(raw_addr)
    print("Raw Address:", raw_addr)
    print("Cleaned Address:", cleaned)
    assert not cleaned.startswith("C/O")
    assert "Di T" not in cleaned
    assert ", ," not in cleaned
    assert cleaned.startswith("A-908")
    print("✅ Address Cleaner Verified!")

def test_aadhaar_schema():
    front = np.full((350, 600, 3), 255, dtype=np.uint8)
    cv2.putText(front, 'Government of India', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, 'Tanuja Sharma', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.putText(front, 'DOB: 23/12/2011', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, 'Female', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(front, '8215 6872 1541', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    back = np.full((350, 600, 3), 255, dtype=np.uint8)
    cv2.putText(back, 'Unique Identification Authority of India', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(back, 'Address: C/O: Prashant Sharma, A-908, Manglam Aadhar', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(back, 'Gandhi Path West, Jaipur, PO:Vaishali Nagar', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(back, 'Jaipur, Rajasthan, 302021', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(back, '8215 6872 1541', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    res = extract_aadhaar_dual_side(front, back, "front.jpeg", "back.jpeg")
    print("\nExtracted Data:")
    print(json.dumps(res, indent=2))

    data = res["data"]
    # Verify unwanted fields are absent
    assert "vid" not in data
    assert "care_of" not in data
    assert "relation" not in data
    assert "district" not in data
    assert "state" not in data
    assert "yob" not in data

    # Verify required fields are present
    assert data["name"] == "Tanuja Sharma"
    assert data["gender"] == "FEMALE"
    assert data["aadhaar_number"] == "8215 6872 1541"
    assert data["pincode"] == "302021"
    assert "A-908" in data["address"]
    assert not data["address"].startswith("C/O")
    print("✅ Aadhaar Clean Schema Verified!")

if __name__ == "__main__":
    test_direct_address()
    test_aadhaar_schema()
    print("\n🎉 ALL TESTS PASSED!")
