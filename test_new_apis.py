"""
Test for Tanuja Sharma DOB (24/12/1990 vs Issue date 23/12/2011)
"""
import json
import cv2
import numpy as np

from aadhaar_engine import extract_aadhaar_dual_side

def test_tanuja_sharma():
    front = np.full((380, 650, 3), 255, dtype=np.uint8)
    cv2.putText(front, 'Government of India', (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    # Enrolment / Issue date at top or side
    cv2.putText(front, 'Enrolment No: 1029/38291/02910 23/12/2011', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    cv2.putText(front, 'Tanuja Sharma', (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    cv2.putText(front, 'DOB: 24/12/1990', (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
    cv2.putText(front, 'Female', (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
    cv2.putText(front, '8215 6872 1541', (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)

    back = np.full((380, 650, 3), 255, dtype=np.uint8)
    cv2.putText(back, 'Unique Identification Authority of India', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    cv2.putText(back, 'Address: C/O: Prashant Sharma, A-908, Manglam Aadhar', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    cv2.putText(back, 'Gandhi Path West, Jaipur, PO:Vaishali Nagar, Di T: Jaipur', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    cv2.putText(back, 'Rajasthan, 302021', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
    cv2.putText(back, '8215 6872 1541', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)

    res = extract_aadhaar_dual_side(front, back, "front.jpeg", "back.jpeg")
    print("\nExtracted Data:")
    print(json.dumps(res, indent=2))

    data = res["data"]
    print("\nVerified DOB:", data.get("dob"))
    assert data.get("dob") == "24/12/1990", f"Expected 24/12/1990, got {data.get('dob')}"
    assert data.get("name") == "Tanuja Sharma"
    assert data.get("gender") == "FEMALE"
    assert data.get("aadhaar_number") == "8215 6872 1541"
    assert "A-908" in data.get("address")
    assert not data.get("address").startswith("C/O")
    print("✅ Tanuja Sharma 24/12/1990 Verification PASSED!")

if __name__ == "__main__":
    test_tanuja_sharma()
