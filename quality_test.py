"""
quality_test.py -- Test the quality gate on one or more images
Usage:
    python3 quality_test.py image1.jpg image2.jpg ...
"""

import sys
import numpy as np
from PIL import Image
from preprocessing import quality_check

def test(path):
    try:
        img = np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return

    result = quality_check(img)
    status = "PASS " if result["passed"] else "FAIL "
    print(f"\n{'='*55}")
    print(f"  File   : {path}")
    print(f"  Result : {status}")
    if not result["passed"]:
        print(f"  Reason : {result['reason']}")
    print(f"  Metrics:")
    for k, v in result["metrics"].items():
        print(f"    {k:<22} {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 quality_test.py <image1> [image2] ...")
        sys.exit(1)
    for path in sys.argv[1:]:
        test(path)
    print(f"\n{'='*55}\n")
