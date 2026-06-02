import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageDraw

# Build a simple test image with text
img = Image.new("RGB", (400, 80), "white")
ImageDraw.Draw(img).text((20, 25), "Hello PP-OCRv5", fill="black")
img_array = np.array(img)

print("Loading PP-OCRv5 model...")
ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    lang="en",
)

print("Running OCR...")
result = ocr.ocr(img_array)
print("Raw result:", result)

# Parse output
if result and result[0]:
    print("\n--- Detected text ---")
    for line in result[0]:
        box, (text, conf) = line
        print(f"  '{text}'  confidence={conf:.3f}")
else:
    print("No text detected.")
