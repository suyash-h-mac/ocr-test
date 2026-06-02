"""
Document OCR Pipeline (PaddleOCR) with Lighting-Readability Gate
================================================================

What it does, per image:
  1. LIGHTING GATE  -> reject over-lit / glare / too-dim photos and ask for a re-upload.
                       (lighting only -- blur/skew are NOT checked, by request)
  2. PRE-PROCESS    -> illumination flattening + contrast normalization for OCR.
  3. OCR (PaddleOCR)-> deep-learning OCR that handles textured/security backgrounds
                       far better than Tesseract.
  4. MRZ PARSE      -> if the document is a passport/visa, also parse the machine-
                       readable zone into clean structured fields.

Run:
  python ocr_pipeline.py img1.png img2.jpg ...

First run downloads the PaddleOCR models automatically (needs internet, ~few hundred MB).
"""

import sys
import re
import cv2
import numpy as np

# PaddleOCR is imported lazily inside get_ocr() so the lighting gate can run
# even before the heavy model is loaded.
_OCR = None


def get_ocr():
    """Load PaddleOCR once and reuse (model load is slow)."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(use_textline_orientation=True, lang="en")
    return _OCR


# ----------------------------------------------------------------------
# 1. LIGHTING-READABILITY GATE
# ----------------------------------------------------------------------
# Design note: this gate works in TWO stages.
#
#   Stage A (cheap pre-filter, runs before OCR): rejects only photos that are
#   so dark or so blown-out that NOTHING could be read - i.e. obvious exposure
#   failures. This is intentionally permissive: pixel statistics alone cannot
#   reliably tell "glare reflection" from "white paper", so we do NOT try to
#   make the accept/reject call here for borderline cases.
#
#   Stage B (the real decision): your stated principle is
#   "if it's readable to the human eye it should be readable by the OCR reader."
#   The most faithful implementation of that is to let the OCR engine judge
#   readability. PaddleOCR reports a confidence per text line. Glare wipes out
#   characters -> confidence collapses. A clean document -> high confidence.
#   So after OCR we compute mean confidence and require it to clear a threshold.
#   Because PaddleOCR (unlike Tesseract) handles security-document backgrounds,
#   a low score genuinely means "unreadable" (e.g. glare), not "busy background".
#
# READABILITY_THRESHOLD is the one knob. 0.55 (=55%) is a sensible default:
# clean docs and well-lit security docs score well above it; glare-ruined
# photos fall below it.
READABILITY_THRESHOLD = 0.55


def prefilter_exposure(image):
    """
    Stage A. Returns (passes, reason, metrics).
    Rejects ONLY unrecoverable exposure: near-total darkness or near-total
    white-out. Everything else passes through to the OCR readability check.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_b = float(np.mean(gray))
    # fraction of the frame that is pure black / pure white
    black_pct = float((gray < 12).mean() * 100)
    white_pct = float((gray > 250).mean() * 100)

    metrics = {
        "mean_brightness": round(mean_b, 1),
        "near_black_pct": round(black_pct, 1),
        "near_white_pct": round(white_pct, 1),
    }

    if mean_b < 35:
        return False, "Photo is far too dark to read. Use brighter lighting.", metrics
    if mean_b > 250 and white_pct > 92:
        return False, "Photo is completely washed out. Reduce the light.", metrics
    return True, "Exposure acceptable.", metrics


# ----------------------------------------------------------------------
# 2. PRE-PROCESSING
# ----------------------------------------------------------------------
def preprocess(image):
    """Flatten illumination + normalize contrast. Returns a BGR image (PaddleOCR wants color)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten uneven background lighting using a large-kernel illumination estimate.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=30)
    norm = cv2.divide(gray, bg, scale=255)

    # Local contrast boost.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    # Upscale small images so text is large enough.
    h, w = norm.shape
    if max(h, w) < 1600:
        s = 1600 / max(h, w)
        norm = cv2.resize(norm, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # PaddleOCR expects a 3-channel image.
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------
# 3. OCR with PaddleOCR
# ----------------------------------------------------------------------
def extract_text(processed_bgr):
    """
    Run PaddleOCR. Returns (lines, mean_confidence).
    lines: list of (text, confidence)
    """
    ocr = get_ocr()
    result = ocr.predict(processed_bgr)

    lines = []
    # PaddleOCR 3.x: result is a list of dict-like objects with rec_texts / rec_scores.
    for page in result:
        data = page.json["res"] if hasattr(page, "json") else page
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        for t, s in zip(texts, scores):
            if t and t.strip():
                lines.append((t.strip(), float(s)))

    mean_conf = float(np.mean([c for _, c in lines]) * 100) if lines else 0.0
    return lines, mean_conf


# ----------------------------------------------------------------------
# 4. MRZ PARSING (passports & visas)
# ----------------------------------------------------------------------
def parse_mrz(image):
    """
    Try to read and parse the Machine-Readable Zone from the bottom band.
    Returns a dict of fields if an MRZ is found, else None.
    Uses PassportEye if installed; otherwise returns None gracefully.
    """
    try:
        from passporteye import read_mrz
    except ImportError:
        return None

    # Save the bottom band to a temp file (PassportEye reads from a path/array).
    import tempfile, os
    h = image.shape[0]
    band = image[int(h * 0.75):h, :]
    band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, band)
    try:
        mrz = read_mrz(tmp.name)
    finally:
        os.unlink(tmp.name)

    if mrz is None:
        return None
    d = mrz.to_dict()
    return {
        "type": d.get("type"),
        "country": d.get("country"),
        "surname": d.get("surname"),
        "given_names": d.get("names"),
        "number": d.get("number"),
        "nationality": d.get("nationality"),
        "date_of_birth": d.get("date_of_birth"),
        "sex": d.get("sex"),
        "expiry": d.get("expiration_date"),
        "valid": d.get("valid_score"),
    }


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def process_document(path):
    print("=" * 66)
    print(f"FILE: {path}")
    print("=" * 66)

    image = cv2.imread(path)
    if image is None:
        print("ERROR: could not read image. Check the path/format.\n")
        return

    # --- Stage A: cheap exposure pre-filter ---
    ok, reason, metrics = prefilter_exposure(image)
    print(f"Exposure pre-filter: {metrics}")
    if not ok:
        print(f"\n  >> REJECTED (lighting): {reason}")
        print("  >> ACTION: please upload another photo.\n")
        return
    print(f"  Pre-filter passed: {reason}")

    # --- Preprocess + OCR ---
    print("  Pre-processing...")
    processed = preprocess(image)
    print("  Running PaddleOCR...")
    lines, mean_conf = extract_text(processed)

    # --- Stage B: readability decision (the real lighting/glare gate) ---
    print(f"\n  OCR mean confidence: {mean_conf:.1f}%  "
          f"(threshold {READABILITY_THRESHOLD*100:.0f}%)")
    if not lines or mean_conf < READABILITY_THRESHOLD * 100:
        print("\n  >> REJECTED (not readable): text could not be read confidently,")
        print("     most likely due to glare / poor lighting on the document.")
        print("  >> ACTION: please upload another photo.\n")
        return

    print(f"\n---------- EXTRACTED TEXT (mean confidence {mean_conf:.1f}%) ----------")
    for t, c in lines:
        print(f"  [{c*100:5.1f}%] {t}")
    print("-" * 66)

    # --- MRZ (travel documents) ---
    mrz = parse_mrz(image)
    if mrz:
        print("\n  MRZ structured fields:")
        for k, v in mrz.items():
            print(f"    {k:15}: {v}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image1> [image2] ...")
        sys.exit(1)
    for p in sys.argv[1:]:
        process_document(p)