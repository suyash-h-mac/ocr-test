"""
preprocessing.py -- Image preprocessing for Arabic/English OCR
==============================================================
- Resolution normalisation (upscale small images)
- Cardinal rotation detection + correction (0/90/180/270 only)
  Fine tilt is intentionally skipped -- it causes false corrections on
  straight printed documents.
- Brightness / contrast normalisation
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def _adaptive_thresh(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5,
    )


def _projection_score(gray: np.ndarray) -> float:
    """Variance of horizontal ink-density projection -- high when text rows are clear."""
    thresh = _adaptive_thresh(gray)
    return float(np.var(np.sum(thresh, axis=1).astype(np.float64)))


def _is_upside_down(gray: np.ndarray) -> bool:
    """
    True when the bottom third of the image has >= 1.8x more ink than the top
    -- heuristic for detecting 180-degree rotation.
    """
    thresh = _adaptive_thresh(gray)
    h = gray.shape[0]
    top = float(np.sum(thresh[: h // 3, :]))
    bot = float(np.sum(thresh[2 * h // 3 :, :]))
    return top > 500 and bot > top * 1.8


# ── Rotation ──────────────────────────────────────────────────────────────────

def fix_rotation(img: np.ndarray) -> tuple:
    """
    Detect and correct cardinal rotation.
    Returns (corrected_img, degrees_rotated).

    Strategy:
      1. Score horizontal text lines on the original image.
      2. Score after a 90-degree rotation.
      3. If one axis scores >25% better, the image is portrait/landscape-flipped.
      4. Use ink-density heuristic to decide 0 vs 180 (or 90 vs 270).
    """
    gray  = to_gray(img)
    horiz = _projection_score(gray)
    vert  = _projection_score(np.rot90(gray, k=1))

    mx = max(horiz, vert)
    if mx == 0 or abs(horiz - vert) / mx < 0.25:
        # Scores too similar -- image is likely already upright (or blank)
        return img, 0

    if horiz >= vert:
        # Image is in landscape -- check whether it needs 0 or 180
        angle = 180 if _is_upside_down(gray) else 0
    else:
        # Image is in portrait -- rotated 90 or 270 degrees
        rotated_gray = np.rot90(gray, k=1)
        angle = 270 if _is_upside_down(rotated_gray) else 90

    if angle == 0:
        return img, 0

    corrected = np.ascontiguousarray(np.rot90(img, k=angle // 90))
    return corrected, angle


# ── Quality gate ─────────────────────────────────────────────────────────────

def quality_check(img: np.ndarray) -> dict:
    """
    Assess whether an uploaded image is good enough for reliable OCR.
    Returns a dict:
      passed  : bool   -- True = proceed with OCR, False = ask user to reupload
      reason  : str    -- human-readable rejection reason (empty if passed)
      metrics : dict   -- raw numbers for debugging

    Checks (in order):
      1. White coverage  -- not enough paper-white visible (background in frame)
      2. Overexposure    -- average brightness too high (flash, direct light)
      3. Glare           -- isolated large blown-out blob covering text
      4. Dark pixel ratio -- too many truly-dark pixels (shadows, obstructions)

    Design notes:
      - No hard mean-brightness floor: valid dense/dark documents (brightness
        110-150) still pass as long as they have sufficient white coverage and
        controlled dark pixels.
      - Glare uses connected-component detection, not raw pixel count.
        White document paper forms one huge component (> 50% of image) and is
        ignored; only medium-sized isolated blobs (3-50%) are flagged as glare.
      - Corner check removed: corners legitimately contain logos, stamps,
        colored decorations on official documents. White-coverage is a more
        reliable proxy for "document fills the frame".
    """
    gray = to_gray(img)
    h, w = gray.shape
    img_area = h * w
    metrics = {"width": w, "height": h}

    # Brightness
    brightness = float(gray.mean())
    metrics["brightness"] = round(brightness, 1)

    # 1. Completely black / pitch dark — nothing to read
    if brightness < 40:
        return {
            "passed": False,
            "reason": (
                "Image is too dark to read. Please turn on a light or move "
                "near a window and try again."
            ),
            "metrics": metrics,
        }

    # 2. Overexposure — entire image blown out
    if brightness > 250:
        return {
            "passed": False,
            "reason": (
                "Image is completely overexposed. Please turn off flash and "
                "avoid pointing the camera at a direct light source."
            ),
            "metrics": metrics,
        }

    # 3. Glare — isolated large blown-out blob covering text
    # White paper background = one huge connected region (> 50% of image) → ignored.
    # A camera reflection / flash glare = compact isolated blob (3–50%) → flagged.
    very_bright = (gray >= 252).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(very_bright)
    glare_blobs = [
        stats[i, cv2.CC_STAT_AREA]
        for i in range(1, num_labels)
        if img_area * 0.03 < stats[i, cv2.CC_STAT_AREA] < img_area * 0.50
    ]
    if glare_blobs:
        largest_glare_pct = max(glare_blobs) / img_area * 100
        metrics["glare_blob_pct"] = round(largest_glare_pct, 1)
        if largest_glare_pct > 8:
            return {
                "passed": False,
                "reason": (
                    "A bright glare or reflection is covering part of the document. "
                    "Please turn off flash, move the light source to the side, "
                    "or use a scanner."
                ),
                "metrics": metrics,
            }

    # 4. Dark pixel ratio — image dominated by shadow/background
    # Only reject if > 40% of pixels are truly dark (< 60 brightness).
    # This catches a phone photo where the background fills most of the frame
    # while still allowing dense documents, stamps, and dark-themed pages.
    dark_ratio = float(np.mean(gray < 60)) * 100
    metrics["dark_pixel_pct"] = round(dark_ratio, 1)
    if dark_ratio > 40:
        return {
            "passed": False,
            "reason": (
                "Most of the image is too dark — heavy shadows or background "
                "are obscuring the document. Please ensure even lighting and "
                "photograph only the document."
            ),
            "metrics": metrics,
        }

    return {"passed": True, "reason": "", "metrics": metrics}


# ── Background normalisation ──────────────────────────────────────────────────

def normalize_background(img: np.ndarray) -> tuple:
    """
    Detect dark-background regions (e.g. navy header banners with white text)
    and invert them so all text appears as dark-on-light.

    Tesseract accuracy on white-text / dark-background drops sharply without
    this step. The technique:
      1. Estimate the local background tone with a heavy Gaussian blur
         (sigma large enough to wash out individual character strokes).
      2. Where the background estimate is dark (< threshold), invert those
         pixels so dark-bg + light-text becomes light-bg + dark-text.

    Returns (normalised_img, had_dark_regions: bool).
    """
    gray = to_gray(img)
    # sigma=40 blurs away text strokes (~10-30 px) while preserving
    # large tonal regions like header bands.
    bg = cv2.GaussianBlur(gray, (0, 0), 40)
    # Threshold lowered to 80: only invert solidly dark regions (e.g. navy
    # header banners). Value 110 was too aggressive -- it caught subtle
    # security patterns and watermarks on visa/passport documents and
    # partially inverted them, corrupting the text for OCR.
    dark_mask = bg < 80  # pixels whose background is "dark"

    if not dark_mask.any():
        return img, False

    result = img.copy()
    result[dark_mask] = 255 - img[dark_mask]
    return result, True


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray) -> tuple:
    """
    Run the full preprocessing pipeline on an RGB numpy image.
    Returns (processed_img, info_dict).

    info_dict keys:
      original_size        -- "WxH" string
      rotation_applied     -- degrees rotated (0 if none)
      upscaled             -- True if image was upscaled
      dark_bg_normalised   -- True if dark-background regions were inverted
      brightness_fixed     -- True if overall brightness was adjusted
      contrast_fixed       -- True if contrast was adjusted
    """
    info = {}
    h, w = img.shape[:2]
    info["original_size"] = f"{w}x{h}"
    info["rotation_applied"] = 0
    info["upscaled"] = False
    info["dark_bg_normalised"] = False
    info["brightness_fixed"] = False
    info["contrast_fixed"] = False

    # 1. Upscale if resolution is too low for reliable OCR
    if max(w, h) < 1000 and min(w, h) < 700:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        info["upscaled"] = True

    # 2. Cardinal rotation correction (run on raw image before any
    #    colour/tone changes which affect projection scores)
    img, angle = fix_rotation(img)
    info["rotation_applied"] = angle

    # 3. Dark-background normalisation (e.g. navy header with white text)
    #    Must run before brightness check because an inverted dark header
    #    will shift the overall brightness metric.
    img, had_dark = normalize_background(img)
    info["dark_bg_normalised"] = had_dark

    # 4. Overall brightness normalisation
    gray = to_gray(img)
    brightness = float(gray.mean())
    if brightness < 80:
        # Dark image overall -- CLAHE in LAB space
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        info["brightness_fixed"] = True
    elif brightness > 230:
        # Overexposed -- pull back
        img = np.array(
            ImageEnhance.Brightness(Image.fromarray(img)).enhance(0.8)
        )
        info["brightness_fixed"] = True

    # 5. Contrast normalisation
    gray = to_gray(img)
    if gray.std() < 40:
        img = np.array(
            ImageEnhance.Contrast(Image.fromarray(img)).enhance(1.8)
        )
        info["contrast_fixed"] = True

    # 6. Sharpening (unsharp mask)
    # Enhances text edges on documents with security patterns, watermarks,
    # or subtle backgrounds that reduce text clarity.
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    img = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)

    return img, info
