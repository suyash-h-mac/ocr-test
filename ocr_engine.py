"""
ocr_engine.py -- Multi-engine OCR (Arabic + English)
=====================================================
Priority order:
  1. PaddleOCR  (ar + en auto-detect) -- best overall; needs one-time model download
  2. Tesseract  (ara+eng)             -- fallback if Paddle models not downloaded yet
  3. EasyOCR   (ar + en)             -- last resort

PaddleOCR model download (one-time, needs non-UAE network or VPN):
    Models cache to ~/.paddleocr/ automatically on first run.

Install Tesseract (system-level):
    brew install tesseract tesseract-lang
    pip install pytesseract
"""

import re
import cv2
import numpy as np
from PIL import Image


# ── Arabic detection ──────────────────────────────────────────────────────────

_ARABIC_RE = re.compile(
    "["
    "؀-ۿ"   # Arabic block
    "ﭐ-﷿"   # Arabic Presentation Forms-A
    "ﹰ-﻿"   # Arabic Presentation Forms-B
    "]"
)


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(words: list) -> float:
    """word_count x avg_confidence -- higher is better."""
    if not words:
        return 0.0
    avg = sum(w["confidence"] for w in words) / len(words)
    return len(words) * avg


# ── Garbage filter ────────────────────────────────────────────────────────────

_VALID_RE = re.compile(
    "["
    "؀-ۿ"   # Arabic block
    "ﭐ-﷿"   # Arabic Presentation Forms-A
    "ﹰ-﻿"   # Arabic Presentation Forms-B
    "a-zA-Z0-9"
    " -/"   # ASCII space + punctuation
    ":-@"
    "[-`"
    "{-~"
    "،؍؛؟"  # Arabic-specific punctuation
    "ـ٪-٭۔"
    "–—"    # en-dash, em-dash
    "‘-‟"   # curly quotes
    "]"
)


def _filter_garbage(words: list) -> list:
    """
    Drop detections where fewer than 55% of characters are valid
    Arabic, English, digits, or punctuation.
    Short words (<= 2 chars) are always kept.
    """
    clean = []
    for w in words:
        t = w["text"].strip()
        if not t:
            continue
        if len(t) <= 2:
            clean.append(w)
            continue
        valid = len(_VALID_RE.findall(t))
        if valid / len(t) >= 0.55:
            clean.append(w)
    return clean


# ── PaddleOCR engine ─────────────────────────────────────────────────────────

_paddle_ar = None
_paddle_en = None
_paddle_available = None   # None = untested, True/False after first attempt


def _paddle_ready() -> bool:
    """
    Returns True if PaddleOCR is installed AND models are already cached.
    Never triggers a download -- if models are missing it returns False
    and we fall through to Tesseract.
    """
    global _paddle_available
    if _paddle_available is not None:
        return _paddle_available
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        import pathlib, os
        cache = pathlib.Path.home() / ".paddleocr"
        # Check at least one recognition model exists
        rec_dirs = list(cache.glob("**/rec/**/*.pdmodel"))
        _paddle_available = len(rec_dirs) > 0
        if not _paddle_available:
            print("[PaddleOCR] Models not yet downloaded -- skipping (use Tesseract)")
    except ImportError:
        print("[PaddleOCR] Not installed -- skipping")
        _paddle_available = False
    return _paddle_available


def _get_paddle_ar():
    global _paddle_ar
    if _paddle_ar is None:
        from paddleocr import PaddleOCR
        print("[PaddleOCR] Loading Arabic model...")
        _paddle_ar = PaddleOCR(
            use_angle_cls=True, lang="ar",
            ocr_version="PP-OCRv4", show_log=False,
        )
    return _paddle_ar


def _get_paddle_en():
    global _paddle_en
    if _paddle_en is None:
        from paddleocr import PaddleOCR
        print("[PaddleOCR] Loading English model...")
        _paddle_en = PaddleOCR(
            use_angle_cls=True, lang="en",
            ocr_version="PP-OCRv4", show_log=False,
        )
    return _paddle_en


def _run_paddle_model(ocr_model, img: np.ndarray) -> list:
    result = ocr_model.ocr(img, cls=True)
    if not result or result[0] is None:
        return []
    words = []
    for box, (text, conf) in result[0]:
        text = text.strip()
        if text:
            words.append({"text": text, "confidence": round(float(conf), 3)})
    return words


def run_paddleocr(img: np.ndarray) -> tuple:
    """
    Run PaddleOCR with Arabic + English models; auto-pick the better result.
    Returns (words, full_text) or ([], '') if models unavailable.
    """
    if not _paddle_ready():
        return [], ""
    try:
        ar_words = _run_paddle_model(_get_paddle_ar(), img)
        en_words = _run_paddle_model(_get_paddle_en(), img)

        # If Arabic model found Arabic characters use it; otherwise score-pick
        ar_text = " ".join(w["text"] for w in ar_words)
        if contains_arabic(ar_text):
            words = ar_words
            lang_used = "ar"
        else:
            words = ar_words if _score(ar_words) >= _score(en_words) else en_words
            lang_used = "ar" if words is ar_words else "en"

        words = _filter_garbage(words)
        full_text = "\n".join(w["text"] for w in words)
        print(f"[PaddleOCR/{lang_used}] {len(words)} words  score={_score(words):.1f}")
        return words, full_text

    except Exception as e:
        print(f"[PaddleOCR] error: {e}")
        return [], ""


# ── Tesseract engine ──────────────────────────────────────────────────────────

_tess_lang = None   # cached after first call


def _tesseract_lang_string() -> str:
    global _tess_lang
    if _tess_lang is not None:
        return _tess_lang
    try:
        import pytesseract
        pytesseract.get_tesseract_version()          # raises if not installed
        available = pytesseract.get_languages()
        _tess_lang = "ara+eng" if "ara" in available else "eng"
        print(f"[Tesseract] available, using lang={_tess_lang}")
    except Exception as e:
        print(f"[Tesseract] not available: {e}")
        _tess_lang = ""
    return _tess_lang


def run_tesseract(img: np.ndarray) -> tuple:
    """
    Run Tesseract OCR.
    Returns (words, full_text) or ([], '') if Tesseract is unavailable.
    """
    lang = _tesseract_lang_string()
    if not lang:
        return [], ""

    try:
        import pytesseract
        pil = Image.fromarray(img)
        config = "--oem 3 --psm 3"
        if "ara" in lang:
            config += " -c preserve_interword_spaces=1"

        data = pytesseract.image_to_data(
            pil, lang=lang, config=config,
            output_type=pytesseract.Output.DICT,
        )
        words = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            raw_conf = float(data["conf"][i])
            if raw_conf < 0:
                continue
            words.append({
                "text": text,
                "confidence": round(min(raw_conf / 100.0, 1.0), 3),
            })

        words = _filter_garbage(words)
        full_text = pytesseract.image_to_string(
            pil, lang=lang, config=config
        ).strip()

        print(f"[Tesseract] {len(words)} words  score={_score(words):.1f}")
        return words, full_text

    except Exception as e:
        print(f"[Tesseract] error: {e}")
        return [], ""


# ── EasyOCR engine ────────────────────────────────────────────────────────────

_easyocr_reader = None


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[EasyOCR] Loading model (ar + en) -- first call only...")
        _easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
        print("[EasyOCR] Model ready.")
    return _easyocr_reader


def _binarize(img: np.ndarray) -> np.ndarray:
    """CLAHE + Otsu binarisation -- improves EasyOCR Arabic accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def run_easyocr(img: np.ndarray) -> tuple:
    """
    Run EasyOCR with dual colour + binary pass; pick whichever scores higher.
    Returns (words, full_text) or ([], '') on failure.
    """
    try:
        reader = _get_easyocr()
        params = dict(
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3,
            paragraph=False,
            width_ths=0.9,   # wider merge window helps Arabic RTL
        )

        def parse(raw):
            return _filter_garbage([
                {"text": t, "confidence": round(float(c), 3)}
                for _, t, c in raw if t.strip()
            ])

        words_c = parse(reader.readtext(img, **params))
        words_b = parse(reader.readtext(_binarize(img), **params))
        words = words_b if _score(words_b) >= _score(words_c) else words_c

        full_text = "\n".join(w["text"] for w in words)
        print(f"[EasyOCR] {len(words)} words  score={_score(words):.1f}")
        return words, full_text

    except Exception as e:
        print(f"[EasyOCR] error: {e}")
        return [], ""


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ocr(img: np.ndarray) -> tuple:
    """
    Run OCR on a preprocessed RGB image.
    Returns (words, full_text, engine_name).

    Priority:
      1. PaddleOCR  -- tried first; skipped silently if models not downloaded
      2. Tesseract  -- fallback; needs: brew install tesseract tesseract-lang
      3. EasyOCR   -- last resort if both above are unavailable / score too low
    """
    results = {}

    # 1. PaddleOCR (best when models are available)
    paddle_words, paddle_text = run_paddleocr(img)
    if paddle_words:
        results["PaddleOCR"] = (paddle_words, paddle_text)

    # 2. Tesseract -- always try unless Paddle already scored well
    if _score(paddle_words) < 5.0:
        tess_words, tess_text = run_tesseract(img)
        if tess_words:
            results["Tesseract"] = (tess_words, tess_text)

    # 3. EasyOCR -- only if neither above engine produced enough output
    best_so_far = max((_score(v[0]) for v in results.values()), default=0.0)
    if best_so_far < 3.0:
        easy_words, easy_text = run_easyocr(img)
        if easy_words:
            results["EasyOCR"] = (easy_words, easy_text)

    if not results:
        return [], "", "none"

    best_engine = max(results, key=lambda k: _score(results[k][0]))
    for name, (w, _) in results.items():
        marker = "=>" if name == best_engine else "  "
        print(f"  {marker} {name}: score={_score(w):.1f}")

    words, full_text = results[best_engine]
    return words, full_text, best_engine
