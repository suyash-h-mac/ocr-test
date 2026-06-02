#!/bin/bash
# setup.sh -- one-command setup for ocr_service
set -e

echo ""
echo "================================================"
echo "  OCR Service -- Setup"
echo "================================================"

# ── 1. Tesseract (system install) ────────────────────────────────────────────
echo ""
echo "Step 1: Tesseract + Arabic language data"
if command -v tesseract &>/dev/null && tesseract --list-langs 2>&1 | grep -q "^ara$"; then
    echo "  Already installed."
else
    echo "  Installing via Homebrew (this may take a few minutes)..."
    brew install tesseract tesseract-lang
    echo "  Done."
fi

# ── 2. Python venv ───────────────────────────────────────────────────────────
echo ""
echo "Step 2: Python virtual environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created venv."
else
    echo "  Already exists."
fi
source venv/bin/activate

# ── 3. Python packages ───────────────────────────────────────────────────────
echo ""
echo "Step 3: Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  To start the service:"
echo "    source venv/bin/activate"
echo "    uvicorn app:app --host 127.0.0.1 --port 8003 --reload"
echo ""
echo "  To test (in another terminal):"
echo "    curl -s -X POST http://127.0.0.1:8003/ocr \\"
echo "      -F \"file=@/path/to/image.jpg\" \\"
echo "      | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['full_text'])\""
echo "================================================"
