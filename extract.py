"""
extract.py -- Extract and display text from one or more documents
================================================================
Usage:
    python3 extract.py image1.jpg image2.png document.pdf ...

The OCR server must be running:
    uvicorn app:app --host 127.0.0.1 --port 8003
"""

import sys
import json
import urllib.request
import urllib.parse
import mimetypes
from pathlib import Path


SERVER = "http://127.0.0.1:8003"
SEP    = "=" * 65


def ocr_file(path: str) -> dict:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

    # Build multipart/form-data manually (no extra dependencies)
    boundary = "----OCRBoundary7f3a9c"
    body  = f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'.encode()
    body += f"Content-Type: {mime}\r\n\r\n".encode()
    body += p.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{SERVER}/ocr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def display(path: str, d: dict):
    print(f"\n{SEP}")
    print(f"  File    : {path}")

    if not d.get("accepted", True):
        print(f"  Status  : REJECTED ❌")
        print(f"  Reason  : {d['rejection_reason']}")
        print(f"  Metrics : {d.get('quality_metrics', {})}")
        print(SEP)
        return

    conf   = d.get("avg_confidence", 0)
    rot    = d.get("rotation_applied", 0)
    engine = d.get("engine", "?")
    words  = d.get("total_words", 0)
    ms     = d.get("elapsed_ms", 0)
    pre    = d.get("preprocessing", {})

    print(f"  Status  : ACCEPTED ✅")
    print(f"  Engine  : {engine}")
    print(f"  Words   : {words}   Confidence: {conf:.0%}   Time: {ms} ms")
    if rot:
        print(f"  Rotated : {rot}°")

    flags = [k for k, v in pre.items() if v]
    if flags:
        print(f"  Fixed   : {', '.join(flags)}")

    text = d.get("full_text", "").strip()
    print(f"\n--- Extracted Text ---")
    print(text if text else "(no text detected)")
    print(SEP)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract.py <file1> [file2] ...")
        print("Example: python3 extract.py ~/Downloads/visa.jpg ~/Downloads/salary.png")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            d = ocr_file(path)
            display(path, d)
        except ConnectionRefusedError:
            print(f"\n[ERROR] Cannot connect to {SERVER}")
            print("Make sure the server is running:")
            print("  uvicorn app:app --host 127.0.0.1 --port 8003")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] {path}: {e}")

    print()


if __name__ == "__main__":
    main()
