"""
app.py -- OCR Service (Arabic + English)
========================================
Handles images (JPG, PNG, BMP, TIFF, WEBP) and PDFs.
Auto-rotates images that need it. Arabic-first OCR.

Run:
    uvicorn app:app --host 127.0.0.1 --port 8003 --reload

Open UI:
    http://127.0.0.1:8003

Test via curl:
    curl -s -X POST http://127.0.0.1:8003/ocr \
         -F "file=@/path/to/image.jpg" | python3 -m json.tool
"""

import base64
import io
import time
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

from preprocessing import preprocess, quality_check
from ocr_engine import run_ocr

app = FastAPI(
    title="OCR Service",
    description="Arabic + English OCR with auto-rotation. Tesseract primary, EasyOCR fallback.",
    version="1.0",
)

# ── Web UI ────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Service</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0f1117;color:#e2e8f0;min-height:100vh}
.topbar{background:#13151f;border-bottom:1px solid #1e2235;padding:14px 24px;
  display:flex;align-items:center;gap:12px}
.topbar h1{font-size:16px;font-weight:700;color:#818cf8}
.topbar span{font-size:12px;color:#374151}
.wrap{max-width:1200px;margin:0 auto;padding:24px 18px}

/* Upload zone */
.drop{border:2px dashed #2d3460;border-radius:12px;padding:40px 24px;
  text-align:center;cursor:pointer;transition:.2s;background:#13151f;margin-bottom:20px}
.drop:hover,.drop.over{border-color:#818cf8;background:#16193a}
.drop strong{color:#a5b4fc;font-size:15px;display:block;margin-bottom:6px}
.drop p{color:#4b5563;font-size:13px}
input[type=file]{display:none}
.btn{display:inline-block;padding:9px 22px;background:#4f46e5;color:#fff;
  border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;
  margin-top:14px;border:none;transition:.2s}
.btn:hover{background:#4338ca}

/* Spinner */
.spin-wrap{display:none;text-align:center;padding:32px;color:#6b7280;font-size:14px}
.ring{display:inline-block;width:32px;height:32px;border:3px solid #1e2235;
  border-top-color:#818cf8;border-radius:50%;
  animation:spin .75s linear infinite;vertical-align:middle;margin-right:10px}
@keyframes spin{to{transform:rotate(360deg)}}

/* Error */
.err-box{background:#1f0707;border:1px solid #7f1d1d;border-radius:8px;
  padding:14px;color:#fca5a5;font-size:13px;margin-bottom:16px;display:none}

/* Rejection banner */
.reject-banner{background:#1c1007;border:2px solid #d97706;border-radius:10px;
  padding:20px 24px;margin-bottom:18px;display:none}
.reject-banner .icon{font-size:28px;margin-bottom:8px}
.reject-banner h2{color:#fbbf24;font-size:16px;font-weight:700;margin-bottom:8px}
.reject-banner p{color:#fcd34d;font-size:14px;line-height:1.6;margin-bottom:12px}
.reject-banner .metrics{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.reject-banner .metric{background:#27180a;border:1px solid #92400e;border-radius:5px;
  padding:4px 10px;font-size:11px;color:#fbbf24}

/* Stats row */
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}
.stat{flex:1;min-width:110px;background:#13151f;border:1px solid #1e2235;
  border-radius:8px;padding:10px 14px}
.stat label{font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:.06em;display:block}
.stat .v{font-size:17px;font-weight:700;margin-top:3px;color:#818cf8}
.v.good{color:#4ade80}.v.mid{color:#facc15}.v.bad{color:#f87171}

/* Preprocessing badges */
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
.badge{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600}
.badge.on{background:#14532d;color:#86efac}
.badge.off{background:#1e2235;color:#374151}

/* Main panels */
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){.panels{grid-template-columns:1fr}}
.panel{background:#13151f;border:1px solid #1e2235;border-radius:10px;overflow:hidden}
.panel-hd{background:#16193a;padding:10px 14px;font-size:11px;font-weight:700;
  color:#6b7280;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:1px solid #1e2235}
.panel-body{padding:14px;max-height:480px;overflow-y:auto}

/* Image preview */
.img-preview{width:100%;display:block;max-height:440px;object-fit:contain;background:#000}

/* Text output — RTL so Arabic renders correctly */
.text-out{white-space:pre-wrap;font-family:'SF Mono',Menlo,Consolas,monospace;
  font-size:13px;line-height:2;color:#cbd5e1;direction:rtl;
  text-align:right;unicode-bidi:plaintext}

/* Word chips */
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{padding:3px 9px;border-radius:4px;font-size:12px;cursor:default;
  white-space:nowrap}
.chip.ok{background:#14532d;color:#86efac}
.chip.mid{background:#78350f;color:#fcd34d}
.chip.bad{background:#450a0a;color:#fca5a5}

/* Copy button */
.copy-btn{float:right;padding:3px 10px;font-size:11px;background:#1e2235;
  border:none;color:#a5b4fc;border-radius:4px;cursor:pointer;margin-top:-2px}
.copy-btn:hover{background:#2d3460}

/* History */
.history{margin-top:32px}
.history h2{font-size:11px;color:#374151;text-transform:uppercase;
  letter-spacing:.07em;margin-bottom:12px}
.hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.hcard{background:#13151f;border:1px solid #1e2235;border-radius:8px;
  overflow:hidden;cursor:pointer;transition:.2s}
.hcard:hover{border-color:#818cf8;transform:translateY(-2px)}
.hcard img{width:100%;height:70px;object-fit:cover;display:block;background:#000}
.hinfo{padding:6px 8px}
.hname{font-size:10px;color:#a5b4fc;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.hmeta{font-size:10px;color:#374151;margin-top:2px}
.hconf{font-size:11px;font-weight:700;margin-top:2px}
</style>
</head>
<body>
<div class="topbar">
  <h1>OCR Service</h1>
  <span>Arabic + English &nbsp;·&nbsp; Tesseract &nbsp;·&nbsp; Auto-rotation</span>
</div>
<div class="wrap">
  <!-- Upload -->
  <div class="drop" id="dz" onclick="document.getElementById('fi').click()">
    <strong>Drop image or PDF here</strong>
    <p>JPG · PNG · BMP · TIFF · WEBP · PDF</p>
    <input type="file" id="fi" accept="image/*,.pdf" onchange="go(this.files[0])">
    <br>
    <button class="btn" onclick="event.stopPropagation();document.getElementById('fi').click()">
      Choose File
    </button>
  </div>

  <div class="spin-wrap" id="sw"><span class="ring"></span>Running OCR…</div>
  <div class="err-box" id="eb"></div>
  <div class="reject-banner" id="rb">
    <div class="icon">⚠️</div>
    <h2>Document quality too low — please reupload</h2>
    <p id="rb-reason"></p>
    <div class="metrics" id="rb-metrics"></div>
  </div>
  <div id="res" style="display:none"></div>

  <div class="history" id="hist" style="display:none">
    <h2>Recent uploads</h2>
    <div class="hist-grid" id="hg"></div>
  </div>
</div>

<script>
const history = [];

// Drag-and-drop
const dz = document.getElementById('dz');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  if (e.dataTransfer.files[0]) go(e.dataTransfer.files[0]);
});

async function go(file) {
  if (!file) return;
  document.getElementById('eb').style.display = 'none';
  document.getElementById('rb').style.display = 'none';
  document.getElementById('res').style.display = 'none';
  document.getElementById('sw').style.display = 'block';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/ocr', { method: 'POST', body: fd });
    const d = await r.json();
    document.getElementById('sw').style.display = 'none';
    if (d.detail || d.error) {
      showErr(d.detail || d.error); return;
    }
    if (!d.accepted) {
      showRejection(d); return;
    }
    render(d);
    addHist(d);
  } catch(e) {
    document.getElementById('sw').style.display = 'none';
    showErr(e.message);
  }
}

function showErr(msg) {
  const eb = document.getElementById('eb');
  eb.textContent = 'Error: ' + msg;
  eb.style.display = 'block';
}

function showRejection(d) {
  document.getElementById('rb').style.display = 'block';
  document.getElementById('rb-reason').textContent = d.rejection_reason;
  const m = d.quality_metrics || {};
  const labels = {
    blur_score: 'Blur score',
    corner_variation: 'Corner variation',
    brightness: 'Brightness',
    brightness_range: 'Brightness range',
    width: 'Width (px)',
    height: 'Height (px)',
  };
  document.getElementById('rb-metrics').innerHTML = Object.entries(m)
    .map(([k, v]) => `<span class="metric">${labels[k] || k}: ${v}</span>`)
    .join('');
}

function cc(c) { return c >= 0.80 ? 'good' : c >= 0.55 ? 'mid' : 'bad'; }

function render(d) {
  const pre = d.preprocessing || {};
  const avgPct = Math.round((d.avg_confidence || 0) * 100);
  document.getElementById('res').style.display = 'block';
  document.getElementById('res').innerHTML = buildHTML(d, avgPct, pre);
}

function buildHTML(d, avgPct, pre) {
  const origSrc = d.original_image  ? 'data:image/jpeg;base64,' + d.original_image  : '';
  const procSrc = d.processed_image ? 'data:image/jpeg;base64,' + d.processed_image : '';

  const badges = [
    ['Dark BG Fixed',  pre.dark_bg_normalised],
    ['Upscaled',       pre.upscaled],
    ['Brightness Fix', pre.brightness_fixed],
    ['Contrast Fix',   pre.contrast_fixed],
    ['Rotated ' + d.rotation_applied + '\xb0', d.rotation_applied !== 0],
  ];
  const badgeHTML = badges.map(([label, on]) =>
    `<span class="badge ${on ? 'on' : 'off'}">${esc(label)}</span>`
  ).join('');

  const chips = (d.words || []).map(w => {
    const cls = w.confidence >= 0.80 ? 'ok' : w.confidence >= 0.55 ? 'mid' : 'bad';
    return `<span class="chip ${cls}" title="${w.confidence}">${esc(w.text)}</span>`;
  }).join('');

  return `
  <div class="stats">
    <div class="stat"><label>Engine</label>
      <div class="v" style="font-size:13px;color:#60a5fa">${esc(d.engine)}</div></div>
    <div class="stat"><label>Confidence</label>
      <div class="v ${cc(d.avg_confidence)}">${avgPct}%</div></div>
    <div class="stat"><label>Words</label>
      <div class="v">${d.total_words}</div></div>
    <div class="stat"><label>Rotation</label>
      <div class="v" style="font-size:14px">${d.rotation_applied}&deg;</div></div>
    <div class="stat"><label>Time</label>
      <div class="v" style="font-size:14px">${d.elapsed_ms} ms</div></div>
  </div>
  <div class="badges">${badgeHTML}</div>
  <div class="panels">
    <div class="panel">
      <div class="panel-hd">&#128247; Original</div>
      <div class="panel-body" style="padding:0">
        ${origSrc ? `<img class="img-preview" src="${origSrc}" alt="original">` : '<p style="padding:14px;color:#374151">No preview</p>'}
      </div>
    </div>
    <div class="panel">
      <div class="panel-hd">&#10024; After Preprocessing (fed to OCR)</div>
      <div class="panel-body" style="padding:0">
        ${procSrc ? `<img class="img-preview" src="${procSrc}" alt="preprocessed">` : '<p style="padding:14px;color:#374151">No preview</p>'}
      </div>
    </div>
    <div class="panel">
      <div class="panel-hd">
        &#128196; Extracted Text
        <button class="copy-btn" onclick="navigator.clipboard.writeText(${JSON.stringify(d.full_text)})">
          Copy
        </button>
      </div>
      <div class="panel-body">
        <pre class="text-out">${esc(d.full_text) || '(no text detected)'}</pre>
      </div>
    </div>
    <div class="panel">
      <div class="panel-hd">&#128288; Word Confidence
        <span style="font-size:10px;margin-left:8px">
          <span style="color:#4ade80">&#9632; &ge;80%</span>
          <span style="color:#facc15;margin-left:6px">&#9632; 55&ndash;79%</span>
          <span style="color:#f87171;margin-left:6px">&#9632; &lt;55%</span>
        </span>
      </div>
      <div class="panel-body"><div class="chips">${chips || '(none)'}</div></div>
    </div>
  </div>`;
}

function addHist(d) {
  if (history.length >= 20) history.pop();
  history.unshift(d);
  document.getElementById('hist').style.display = 'block';
  const grid = document.getElementById('hg');
  const avgPct = Math.round((d.avg_confidence || 0) * 100);
  const card = document.createElement('div');
  card.className = 'hcard';
  card.innerHTML = `
    <div class="hinfo">
      <div class="hname" title="${esc(d.filename)}">${esc(d.filename)}</div>
      <div class="hmeta">${d.total_words} words &middot; ${esc(d.engine)}</div>
      <div class="hconf ${cc(d.avg_confidence)}">${avgPct}% conf</div>
    </div>`;
  card.onclick = () => {
    document.getElementById('res').style.display = 'block';
    document.getElementById('res').innerHTML = buildHTML(d, avgPct, d.preprocessing || {});
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  grid.prepend(card);
  while (grid.children.length > 20) grid.removeChild(grid.lastChild);
}

function esc(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the web UI."""
    return _HTML

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/bmp",
    "image/tiff", "image/webp", "image/gif",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def bytes_to_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def img_to_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def process_image(img: np.ndarray) -> tuple:
    """
    Quality-check, preprocess, then OCR a single image.
    Returns (words, text, info, engine, processed_b64, quality).
    If quality['passed'] is False, words/text will be empty.
    """
    quality = quality_check(img)
    if not quality["passed"]:
        return [], "", {}, "none", img_to_b64(img), quality

    processed, info = preprocess(img)
    words, text, engine = run_ocr(processed)
    processed_b64 = img_to_b64(processed)
    quality["passed"] = True
    return words, text, info, engine, processed_b64, quality


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick health check."""
    return {"status": "ok", "service": "ocr"}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Upload an image or PDF and get back the extracted text.

    Supports: JPG, PNG, BMP, TIFF, WEBP, PDF

    Response fields:
    - full_text         -- complete extracted text
    - engine            -- which OCR engine won (Tesseract or EasyOCR)
    - rotation_applied  -- degrees rotated (0 if image was already upright)
    - total_words       -- number of words detected
    - avg_confidence    -- average detection confidence (0-1)
    - elapsed_ms        -- processing time
    - words             -- list of {text, confidence, page}
    """
    content_type = (file.content_type or "").lower()
    is_pdf = content_type == "application/pdf" or (
        file.filename or ""
    ).lower().endswith(".pdf")
    is_image = content_type in SUPPORTED_IMAGE_TYPES or any(
        (file.filename or "").lower().endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    )

    if not is_pdf and not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Upload an image or PDF.",
        )

    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}")

    t0 = time.time()
    all_words = []
    all_text_parts = []
    info_out = {"rotation_applied": 0}
    engine_used = "none"
    original_b64 = None
    processed_b64 = None
    quality_out = {"passed": True, "reason": "", "metrics": {}}

    try:
        if is_pdf:
            try:
                import fitz  # PyMuPDF
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="PDF support requires PyMuPDF: pip install pymupdf",
                )
            doc = fitz.open(stream=data, filetype="pdf")
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = bytes_to_image(pix.tobytes("png"))
                words, text, info, engine, proc_b64, quality = process_image(img)
                if page_num == 0:
                    info_out = info
                    original_b64 = img_to_b64(img)
                    processed_b64 = proc_b64
                    quality_out = quality
                if not quality["passed"]:
                    break   # reject on first bad page
                engine_used = engine
                for w in words:
                    w["page"] = page_num + 1
                all_words.extend(words)
                if text.strip():
                    all_text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        else:
            try:
                img = bytes_to_image(data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
            original_b64 = img_to_b64(img)
            words, text, info_out, engine_used, processed_b64, quality_out = process_image(img)
            if quality_out["passed"]:
                for w in words:
                    w["page"] = 1
                all_words = words
                if text.strip():
                    all_text_parts.append(text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {e}\n{traceback.format_exc()}",
        )

    elapsed = round((time.time() - t0) * 1000)
    avg_conf = (
        round(sum(w["confidence"] for w in all_words) / len(all_words), 3)
        if all_words else 0.0
    )

    return JSONResponse({
        "filename":         file.filename,
        "accepted":         quality_out["passed"],
        "rejection_reason": quality_out.get("reason", ""),
        "quality_metrics":  quality_out.get("metrics", {}),
        "engine":           engine_used,
        "elapsed_ms":       elapsed,
        "total_words":      len(all_words),
        "avg_confidence":   avg_conf,
        "rotation_applied": info_out.get("rotation_applied", 0),
        "preprocessing":    {
            "upscaled":           info_out.get("upscaled", False),
            "dark_bg_normalised": info_out.get("dark_bg_normalised", False),
            "brightness_fixed":   info_out.get("brightness_fixed", False),
            "contrast_fixed":     info_out.get("contrast_fixed", False),
        },
        "original_image":   original_b64,
        "processed_image":  processed_b64,
        "full_text": "\n\n".join(all_text_parts),
        "words":     all_words,
    })
