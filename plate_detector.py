"""
plate_detector.py
=================
Automatic Number Plate Recognition (ANPR) engine.

Pipeline
--------
1. Pre-process the frame (grayscale → bilateral filter → Canny edges)
2. Find contour candidates whose geometry strictly matches a number plate
   (aspect ratio 2–6 : 1, minimum area, near-rectangular 4-corner shape)
3. Multi-pass OCR: run EasyOCR on the raw crop AND a thresholded version
4. Filter OCR output through a strict plate-text regex
5. Return only confirmed plate detections — no generic text regions

This module is completely isolated from the facial recognition system.
No shared state, no shared imports.
"""

import re
import logging
import threading
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Plate geometry constraints ────────────────────────────────────────────────
# A real plate is wider than it is tall, with aspect ratio between 2 and 6.
# Anything outside this range is not a plate.
PLATE_RATIO_MIN  = 1.8
PLATE_RATIO_MAX  = 6.5
PLATE_MIN_WIDTH  = 60     # pixels in original frame
PLATE_MAX_WIDTH_FRAC = 0.70   # no plate can be >70% of the frame width
PLATE_MIN_HEIGHT = 18
PLATE_MAX_HEIGHT_FRAC = 0.25  # no plate taller than 25% of frame height

# ── OCR quality thresholds ────────────────────────────────────────────────────
_MIN_CONF        = 0.35   # EasyOCR confidence — higher than before
_MIN_PLATE_CHARS = 4      # after removing spaces/hyphens

# ── Plate text regex — strict ─────────────────────────────────────────────────
# Must contain at least one digit AND at least one letter.
# Matches common formats: AB12CDE  AB 12 CDE  ABC-1234  1234-ABC  etc.
# Does NOT match pure words or pure numbers (would create false positives).
_PLATE_RE = re.compile(
    r'^[A-Z0-9](?:[A-Z0-9\s\-]{2,9})[A-Z0-9]$',
    re.IGNORECASE
)

def _has_letter_and_digit(text: str) -> bool:
    s = text.replace(' ', '').replace('-', '')
    return any(c.isalpha() for c in s) and any(c.isdigit() for c in s)

# ── EasyOCR lazy singleton ────────────────────────────────────────────────────
_reader      = None
_reader_lock = threading.Lock()

def _get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                logger.info("Loading EasyOCR reader…")
                _reader = easyocr.Reader(
                    ['en'], gpu=False, verbose=False,
                    # Disable paragraph detection — we want individual text blocks
                    # so a single plate line doesn't get merged with surroundings
                )
                logger.info("EasyOCR ready.")
    return _reader


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.bilateralFilter(gray, 13, 15, 15)
    return cv2.Canny(blurred, 25, 180)


def _find_plate_candidates(frame: np.ndarray) -> list[tuple]:
    """
    Return (x1, y1, x2, y2) boxes that geometrically look like plates.
    Uses two passes: 4-corner approximation on Canny + morphological
    rectangle detector for plates with rounded corners or slight perspective.
    """
    h_frame, w_frame = frame.shape[:2]
    min_area  = (w_frame * h_frame) * 0.0012
    max_area  = (w_frame * h_frame) * 0.20

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = _edges(gray)

    # Pass 1: 4-corner contour approach
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    candidates = []
    seen = set()

    def _add(x, y, w, h):
        """Add candidate after strict geometry check."""
        if w < PLATE_MIN_WIDTH or h < PLATE_MIN_HEIGHT:
            return
        if w > w_frame * PLATE_MAX_WIDTH_FRAC:
            return
        if h > h_frame * PLATE_MAX_HEIGHT_FRAC:
            return
        ratio = w / max(h, 1)
        if not (PLATE_RATIO_MIN <= ratio <= PLATE_RATIO_MAX):
            return
        key = (x // 10, y // 10)   # deduplicate near-identical boxes
        if key in seen:
            return
        seen.add(key)
        candidates.append((x, y, x + w, y + h))

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.016 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            _add(x, y, w, h)
        elif 4 < len(approx) <= 8:
            # Slightly imperfect rectangle (slight perspective, dents) — still try
            x, y, w, h = cv2.boundingRect(approx)
            _add(x, y, w, h)

    # Pass 2: morphological rectangle detector catches plates missed by contours
    # (white text on dark background, dirty plates, partial occlusion)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph    = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cnts2, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts2:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        _add(x, y, w, h)

    return candidates


def _enhance_crop(frame: np.ndarray, x1, y1, x2, y2):
    """
    Return two versions of the plate crop for dual-pass OCR:
    - Version A: CLAHE-enhanced grayscale (good for dark plates)
    - Version B: Adaptive threshold binary (good for light plates with dark text)
    """
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    # Upscale to at least 240px wide so OCR has enough resolution
    h, w = crop.shape[:2]
    target_w = max(w, 240)
    scale    = target_w / w
    crop     = cv2.resize(crop, (target_w, int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Version A: CLAHE
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    ver_a  = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    ver_a  = cv2.filter2D(ver_a, -1, kernel)

    # Version B: Otsu threshold for high-contrast plates
    _, ver_b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ver_a, ver_b


def _clean(raw: str) -> str:
    """Strip OCR noise; keep only plate-safe characters."""
    s = re.sub(r'[^A-Z0-9\- ]', '', raw.upper()).strip()
    return re.sub(r'\s{2,}', ' ', s)


def _is_plate(text: str) -> bool:
    """True only if text passes geometry + content checks."""
    body = text.replace(' ', '').replace('-', '')
    if len(body) < _MIN_PLATE_CHARS or len(body) > 12:
        return False
    if not _has_letter_and_digit(text):
        return False
    return bool(_PLATE_RE.match(text.strip()))


# ── Public API ────────────────────────────────────────────────────────────────

def detect_plates(frame: np.ndarray) -> list[dict]:
    """
    Run the full ANPR pipeline on a BGR frame.

    Returns list of:
    {
      'bbox':       [x1, y1, x2, y2],
      'plate_text': str,
      'confidence': float,
      'status':     'detected',
      'label':      str,   # same as plate_text (for generic drawBoxes)
      'color':      '#f5a623',
      'track_id':   int,
    }
    Only genuine number plates are returned — false positives from random
    rectangular objects are suppressed by the letter+digit requirement.
    """
    candidates = _find_plate_candidates(frame)
    if not candidates:
        return []

    reader   = _get_reader()
    h_max, w_max = frame.shape[:2]
    results  = []
    seen     = set()

    for idx, (x1, y1, x2, y2) in enumerate(candidates):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_max, x2), min(h_max, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        ver_a, ver_b = _enhance_crop(frame, x1, y1, x2, y2)
        if ver_a is None:
            continue

        best_text = None
        best_conf = 0.0

        # Run OCR on both versions, keep best-confidence valid result
        for img in (ver_a, ver_b):
            try:
                for (_, raw, conf) in reader.readtext(img, detail=1,
                                                       paragraph=False):
                    if conf < _MIN_CONF:
                        continue
                    cleaned = _clean(raw)
                    if not _is_plate(cleaned):
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
            except Exception as exc:
                logger.debug("EasyOCR error: %s", exc)

        if best_text is None or best_text in seen:
            continue

        seen.add(best_text)
        results.append({
            'bbox':       [float(x1), float(y1), float(x2), float(y2)],
            'plate_text': best_text,
            'confidence': round(best_conf, 3),
            'status':     'detected',
            'label':      best_text,
            'color':      '#f5a623',
            'track_id':   idx + 1,
        })

    return results
