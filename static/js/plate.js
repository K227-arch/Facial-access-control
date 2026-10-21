/**
 * plate.js — Number Plate Detection (ANPR) UI
 * ============================================
 * Completely separate from main.js / facial recognition.
 *
 * Flow:
 *   1. User clicks "Connect" → getUserMedia opens the webcam
 *   2. Every FRAME_INTERVAL_MS a frame is captured to an off-screen canvas
 *      and POSTed as base64 JPEG to POST /api/plates/detect
 *   3. Server runs OpenCV + EasyOCR pipeline and returns detections
 *   4. drawPlateBboxes() renders amber bounding boxes + plate text on the
 *      <canvas id="anprCanvas"> overlay
 *   5. Each new unique plate is prepended to #plateLogContainer
 *   6. Stats card (#stat-plates-today) is polled from GET /api/plates/kpi
 */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const FRAME_INTERVAL_MS  = 500;   // 2 fps — OCR is heavier than face detection
const PLATE_COLOR        = '#f5a623';   // var(--neutral) amber
const MAX_LOG_ENTRIES    = 60;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const videoEl        = document.getElementById('anprVideo');
const canvasEl       = document.getElementById('anprCanvas');
const ctx            = canvasEl ? canvasEl.getContext('2d') : null;
const placeholderEl  = document.getElementById('anprPlaceholder');
const statusEl       = document.getElementById('anprStatus');
const toggleBtn      = document.getElementById('anprToggle');
const logContainer   = document.getElementById('plateLogContainer');
const emptyState     = document.getElementById('plateEmptyState');
const statPlates     = document.getElementById('stat-plates-today');
const statChange     = document.getElementById('stat-plates-change');

// ── State ─────────────────────────────────────────────────────────────────────
let stream           = null;
let isActive         = false;
let processing       = false;
let lastFrameTime    = 0;
let rafHandle        = null;
const seenPlates     = new Map();   // plateText → { count, firstSeen }
let totalPlatesSession = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(text, cssClass) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className   = `status-indicator ${cssClass}`;
}

function setButton(text, disabled = false) {
    if (!toggleBtn) return;
    toggleBtn.textContent = text;
    toggleBtn.disabled    = disabled;
}

function showPlaceholder(visible) {
    if (!placeholderEl) return;
    placeholderEl.style.display = visible ? 'flex' : 'none';
}

function showEmptyState(visible) {
    if (!emptyState)     emptyState && (emptyState.style.display = 'none');
    if (emptyState)      emptyState.style.display = visible ? 'flex' : 'none';
    if (logContainer)    logContainer.style.display = visible ? 'none' : 'flex';
}

// ── Canvas / drawing ─────────────────────────────────────────────────────────

function syncCanvas() {
    if (!canvasEl || !videoEl) return;
    canvasEl.width  = videoEl.videoWidth  || 1280;
    canvasEl.height = videoEl.videoHeight || 720;
    // Remove inline size so CSS 100% takes over for display scaling
    canvasEl.style.width  = '';
    canvasEl.style.height = '';
}

function drawPlateBboxes(detections) {
    if (!ctx || !canvasEl) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!detections || !detections.length) return;

    // Display ratio: buffer (native) → screen (CSS display size)
    const displayW      = canvasEl.offsetWidth  || canvasEl.width;
    const displayRatioX = canvasEl.width  / displayW;
    const displayRatioY = canvasEl.height / (canvasEl.offsetHeight || canvasEl.height);

    const SCREEN_LINE   = 2.5;
    const SCREEN_FONT   = 15;
    const SCREEN_PAD    = 7;
    const SCREEN_CORNER = 14;

    const bufLine   = Math.max(1.5, SCREEN_LINE   * displayRatioX);
    const bufFont   = Math.round(SCREEN_FONT  * displayRatioX);
    const bufPad    = Math.round(SCREEN_PAD   * displayRatioX);
    const bufCorner = Math.round(SCREEN_CORNER * displayRatioX);
    const bufRadius = Math.round(4 * displayRatioX);

    ctx.font      = `bold ${bufFont}px "Courier New", monospace`;
    ctx.lineWidth = bufLine;

    detections.forEach(det => {
        if (!det.bbox) return;
        const [x1, y1, x2, y2] = det.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;

        // ── Box ──
        ctx.strokeStyle = PLATE_COLOR;
        ctx.strokeRect(x1, y1, bw, bh);

        // ── Corner accents ──
        const cs = Math.min(bufCorner, bw * 0.25, bh * 0.25);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = bufLine * 1.8;
        ctx.lineCap     = 'round';
        for (const [ox, oy, dx, dy] of [
            [x1,      y1,      1,  1],
            [x1 + bw, y1,     -1,  1],
            [x1,      y1 + bh, 1, -1],
            [x1 + bw, y1 + bh,-1, -1],
        ]) {
            ctx.beginPath();
            ctx.moveTo(ox + dx * cs, oy);
            ctx.lineTo(ox, oy);
            ctx.lineTo(ox, oy + dy * cs);
            ctx.stroke();
        }
        ctx.lineCap   = 'butt';
        ctx.lineWidth = bufLine;

        // ── Label chip ──
        const plateText = det.plate_text || det.label || '???';
        const conf      = Math.round((det.confidence || 0) * 100);
        const topLine   = plateText;
        const botLine   = `PLATE  ${conf}%`;

        ctx.font         = `bold ${bufFont}px "Courier New", monospace`;
        const tw1        = ctx.measureText(topLine).width;
        ctx.font         = `${Math.round(bufFont * 0.82)}px "Segoe UI", sans-serif`;
        const tw2        = ctx.measureText(botLine).width;

        const chipW  = Math.max(tw1, tw2) + bufPad * 2;
        const lineH  = bufFont + bufPad;
        const chipH  = lineH * 2 + bufPad;
        const chipX  = Math.min(x1, canvasEl.width - chipW);
        const chipY  = y1 >= chipH + bufPad ? y1 - chipH - bufPad : y1 + bufPad;

        // Background
        ctx.globalAlpha = 0.9;
        ctx.fillStyle   = 'rgba(10,14,24,0.92)';
        ctx.beginPath();
        ctx.roundRect
            ? ctx.roundRect(chipX, chipY, chipW, chipH, bufRadius)
            : ctx.rect(chipX, chipY, chipW, chipH);
        ctx.fill();

        // Amber stripe
        ctx.globalAlpha = 1;
        const stripeW   = Math.max(3, Math.round(4 * displayRatioX));
        ctx.fillStyle   = PLATE_COLOR;
        ctx.beginPath();
        ctx.roundRect
            ? ctx.roundRect(chipX, chipY, stripeW, chipH, [bufRadius, 0, 0, bufRadius])
            : ctx.rect(chipX, chipY, stripeW, chipH);
        ctx.fill();

        // Plate text (monospace, white)
        ctx.fillStyle    = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.font         = `bold ${bufFont}px "Courier New", monospace`;
        ctx.fillText(topLine, chipX + stripeW + bufPad, chipY + lineH * 0.5);

        // Conf line (amber)
        ctx.fillStyle = PLATE_COLOR;
        ctx.font      = `${Math.round(bufFont * 0.82)}px "Segoe UI", sans-serif`;
        ctx.fillText(botLine, chipX + stripeW + bufPad, chipY + lineH * 1.5);

        ctx.globalAlpha = 1;
    });
}

// ── Log panel ─────────────────────────────────────────────────────────────────

function addPlateLogEntry(det) {
    const plate = det.plate_text || det.label || '???';
    const conf  = Math.round((det.confidence || 0) * 100);
    const now   = new Date();
    const time  = now.toLocaleTimeString();

    // Dedup: update count if seen before, otherwise create new entry
    if (seenPlates.has(plate)) {
        const info = seenPlates.get(plate);
        info.count++;
        const el = document.getElementById(`plate-entry-${CSS.escape(plate)}`);
        if (el) {
            el.querySelector('.plate-conf').textContent = `${conf}%`;
            el.querySelector('.plate-meta').textContent =
                `${time}  ·  seen ${info.count}×`;
        }
        return;
    }

    // New plate
    seenPlates.set(plate, { count: 1, firstSeen: time });
    totalPlatesSession++;
    if (statPlates) statPlates.textContent = totalPlatesSession;

    showEmptyState(false);

    const item = document.createElement('div');
    item.className = 'plate-item';
    item.id        = `plate-entry-${CSS.escape(plate)}`;
    item.innerHTML = `
        <i class="bi bi-car-front-fill plate-item-icon"></i>
        <div class="plate-item-body">
            <div class="plate-text">${plate}</div>
            <div class="plate-meta">${time}  ·  seen 1×</div>
        </div>
        <span class="plate-conf">${conf}%</span>`;

    logContainer.prepend(item);

    // Cap DOM entries
    while (logContainer.children.length > MAX_LOG_ENTRIES) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

function clearPlateLog() {
    seenPlates.clear();
    totalPlatesSession = 0;
    if (statPlates) statPlates.textContent = '0';
    if (logContainer) logContainer.innerHTML = '';
    showEmptyState(true);
}

// ── Frame capture + API call ──────────────────────────────────────────────────

function captureFrame() {
    const fc  = document.createElement('canvas');
    fc.width  = videoEl.videoWidth;
    fc.height = videoEl.videoHeight;
    fc.getContext('2d').drawImage(videoEl, 0, 0);
    return fc.toDataURL('image/jpeg', 0.7);
}

async function processFrame() {
    if (processing || !videoEl || videoEl.readyState !== 4) return;
    processing = true;
    try {
        const frameData = captureFrame();
        const resp = await fetch('/api/plates/detect', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ image: frameData, camera_id: 'ANPR-01' }),
        });
        if (!resp.ok) return;
        const results = await resp.json();
        if (!Array.isArray(results)) return;

        // Sync canvas buffer to video native size (if changed)
        if (canvasEl.width !== videoEl.videoWidth ||
            canvasEl.height !== videoEl.videoHeight) {
            syncCanvas();
        }
        drawPlateBboxes(results);
        results.forEach(det => addPlateLogEntry(det));
    } catch (e) {
        console.warn('[ANPR] Frame error:', e);
    } finally {
        processing = false;
    }
}

function processingLoop() {
    if (!isActive) return;
    const now = Date.now();
    if (now - lastFrameTime >= FRAME_INTERVAL_MS) {
        processFrame();
        lastFrameTime = now;
    }
    rafHandle = requestAnimationFrame(processingLoop);
}

// ── Camera lifecycle ──────────────────────────────────────────────────────────

async function anprConnect() {
    if (isActive) return;
    setButton('Connecting…', true);
    setStatus('CONNECTING', 'status-connecting');

    try {
        // Permission probe
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach(t => t.stop());

        const devices   = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devices.filter(d => d.kind === 'videoinput');
        if (!videoDevs.length) throw new Error('No video devices found');

        // Use the first available device (separate from the face webcam slot)
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { ideal: videoDevs[0].deviceId },
                width:  { ideal: 1280 },
                height: { ideal: 720 },
            },
        });

        videoEl.srcObject = stream;
        await new Promise((res, rej) => {
            videoEl.onloadedmetadata = () => {
                videoEl.style.display = 'block';
                syncCanvas();
                res();
            };
            videoEl.onerror = rej;
        });

        isActive = true;
        showPlaceholder(false);
        videoEl.play();
        setStatus('LIVE', 'status-live');
        setButton('Disconnect', false);
        processingLoop();

    } catch (err) {
        console.error('[ANPR] Connect error:', err);
        let msg = 'ERROR';
        if (err.name === 'NotAllowedError')   msg = 'NO PERMISSION';
        else if (err.name === 'NotFoundError') msg = 'NOT FOUND';
        else if (err.name === 'NotReadableError') msg = 'IN USE';
        setStatus(msg, 'status-error');
        setButton('Retry', false);
        isActive = false;
    }
}

function anprDisconnect() {
    isActive = false;
    if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = null; }
    if (stream)    { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (videoEl)   { videoEl.srcObject = null; videoEl.style.display = 'none'; }
    if (ctx)       { ctx.clearRect(0, 0, canvasEl.width, canvasEl.height); }
    showPlaceholder(true);
    setStatus('OFFLINE', 'status-offline');
    setButton('Connect', false);
}

// Called by the HTML onclick attribute
function anprToggle() {
    if (isActive) anprDisconnect();
    else          anprConnect();
}

// ── Stats card polling ────────────────────────────────────────────────────────

async function refreshPlateKpi() {
    try {
        const resp = await fetch('/api/plates/kpi?range=today');
        if (!resp.ok) return;
        const data = await resp.json();
        if (statPlates && !isActive) {
            // Only update from server when camera is off;
            // when live, the session counter is already shown
            statPlates.textContent = data.count ?? 0;
        }
        if (statChange) {
            statChange.textContent = `${data.count ?? 0} plates today`;
            statChange.className   = 'stat-change' + (data.count > 0 ? ' positive' : '');
        }
    } catch (_) {}
}

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    showEmptyState(true);
    refreshPlateKpi();
    // Refresh KPI every 30 s
    setInterval(refreshPlateKpi, 30_000);
});
