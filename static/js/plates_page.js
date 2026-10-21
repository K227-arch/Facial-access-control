/**
 * plates_page.js — Dedicated Number Plate Detection Page
 * =======================================================
 * Completely independent of main.js and the facial recognition system.
 *
 * Responsibilities:
 *   - Open the webcam via getUserMedia
 *   - Capture frames at 2 fps → POST /api/plates/detect
 *   - Draw amber bounding boxes + plate text on <canvas id="plateCanvas">
 *   - Populate the live log panel with new detections (deduplicated)
 *   - Load detection history from GET /api/plates/records
 *   - Poll GET /api/plates/kpi for the "Today" stat card
 */

'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const FRAME_MS   = 500;    // 2 fps — OCR is expensive
const PLATE_CLR  = '#f5a623';
const MAX_LOG    = 80;

// ── DOM ───────────────────────────────────────────────────────────────────────
const videoEl      = document.getElementById('plateVideo');
const canvasEl     = document.getElementById('plateCanvas');
const ctx          = canvasEl?.getContext('2d');
const offlineEl    = document.getElementById('plateOffline');
const scanLine     = document.getElementById('plateScanLine');
const camStatusEl  = document.getElementById('plateCamStatus');
const camBtnEl     = document.getElementById('plateCamBtn');
const snapBtnEl    = document.getElementById('plateSnapBtn');
const logContainer = document.getElementById('plateLogContainer');
const emptyState   = document.getElementById('plateEmptyState');
const lastCard     = document.getElementById('lastPlateCard');
const lastValue    = document.getElementById('lastPlateValue');
const lastMeta     = document.getElementById('lastPlateMeta');
const lastScanEl   = document.getElementById('lastScanTime');

const statSession  = document.getElementById('statSession');
const statUnique   = document.getElementById('statUnique');
const statToday    = document.getElementById('statToday');
const statCam      = document.getElementById('statCamStatus');
const statCamNote  = document.getElementById('statCamNote');

// ── State ─────────────────────────────────────────────────────────────────────
let stream        = null;
let isActive      = false;
let processing    = false;
let lastFrameT    = 0;
let rafId         = null;

let sessionCount  = 0;
const seenPlates  = new Map();   // plateText → { count, firstSeen }

// ── Camera helpers ────────────────────────────────────────────────────────────

function setStatus(text, cls) {
    if (camStatusEl) { camStatusEl.textContent = text; camStatusEl.className = `status-indicator ${cls}`; }
    if (statCam)     { statCam.textContent = text; }
    if (statCamNote) { statCamNote.textContent = cls === 'status-live' ? 'Scanning live' : 'Not connected'; }
}

function setBtn(text, activeClass = false) {
    if (!camBtnEl) return;
    camBtnEl.textContent = '';
    const icon = document.createElement('i');
    icon.className = activeClass ? 'bi bi-camera-video-off-fill' : 'bi bi-camera-video-fill';
    camBtnEl.appendChild(icon);
    camBtnEl.appendChild(document.createTextNode(' ' + text));
    camBtnEl.classList.toggle('active', activeClass);
}

function showOffline(v) {
    if (offlineEl) offlineEl.style.display = v ? 'flex' : 'none';
}

function showScanLine(v) {
    if (scanLine) scanLine.classList.toggle('scanning', v);
}

// ── Canvas ────────────────────────────────────────────────────────────────────

function syncCanvas() {
    if (!canvasEl || !videoEl) return;
    canvasEl.width  = videoEl.videoWidth  || 1280;
    canvasEl.height = videoEl.videoHeight || 720;
    canvasEl.style.width  = '';
    canvasEl.style.height = '';
}

function drawBoxes(detections) {
    if (!ctx || !canvasEl) return;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!detections?.length) return;

    const dW  = canvasEl.offsetWidth  || canvasEl.width;
    const dH  = canvasEl.offsetHeight || canvasEl.height;
    const rx  = canvasEl.width  / dW;
    const ry  = canvasEl.height / dH;

    const bufLine   = Math.max(1.5, 2.5  * rx);
    const bufFont   = Math.round(14 * rx);
    const bufPad    = Math.round(7  * rx);
    const bufCorner = Math.round(13 * rx);
    const bufR      = Math.round(4  * rx);

    detections.forEach((det, i) => {
        if (!det.bbox) return;
        const [x1, y1, x2, y2] = det.bbox;
        const bw = x2 - x1, bh = y2 - y1;

        // Box
        ctx.strokeStyle = PLATE_CLR;
        ctx.lineWidth   = bufLine;
        ctx.strokeRect(x1, y1, bw, bh);

        // Corner accents
        const cs = Math.min(bufCorner, bw * 0.22, bh * 0.22);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = bufLine * 1.8;
        ctx.lineCap     = 'round';
        for (const [ox, oy, dx, dy] of [
            [x1, y1, 1, 1], [x1+bw, y1, -1, 1],
            [x1, y1+bh, 1, -1], [x1+bw, y1+bh, -1, -1]
        ]) {
            ctx.beginPath();
            ctx.moveTo(ox + dx*cs, oy); ctx.lineTo(ox, oy); ctx.lineTo(ox, oy + dy*cs);
            ctx.stroke();
        }
        ctx.lineCap   = 'butt';
        ctx.lineWidth = bufLine;

        // Label chip
        const plate   = det.plate_text || '???';
        const conf    = Math.round((det.confidence || 0) * 100);
        const topLine = plate;
        const botLine = `${conf}% CONFIDENCE`;

        ctx.font  = `bold ${bufFont}px "Courier New", monospace`;
        const tw1 = ctx.measureText(topLine).width;
        ctx.font  = `${Math.round(bufFont*0.78)}px "Segoe UI", sans-serif`;
        const tw2 = ctx.measureText(botLine).width;

        const cW   = Math.max(tw1, tw2) + bufPad * 2;
        const lH   = bufFont + bufPad;
        const cH   = lH * 2 + bufPad;
        const cX   = Math.min(x1, canvasEl.width - cW);
        const cY   = y1 >= cH + bufPad ? y1 - cH - bufPad : y1 + bufPad;

        ctx.globalAlpha = 0.9;
        ctx.fillStyle   = 'rgba(10,14,24,0.92)';
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(cX, cY, cW, cH, bufR) : ctx.rect(cX, cY, cW, cH);
        ctx.fill();

        ctx.globalAlpha = 1;
        const sw = Math.max(3, Math.round(4*rx));
        ctx.fillStyle = PLATE_CLR;
        ctx.beginPath();
        ctx.roundRect
            ? ctx.roundRect(cX, cY, sw, cH, [bufR,0,0,bufR])
            : ctx.rect(cX, cY, sw, cH);
        ctx.fill();

        ctx.fillStyle    = '#fff';
        ctx.textBaseline = 'middle';
        ctx.font  = `bold ${bufFont}px "Courier New", monospace`;
        ctx.fillText(topLine, cX + sw + bufPad, cY + lH*0.5);

        ctx.fillStyle = PLATE_CLR;
        ctx.font      = `${Math.round(bufFont*0.78)}px "Segoe UI", sans-serif`;
        ctx.fillText(botLine, cX + sw + bufPad, cY + lH*1.5);

        ctx.globalAlpha = 1;
    });
}

// ── Log panel ─────────────────────────────────────────────────────────────────

function addLogEntry(det) {
    const plate = det.plate_text || '???';
    const conf  = Math.round((det.confidence || 0) * 100);
    const time  = new Date().toLocaleTimeString();

    if (seenPlates.has(plate)) {
        // Update existing entry
        const info = seenPlates.get(plate);
        info.count++;
        const el = document.getElementById(`plg-${CSS.escape(plate)}`);
        if (el) {
            el.querySelector('.plate-log-conf').textContent = `${conf}%`;
            el.querySelector('.plate-log-meta').textContent = `${time}  ·  ${info.count} times`;
        }
        return;
    }

    // New plate
    seenPlates.set(plate, { count: 1, firstSeen: time });
    sessionCount++;
    if (statSession) statSession.textContent = sessionCount;
    if (statUnique)  statUnique.textContent  = seenPlates.size;

    // Hide empty state
    if (emptyState)   emptyState.style.display = 'none';

    // Last plate card
    if (lastCard)  lastCard.style.display  = 'block';
    if (lastValue) lastValue.textContent  = plate;
    if (lastMeta)  lastMeta.textContent   = `${conf}% confidence · ${time}`;
    if (lastScanEl) lastScanEl.textContent = `Last scan: ${time}`;

    const item = document.createElement('div');
    item.className = 'plate-log-item';
    item.id        = `plg-${CSS.escape(plate)}`;
    item.innerHTML = `
        <i class="bi bi-car-front-fill plate-log-icon"></i>
        <div class="plate-log-body-text">
            <div class="plate-log-text">${plate}</div>
            <div class="plate-log-meta">${time}  ·  1 time</div>
        </div>
        <span class="plate-log-conf">${conf}%</span>`;

    logContainer.prepend(item);
    while (logContainer.children.length > MAX_LOG) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

function clearPlateLog() {
    seenPlates.clear();
    sessionCount = 0;
    if (statSession) statSession.textContent = '0';
    if (statUnique)  statUnique.textContent  = '0';
    if (logContainer) {
        logContainer.innerHTML = '';
        if (emptyState) {
            emptyState.style.display = 'flex';
            logContainer.appendChild(emptyState);
        }
    }
    if (lastCard) lastCard.style.display = 'none';
}

// ── Frame processing ──────────────────────────────────────────────────────────

function captureFrame() {
    const fc = document.createElement('canvas');
    fc.width  = videoEl.videoWidth;
    fc.height = videoEl.videoHeight;
    fc.getContext('2d').drawImage(videoEl, 0, 0);
    return fc.toDataURL('image/jpeg', 0.75);
}

async function processFrame() {
    if (processing || !videoEl || videoEl.readyState !== 4) return;
    processing = true;
    try {
        const data = captureFrame();
        const resp = await fetch('/api/plates/detect', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ image: data, camera_id: 'ANPR-01' }),
        });
        if (!resp.ok) return;
        const results = await resp.json();
        if (!Array.isArray(results)) return;

        if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
            syncCanvas();
        }
        drawBoxes(results);
        results.forEach(addLogEntry);

        const ts = new Date().toLocaleTimeString();
        if (lastScanEl && !results.length) lastScanEl.textContent = `Last scan: ${ts}`;
    } catch (e) {
        console.warn('[ANPR] Frame error:', e);
    } finally {
        processing = false;
    }
}

function loop() {
    if (!isActive) return;
    const now = Date.now();
    if (now - lastFrameT >= FRAME_MS) {
        processFrame();
        lastFrameT = now;
    }
    rafId = requestAnimationFrame(loop);
}

// ── Camera connect / disconnect ───────────────────────────────────────────────

async function plateConnect() {
    if (isActive) return;
    setBtn('Connecting…', false);
    setStatus('CONNECTING', 'status-connecting');
    camBtnEl && (camBtnEl.disabled = true);

    try {
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach(t => t.stop());

        const devs = (await navigator.mediaDevices.enumerateDevices())
            .filter(d => d.kind === 'videoinput');
        if (!devs.length) throw new Error('No camera found');

        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { ideal: devs[0].deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
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
        showOffline(false);
        showScanLine(true);
        videoEl.play();
        setStatus('LIVE', 'status-live');
        setBtn('Disconnect', true);
        if (snapBtnEl) snapBtnEl.disabled = false;
        if (camBtnEl)  camBtnEl.disabled  = false;
        loop();

    } catch (err) {
        console.error('[ANPR] Connect error:', err);
        let msg = 'ERROR';
        if (err.name === 'NotAllowedError')   msg = 'NO PERMISSION';
        else if (err.name === 'NotFoundError') msg = 'NOT FOUND';
        else if (err.name === 'NotReadableError') msg = 'IN USE';
        setStatus(msg, 'status-error');
        setBtn('Retry', false);
        if (camBtnEl) camBtnEl.disabled = false;
    }
}

function plateDisconnect() {
    isActive = false;
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (videoEl) { videoEl.srcObject = null; videoEl.style.display = 'none'; }
    if (ctx) ctx.clearRect(0, 0, canvasEl?.width || 0, canvasEl?.height || 0);
    showOffline(true);
    showScanLine(false);
    setStatus('OFFLINE', 'status-offline');
    setBtn('Connect', false);
    if (snapBtnEl) snapBtnEl.disabled = true;
}

// Called by HTML onclick
function plateToggle() {
    if (isActive) plateDisconnect();
    else          plateConnect();
}

// ── Snapshot ──────────────────────────────────────────────────────────────────

function plateSnapshot() {
    if (!isActive || !videoEl) return;
    const fc = document.createElement('canvas');
    fc.width  = videoEl.videoWidth;
    fc.height = videoEl.videoHeight;
    const c   = fc.getContext('2d');
    c.drawImage(videoEl, 0, 0);
    // Composite canvas overlay on snapshot
    if (canvasEl) c.drawImage(canvasEl, 0, 0, fc.width, fc.height);

    const a = document.createElement('a');
    a.href     = fc.toDataURL('image/png');
    a.download = `plate-snapshot-${Date.now()}.png`;
    a.click();
}

// ── History table ─────────────────────────────────────────────────────────────

async function loadHistory() {
    const range  = document.getElementById('historyRange')?.value || 'all';
    const tbody  = document.getElementById('plateHistoryBody');
    const footer = document.getElementById('historyCount');
    if (!tbody) return;

    tbody.innerHTML = '<tr class="table-empty-row"><td colspan="5">Loading…</td></tr>';

    try {
        const resp = await fetch(`/api/plates/records?range=${range}`);
        if (!resp.ok) throw new Error(resp.statusText);
        const records = await resp.json();

        if (footer) footer.textContent = `${records.length} record${records.length !== 1 ? 's' : ''}`;

        if (!records.length) {
            tbody.innerHTML = '<tr class="table-empty-row"><td colspan="5">No records found for this range</td></tr>';
            return;
        }

        tbody.innerHTML = records.map(r => {
            const [date, time] = (r.Timestamp || '').split(' ');
            const conf = Math.round((r.Confidence || 0) * 100);
            return `
            <tr>
                <td class="plate-cell">${r.PlateText || '—'}</td>
                <td class="conf-cell">${conf}%</td>
                <td>${r.CameraID || '—'}</td>
                <td>${time || '—'}</td>
                <td>${date || '—'}</td>
            </tr>`;
        }).join('');

    } catch (e) {
        tbody.innerHTML = `<tr class="table-empty-row"><td colspan="5">Error loading records: ${e.message}</td></tr>`;
    }
}

// ── KPI polling ───────────────────────────────────────────────────────────────

async function refreshKpi() {
    try {
        const r = await fetch('/api/plates/kpi?range=today');
        if (!r.ok) return;
        const d = await r.json();
        if (statToday) statToday.textContent = d.count ?? 0;
    } catch (_) {}
}

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    refreshKpi();
    setInterval(refreshKpi, 30_000);
    setInterval(loadHistory, 60_000);   // auto-refresh history every minute
});
