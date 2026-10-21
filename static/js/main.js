/**
 * main.js — Unified camera system for Facial Access Control
 *
 * Supports two stream modes per camera, set by the server config:
 *
 *   "rtsp"   — The server reads the Hikvision RTSP stream via OpenCV/FFmpeg
 *              and serves it as a MJPEG stream at /api/cameras/stream/<id>.
 *              The browser displays it with a plain <img> tag (no WebRTC, no
 *              getUserMedia required).  Detections are polled separately from
 *              /api/cameras/<id>/detections and drawn on the canvas overlay.
 *
 *   "webcam" — The browser opens getUserMedia, captures frames, POSTs them as
 *              base64 JPEG to /getdata/, and draws bounding boxes on the canvas.
 *              This is the original mode kept for local USB cameras.
 *
 * On page load, JS fetches /api/cameras/config, then for each cam-card whose
 * data-cam-id matches a config entry it creates the right CameraSystem subclass.
 */

'use strict';

// ---------------------------------------------------------------------------
// Global detection state (shared across all cameras)
// ---------------------------------------------------------------------------
const detectionsContainer = document.getElementById('detectionsContainer');

// cam-scoped key: `${cam_id}:${track_id}` → shown flag
const trackHistory = new Map();

let detectionCount  = 0;
let recognizedCount = 0;
let unknownCount    = 0;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function updateStatElements() {
    const map = {
        '.card:nth-child(2) .stat-number': detectionCount,
        '.card:nth-child(3) .stat-number': unknownCount,
        '.card:nth-child(4) .stat-number': recognizedCount,
    };
    for (const [sel, val] of Object.entries(map)) {
        const el = document.querySelector(sel);
        if (el) el.textContent = val;
    }
}

function createDetectionEntry(detection, camLabel) {
    const timeStr = detection.timestamp
        ? new Date(detection.timestamp).toLocaleTimeString()
        : new Date().toLocaleTimeString();

    const item = document.createElement('div');
    item.className = 'det-item';
    item.innerHTML = `
        <div class="det-item-name">
            <i class="fas fa-video det-icon"></i>
            <div>
                <strong>${detection.label || 'Unknown'}</strong>
                <small>${camLabel} — ${timeStr}</small>
            </div>
        </div>
        <div class="det-det">
            <div class="det-status ${detection.status || 'unknown'}">
                ${(detection.status || 'Unknown').toUpperCase()}
            </div>
            <small>${Math.round((detection.confidence || 0) * 100)}% match</small>
        </div>`;

    detectionsContainer.prepend(item);

    // Keep at most 80 entries in the DOM
    while (detectionsContainer.children.length > 80) {
        detectionsContainer.removeChild(detectionsContainer.lastChild);
    }
}

/**
 * Process a detections array and update the sidebar + stats.
 * @param {Array}  results    - Array of detection objects from /getdata/ or polling
 * @param {number} camId      - Numeric camera id (0-based)
 * @param {string} camLabel   - Human label e.g. "CAM-01"
 */
function processDetectionResults(results, camId, camLabel) {
    if (!Array.isArray(results)) return;

    results.forEach(result => {
        if (!result.track_id) return;

        const key = `${camId}:${result.track_id}`;

        if (!trackHistory.has(key)) {
            trackHistory.set(key, { shown: false, status: result.status });

            detectionCount++;
            if (result.status === 'recognized') recognizedCount++;
            else unknownCount++;
        }

        const info = trackHistory.get(key);
        if (!info.shown) {
            createDetectionEntry(result, camLabel);
            info.shown = true;
            updateStatElements();
        }
    });
}

// ---------------------------------------------------------------------------
// Canvas bounding-box renderer (shared between both modes)
// ---------------------------------------------------------------------------
function drawBoxes(ctx, canvas, detections, refWidth, refHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!detections || !detections.length) return;

    // Coordinate transform: source pixels → canvas buffer pixels.
    // For webcam mode canvas.width === refWidth, so scaleX === 1.0.
    const scaleX = canvas.width  / (refWidth  || canvas.width);
    const scaleY = canvas.height / (refHeight || canvas.height);

    // ── Display-aware sizing ──────────────────────────────────────────────
    // The canvas buffer is native resolution (e.g. 1280×720) but it is
    // displayed CSS-scaled down to the card width (e.g. ~450px).
    // Using buffer coords directly gives tiny text/lines on screen.
    // We compute a displayRatio = buffer / display so every px value we
    // write is bumped up proportionally and looks right at screen size.
    const displayW   = canvas.offsetWidth  || canvas.width;   // CSS display px
    const displayH   = canvas.offsetHeight || canvas.height;
    const displayRatioX = canvas.width  / displayW;   // e.g. 1280/450 ≈ 2.84
    const displayRatioY = canvas.height / displayH;

    // Target sizes in *screen* pixels, then scale up to buffer coords:
    const SCREEN_FONT_PX  = 16;   // readable label text (screen px)
    const SCREEN_LINE_PX  = 2.5;  // box stroke (screen px)
    const SCREEN_PAD_PX   = 8;    // chip horizontal padding (screen px)
    const SCREEN_VPAD_PX  = 6;    // chip vertical padding (screen px)
    const SCREEN_CORNER   = 14;   // corner accent length (screen px)

    const bufFontPx   = Math.round(SCREEN_FONT_PX  * displayRatioX);
    const bufLineW    = Math.max(1.5, SCREEN_LINE_PX  * displayRatioX);
    const bufPadH     = Math.round(SCREEN_PAD_PX   * displayRatioX);
    const bufPadV     = Math.round(SCREEN_VPAD_PX  * displayRatioY);
    const bufCorner   = Math.round(SCREEN_CORNER   * displayRatioX);
    const bufRadius   = Math.round(5 * displayRatioX);   // chip corner radius

    ctx.font      = `bold ${bufFontPx}px "Segoe UI", Arial, sans-serif`;
    ctx.lineWidth = bufLineW;

    detections.forEach(det => {
        if (!det.bbox) return;
        const [x1, y1, x2, y2] = det.bbox;
        const bx = x1 * scaleX;
        const by = y1 * scaleY;
        const bw = (x2 - x1) * scaleX;
        const bh = (y2 - y1) * scaleY;
        const color = det.color || '#FF0000';

        // ── Bounding box ─────────────────────────────────────────────────
        ctx.strokeStyle = color;
        ctx.lineWidth   = bufLineW;
        ctx.strokeRect(bx, by, bw, bh);

        // ── Corner accent marks ───────────────────────────────────────────
        const cs = Math.min(bufCorner, bw * 0.25, bh * 0.25);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = bufLineW * 1.6;
        ctx.lineCap     = 'round';
        for (const [ox, oy, dx, dy] of [
            [bx,      by,      1,  1],   // top-left
            [bx + bw, by,     -1,  1],   // top-right
            [bx,      by + bh, 1, -1],   // bottom-left
            [bx + bw, by + bh,-1, -1],   // bottom-right
        ]) {
            ctx.beginPath();
            ctx.moveTo(ox + dx * cs, oy);
            ctx.lineTo(ox, oy);
            ctx.lineTo(ox, oy + dy * cs);
            ctx.stroke();
        }
        ctx.lineCap   = 'butt';
        ctx.lineWidth = bufLineW;

        // ── Label chip ────────────────────────────────────────────────────
        const conf  = Math.round((det.confidence || 0) * 100);
        const label = det.label || 'Unknown';
        const status = (det.status || '').toLowerCase();

        // Two-line chip: name on top, status + confidence below
        const nameLine   = label;
        const statusLine = `${status.toUpperCase()}  ${conf}%`;

        ctx.font         = `bold ${bufFontPx}px "Segoe UI", Arial, sans-serif`;
        const nameW      = ctx.measureText(nameLine).width;
        ctx.font         = `${Math.round(bufFontPx * 0.82)}px "Segoe UI", Arial, sans-serif`;
        const statusW    = ctx.measureText(statusLine).width;

        const chipW  = Math.max(nameW, statusW) + bufPadH * 2;
        const lineH  = bufFontPx + bufPadV;
        const chipH  = lineH * 2 + bufPadV;

        // Clamp chip so it never goes outside the canvas
        const chipX = Math.min(bx, canvas.width  - chipW);
        const chipY = by >= chipH + bufPadV
            ? by - chipH - bufPadV          // above the box (preferred)
            : by + bufPadV;                 // inside top of box (fallback)

        // Semi-transparent dark base + colour left accent
        ctx.globalAlpha = 0.88;
        ctx.fillStyle = 'rgba(10, 14, 24, 0.92)';
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(chipX, chipY, chipW, chipH, bufRadius);
        } else {
            ctx.rect(chipX, chipY, chipW, chipH);
        }
        ctx.fill();

        // Colour left stripe
        ctx.globalAlpha = 1;
        const stripeW = Math.max(3, Math.round(4 * displayRatioX));
        ctx.fillStyle = color;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(chipX, chipY, stripeW, chipH, [bufRadius, 0, 0, bufRadius]);
        } else {
            ctx.rect(chipX, chipY, stripeW, chipH);
        }
        ctx.fill();

        // Name line (large, white)
        ctx.fillStyle    = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.font         = `bold ${bufFontPx}px "Segoe UI", Arial, sans-serif`;
        ctx.fillText(nameLine,   chipX + stripeW + bufPadH, chipY + lineH * 0.5);

        // Status + confidence line (smaller, coloured)
        ctx.fillStyle = color;
        ctx.font      = `${Math.round(bufFontPx * 0.82)}px "Segoe UI", Arial, sans-serif`;
        ctx.fillText(statusLine, chipX + stripeW + bufPadH, chipY + lineH * 1.5);

        ctx.globalAlpha = 1;
    });
}

// ---------------------------------------------------------------------------
// Base class — shared lifecycle helpers
// ---------------------------------------------------------------------------
class BaseCameraSystem {
    /**
     * @param {HTMLElement} card   - .cam-card DOM element
     * @param {Object}      config - Camera config object from /api/cameras/config
     */
    constructor(card, config) {
        this.card            = card;
        this.config          = config;
        this.camId           = config.id;
        this.camLabel        = config.cam_id || `CAM-${config.id + 1}`;
        this.camName         = config.name   || this.camLabel;
        this.isActive        = false;

        // DOM refs
        this.statusIndicator = card.querySelector('.status-indicator');
        this.toggleButton    = card.querySelector('.toggle-camera');
        this.canvas          = card.querySelector('.cam-overlay');
        this.ctx             = this.canvas ? this.canvas.getContext('2d') : null;
        this.placeholder     = card.querySelector('.cam-offline-placeholder');

        // Patch the card header with server-side name
        const h4 = card.querySelector('h4');
        if (h4) h4.textContent = this.camName;
        const idLabel = card.querySelector('.cam-id-label');
        if (idLabel) idLabel.textContent = `ID: ${this.camLabel}`;

        this._bindToggle();
    }

    _bindToggle() {
        if (this.toggleButton && !this.toggleButton._bound) {
            this.toggleButton._bound = true;
            this.toggleButton.addEventListener('click', () => this.toggle());
        }
    }

    toggle() {
        if (this.isActive) this.disconnect();
        else this.connect();
    }

    setStatus(text, cssClass) {
        if (this.statusIndicator) {
            this.statusIndicator.textContent = text;
            this.statusIndicator.className   = `status-indicator ${cssClass}`;
        }
    }

    setButton(text, disabled = false) {
        if (this.toggleButton) {
            this.toggleButton.textContent = text;
            this.toggleButton.disabled    = disabled;
        }
    }

    showPlaceholder(visible) {
        if (this.placeholder) {
            this.placeholder.style.display = visible ? 'flex' : 'none';
        }
    }

    resizeCanvas(refWidth, refHeight) {
        if (!this.canvas) return;
        const container = this.canvas.parentElement;
        if (!container) return;

        // The canvas CSS is set to position:absolute width:100% height:100%
        // so it already fills the container visually via CSS.
        // We only need to set the *pixel buffer* size to match the source
        // resolution so drawBoxes can map coordinates without scaling errors.
        // Use the source dimensions directly when provided.
        if (refWidth && refHeight) {
            this.canvas.width  = refWidth;
            this.canvas.height = refHeight;
        } else {
            // Fallback: match container display size
            this.canvas.width  = container.clientWidth  || 640;
            this.canvas.height = container.clientHeight || 360;
        }
        // Remove any inline width/height that would fight the CSS 100% rule
        this.canvas.style.width  = '';
        this.canvas.style.height = '';
    }

    // Subclasses must implement:
    connect()    { throw new Error('connect() not implemented'); }
    disconnect() { throw new Error('disconnect() not implemented'); }
}

// ---------------------------------------------------------------------------
// RTSPCameraSystem
// ---------------------------------------------------------------------------
/**
 * Displays the server-side MJPEG stream in an <img> tag.
 * Polls /api/cameras/<id>/detections every POLL_MS ms and draws boxes on
 * the canvas overlay.
 *
 * The server (rtsp_camera.py) already draws boxes on the JPEG itself, but we
 * also draw them on the canvas so the UI sidebar gets populated.
 */
class RTSPCameraSystem extends BaseCameraSystem {
    static POLL_MS = 200;   // detection polling interval (ms)

    constructor(card, config) {
        super(card, config);

        this.imgEl        = card.querySelector('.rtsp-feed');
        this._pollTimer   = null;
        this._statusTimer = null;

        // Hide webcam elements, show RTSP img
        const video = card.querySelector('video');
        if (video) video.style.display = 'none';
        if (this.imgEl) this.imgEl.style.display = 'none';

        this.showPlaceholder(true);
        this.setStatus('OFFLINE', 'status-offline');
        this.setButton('Connect');

        // Auto-connect
        this.connect();
    }

    connect() {
        if (this.isActive) return;
        this.isActive = true;
        this.setButton('Connecting…', true);
        this.setStatus('CONNECTING', 'status-connecting');

        // Point the img at the MJPEG stream endpoint
        const streamUrl = `/api/cameras/stream/${this.camId}?t=${Date.now()}`;
        if (this.imgEl) {
            this.imgEl.onload  = () => this._onStreamLoad();
            this.imgEl.onerror = () => this._onStreamError();
            this.imgEl.src     = streamUrl;
            this.imgEl.style.display = 'block';
        }

        this.showPlaceholder(false);
        this._startPolling();
        this._startStatusCheck();
    }

    disconnect() {
        this.isActive = false;
        this._stopPolling();
        this._stopStatusCheck();

        if (this.imgEl) {
            this.imgEl.src          = '';
            this.imgEl.style.display = 'none';
        }
        if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.showPlaceholder(true);
        this.setStatus('OFFLINE', 'status-offline');
        this.setButton('Connect');

        // Tell server to stop the camera thread
        fetch(`/api/cameras/${this.camId}/stop`, { method: 'POST' }).catch(() => {});
    }

    _onStreamLoad() {
        this.setStatus('LIVE', 'status-live');
        this.setButton('Disconnect', false);
    }

    _onStreamError() {
        this.setStatus('ERROR', 'status-error');
        this.setButton('Retry', false);
        this.isActive = false;
        this.showPlaceholder(true);
        if (this.imgEl) this.imgEl.style.display = 'none';
        // Retry after 5 s
        setTimeout(() => { if (!this.isActive) this.connect(); }, 5000);
    }

    // Poll /api/cameras/<id>/detections to get latest AI results
    _startPolling() {
        this._pollTimer = setInterval(() => this._pollDetections(), RTSPCameraSystem.POLL_MS);
    }

    _stopPolling() {
        if (this._pollTimer) { clearInterval(this._pollTimer); this._pollTimer = null; }
    }

    async _pollDetections() {
        if (!this.isActive) return;
        try {
            const resp = await fetch(`/api/cameras/${this.camId}/detections`);
            if (!resp.ok) return;
            const results = await resp.json();
            if (!Array.isArray(results) || !results.length) return;

            // Draw overlay on canvas using the MJPEG img dimensions
            if (this.imgEl && this.imgEl.naturalWidth) {
                this.resizeCanvas(this.imgEl.naturalWidth, this.imgEl.naturalHeight);
                drawBoxes(
                    this.ctx, this.canvas, results,
                    this.imgEl.naturalWidth, this.imgEl.naturalHeight
                );
            }

            processDetectionResults(results, this.camId, this.camLabel);
        } catch (_) { /* network blip — ignore */ }
    }

    // Periodically check server-side connection status and update badge
    _startStatusCheck() {
        this._statusTimer = setInterval(() => this._checkStatus(), 5000);
    }

    _stopStatusCheck() {
        if (this._statusTimer) { clearInterval(this._statusTimer); this._statusTimer = null; }
    }

    async _checkStatus() {
        if (!this.isActive) return;
        try {
            const resp = await fetch('/api/cameras/status');
            if (!resp.ok) return;
            const statuses = await resp.json();
            const st = statuses.find(s => s.id === this.camId);
            if (!st) return;

            if (st.connected) {
                this.setStatus('LIVE', 'status-live');
                this.setButton('Disconnect', false);
            } else {
                this.setStatus('RECONNECTING…', 'status-connecting');
            }
        } catch (_) {}
    }
}

// ---------------------------------------------------------------------------
// WebcamCameraSystem  (original behaviour, unchanged logic, new structure)
// ---------------------------------------------------------------------------
class WebcamCameraSystem extends BaseCameraSystem {
    static FRAME_INTERVAL_MS = 100;  // 10 fps inference

    constructor(card, config, webcamIndex) {
        super(card, config);

        this.webcamIndex     = webcamIndex;  // which getUserMedia device to pick
        this.videoEl         = card.querySelector('video');
        this.stream          = null;
        this.processing      = false;
        this.lastProcessTime = 0;
        this._rafHandle      = null;

        // Hide RTSP img
        const img = card.querySelector('.rtsp-feed');
        if (img) img.style.display = 'none';

        this.showPlaceholder(true);
        this.setStatus('OFFLINE', 'status-offline');
        this.setButton('Connect');

        this.connect();
    }

    async connect() {
        if (this.isActive) return;
        this.setButton('Connecting…', true);
        this.setStatus('CONNECTING', 'status-connecting');

        try {
            await this._openStream();
            this.isActive = true;
            // video was already made visible inside _openStream onloadedmetadata
            this.showPlaceholder(false);
            this.setStatus('LIVE', 'status-live');
            this.setButton('Disconnect', false);
            this._startProcessing();
        } catch (err) {
            this._handleError(err);
        }
    }

    disconnect() {
        this.isActive = false;
        if (this._rafHandle) cancelAnimationFrame(this._rafHandle);
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.videoEl) {
            this.videoEl.srcObject = null;
            this.videoEl.style.display = 'none';
        }
        if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.showPlaceholder(true);
        this.setStatus('OFFLINE', 'status-offline');
        this.setButton('Connect', false);
    }

    async _openStream() {
        // Permission probe
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach(t => t.stop());

        const devices     = await navigator.mediaDevices.enumerateDevices();
        const videoDevs   = devices.filter(d => d.kind === 'videoinput');
        if (!videoDevs.length) throw new Error('No video devices found');

        const target = videoDevs[this.webcamIndex] || videoDevs[0];
        this.stream  = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: target.deviceId },
                width:  { ideal: 1280 },
                height: { ideal: 720 },
            },
        });

        this.videoEl.srcObject = this.stream;
        await new Promise((res, rej) => {
            this.videoEl.onloadedmetadata = () => {
                // Make video visible first so the container has a real
                // clientHeight, then size the canvas to the native resolution.
                this.videoEl.style.display = 'block';
                this.resizeCanvas(this.videoEl.videoWidth, this.videoEl.videoHeight);
                res();
            };
            this.videoEl.onerror = rej;
        });
    }

    _startProcessing() {
        this.videoEl.play();
        const loop = () => {
            if (!this.isActive) return;
            const now = Date.now();
            if (now - this.lastProcessTime >= WebcamCameraSystem.FRAME_INTERVAL_MS) {
                this._processFrame();
                this.lastProcessTime = now;
            }
            this._rafHandle = requestAnimationFrame(loop);
        };
        this._rafHandle = requestAnimationFrame(loop);
    }

    async _processFrame() {
        if (this.processing || !this.videoEl || this.videoEl.readyState !== 4) return;
        this.processing = true;
        try {
            const frameData = this._captureFrame();
            const results   = await this._sendFrame(frameData);
            if (Array.isArray(results)) {
                // Re-sync canvas buffer only when native video dimensions change
                const vw = this.videoEl.videoWidth;
                const vh = this.videoEl.videoHeight;
                if (this.canvas.width !== vw || this.canvas.height !== vh) {
                    this.resizeCanvas(vw, vh);
                }
                drawBoxes(this.ctx, this.canvas, results, vw, vh);
                processDetectionResults(results, this.camId, this.camLabel);
            }
        } catch (e) {
            console.error(`[${this.camLabel}] Frame error:`, e);
        } finally {
            this.processing = false;
        }
    }

    _captureFrame() {
        const fc  = document.createElement('canvas');
        fc.width  = this.videoEl.videoWidth;
        fc.height = this.videoEl.videoHeight;
        fc.getContext('2d').drawImage(this.videoEl, 0, 0);
        return fc.toDataURL('image/jpeg', 0.7);
    }

    async _sendFrame(frameData) {
        const resp = await fetch('/getdata/', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ image: frameData, camera_id: this.camId }),
        });
        return resp.json();
    }

    _handleError(err) {
        console.error(`[${this.camLabel}] Error:`, err);
        this.isActive = false;
        this.showPlaceholder(true);
        if (this.videoEl) this.videoEl.style.display = 'none';

        let status = 'ERROR';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')
            status = 'NO PERMISSION';
        else if (err.name === 'NotFoundError')
            status = 'NOT FOUND';
        else if (err.name === 'NotReadableError')
            status = 'IN USE';

        this.setStatus(status, 'status-error');
        this.setButton('Retry', false);
    }
}

// ---------------------------------------------------------------------------
// Boot — fetch config then wire up each cam-card
// ---------------------------------------------------------------------------
async function initializeCameraSystems() {
    // Fetch server config
    let cameraConfigs = [];
    try {
        const resp = await fetch('/api/cameras/config');
        if (resp.ok) cameraConfigs = await resp.json();
    } catch (e) {
        console.warn('Could not fetch /api/cameras/config — using webcam fallback.', e);
    }

    // Build a map: numeric id → config
    const configMap = new Map(cameraConfigs.map(c => [c.id, c]));

    // Track how many webcam slots we've handed out
    let webcamSlot = 0;

    const cards = document.querySelectorAll('.cam-card');
    cards.forEach(card => {
        const camId  = parseInt(card.dataset.camId ?? '-1', 10);
        const config = configMap.get(camId) || {
            id:          camId,
            cam_id:      `CAM-0${camId + 1}`,
            name:        card.querySelector('h4')?.textContent || `Camera ${camId + 1}`,
            stream_type: 'webcam',
            enabled:     true,
        };

        if (!config.enabled) {
            card.classList.add('cam-disabled');
            const btn = card.querySelector('.toggle-camera');
            if (btn) { btn.textContent = 'Disabled'; btn.disabled = true; }
            // Show a static offline placeholder with "DISABLED" text
            const ph = card.querySelector('.cam-offline-placeholder span');
            if (ph) ph.textContent = 'DISABLED';
            return;
        }

        if (config.stream_type === 'rtsp') {
            new RTSPCameraSystem(card, config);
        } else {
            new WebcamCameraSystem(card, config, webcamSlot++);
        }
    });
}

// ---------------------------------------------------------------------------
// DOMContentLoaded
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initializeCameraSystems();

    window.addEventListener('resize', () => {
        // Canvas resize is handled per-instance; nothing global needed
    });

    document.querySelector('.clear-detections')?.addEventListener('click', () => {
        detectionsContainer.innerHTML = '';
        detectionCount  = 0;
        recognizedCount = 0;
        unknownCount    = 0;
        trackHistory.clear();
        updateStatElements();
    });
});
