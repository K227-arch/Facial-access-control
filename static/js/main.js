// Global Detection State
const detectionsContainer = document.getElementById('detectionsContainer');
const trackHistory = new Map(); // track_id -> { shown: boolean, status: string, label: string }
let detectionCount = 0;
let recognizedCount = 0;
let unknownCount = 0;

class CameraSystem {
    constructor(videoElement, canvasElement, toggleButton, cameraIndex) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.toggleButton = toggleButton;
        this.statusIndicator = this.toggleButton.parentElement.querySelector('.status-indicator');
        this.cameraIndex = cameraIndex;
        this.stream = null;
        this.processing = false;
        this.lastProcessTime = 0;
        this.frameInterval = 100; // 10 FPS (1000ms/10 = 100ms)
        this.isActive = false;
        this.activeTracks = new Set();

        this.initializeCamera();
    }

    async initializeCamera() {
        try {
            await this.setupMediaStream();
            this.setupEventListeners();
            this.startProcessing();
            this.updateButtonState('Disconnect', 'LIVE', 'status-live');
        } catch (error) {
            console.error(`Camera ${this.cameraIndex} initialization failed:`, error);
            this.handleCameraError();
        }
    }

    async setupMediaStream() {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        if (videoDevices.length <= this.cameraIndex) {
            throw new Error(`Camera ${this.cameraIndex} not available`);
        }

        this.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: videoDevices[this.cameraIndex].deviceId },
                width: { min: 640, ideal: 1280 },
                height: { min: 480, ideal: 720 }
            }
        });

        this.video.srcObject = this.stream;
        await new Promise((resolve) => {
            this.video.onloadedmetadata = () => {
                this.updateCanvasDimensions();
                resolve();
            };
        });
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.updateCanvasDimensions());
        this.toggleButton.addEventListener('click', () => this.toggleCamera());
    }

    startProcessing() {
        this.isActive = true;
        this.video.play();
        this.processVideoFrames();
    }

    async processVideoFrames() {
        if (!this.isActive) return;

        const currentTime = Date.now();
        if (currentTime - this.lastProcessTime >= this.frameInterval) {
            await this.processFrame();
            this.lastProcessTime = currentTime;
        }

        requestAnimationFrame(() => this.processVideoFrames());
    }

    async processFrame() {
        if (this.processing || this.video.readyState !== 4) return;
        this.processing = true;

        try {
            const frame = this.captureVideoFrame();
            const detections = await this.sendForAnalysis(frame);
            this.handleDetectionResults(detections);
        } catch (error) {
            console.error('Frame processing error:', error);
        } finally {
            this.processing = false;
        }
    }

    captureVideoFrame() {
        const frameCanvas = document.createElement('canvas');
        frameCanvas.width = this.video.videoWidth;
        frameCanvas.height = this.video.videoHeight;
        const frameCtx = frameCanvas.getContext('2d');
        frameCtx.drawImage(this.video, 0, 0);
        return frameCanvas.toDataURL('image/jpeg', 0.7);
    }

    async sendForAnalysis(frameData) {
        const response = await fetch('/getdata/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: frameData,
                camera_id: this.cameraIndex
            })
        });
        return response.json();
    }

    handleDetectionResults(results) {
        if (Array.isArray(results)) {
            this.drawDetectionBoxes(results);

            // Process detections for this camera
            results.forEach(result => {
                if (result.track_id) {
                    // Only process new tracks
                    if (!trackHistory.has(result.track_id)) {
                        const isRecognized = result.status === 'recognized';

                        trackHistory.set(result.track_id, {
                            shown: false,
                            status: result.status,
                            label: result.label,
                            cameraIndex: this.cameraIndex
                        });

                        // Update counts
                        detectionCount++;
                        if (isRecognized) {
                            recognizedCount++;
                        } else {
                            unknownCount++;
                        }
                    }

                    // Get track info
                    const trackInfo = trackHistory.get(result.track_id);

                    // Create UI entry only if not shown before
                    if (!trackInfo.shown) {
                        createDetectionEntry(result, this.cameraIndex);
                        trackInfo.shown = true;
                        updateStatElements();
                    }
                }
            });
        }
    }

    drawDetectionBoxes(detections) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;

//        detections.forEach(detection => {
//            if (!detection.bbox) return;
//
//            const [x1, y1, x2, y2] = detection.bbox;
//            const width = x2 - x1;
//            const height = y2 - y1;
//
//            this.ctx.strokeStyle = detection.color || '#FF0000';
//            this.ctx.lineWidth = 2;
//            this.ctx.strokeRect(
//                x1 * scaleX,
//                y1 * scaleY,
//                width * scaleX,
//                height * scaleY
//            );
//
//            this.ctx.fillStyle = detection.color || '#FF0000';
//            this.ctx.font = '20px Arial';
//
//            this.ctx.fillText(
//                `${detection.label || 'Processing...'} `,
////                `${detection.label || 'Processing...'} ${Math.round(detection.confidence * 100)}%`,
//
//                x1 * scaleX,
//                y1 * scaleY - 10
//            );
//        });
            detections.forEach(detection => {
                    if (!detection.bbox) return;

                    const [x1, y1, x2, y2] = detection.bbox;
                    const width = x2 - x1;
                    const height = y2 - y1;

                    // --- Draw Bounding Box ---
                    this.ctx.strokeStyle = detection.color || '#FF0000';
                    this.ctx.lineWidth = 2;
                    this.ctx.strokeRect(
                        x1 * scaleX,
                        y1 * scaleY,
                        width * scaleX,
                        height * scaleY
                    );

                    // --- Text Label with Background ---

                    const text = `${detection.label || 'Processing...'} ${Math.round(detection.confidence * 100)}%`;
                    this.ctx.font = '20px Arial';

                    // 1. Measure text dimensions
                    const textMetrics = this.ctx.measureText(text);
                    const textWidth = textMetrics.width;
                    const textHeight = 20; // Approximate height based on font size
                    const padding = 4;

                    // 2. Draw the background rectangle
                    const rectX = x1 * scaleX;
                    // Position the rectangle right above the bounding box
                    const rectY = y1 * scaleY - textHeight - (padding * 2);
                    this.ctx.fillStyle = detection.color || '#FF0000'; // Background color
                    this.ctx.fillRect(
                        rectX,
                        rectY,
                        textWidth + (padding * 2),
                        textHeight + (padding * 2)
                    );

                    // 3. Draw the text on top of the background
                    this.ctx.fillStyle = '#FFFFFF'; // Text color (white for contrast)
                    this.ctx.fillText(
                        text,
                        rectX + padding,
                        rectY + textHeight + padding // Position text inside the rectangle
                    );
                });
    }

    updateCanvasDimensions() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;

        const videoRatio = this.video.videoWidth / this.video.videoHeight;
        const container = this.video.parentElement;

        if (!container) return;

        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;

        let width, height;
        if (containerWidth / containerHeight > videoRatio) {
            height = containerHeight;
            width = height * videoRatio;
        } else {
            width = containerWidth;
            height = width / videoRatio;
        }

        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
    }

    stopCamera() {
        this.isActive = false;
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.video.srcObject = null;
        }
        this.updateButtonState('Connect', 'OFFLINE', 'status-offline');
    }

    async connectCamera() {
        try {
            await this.setupMediaStream();
            this.startProcessing();
            this.updateButtonState('Disconnect', 'LIVE', 'status-live');
        } catch (error) {
            console.error(`Camera ${this.cameraIndex} connection failed:`, error);
            this.handleCameraError();
        }
    }

    toggleCamera() {
        if (this.isActive) {
            this.stopCamera();
        } else {
            this.connectCamera();
        }
    }

    updateButtonState(buttonText, statusText, statusClass) {
        this.toggleButton.textContent = buttonText;
        this.statusIndicator.textContent = statusText;
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add(statusClass);
    }

    handleCameraError() {
        console.error(`Camera ${this.cameraIndex} failed`);
        if (this.canvas) {
            this.canvas.style.backgroundColor = '#ff000020';
        }
        this.toggleButton.disabled = true;
        this.toggleButton.textContent = 'Error';
        this.statusIndicator.textContent = 'ERROR';
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add('status-error');
    }
}

// Global Functions
function updateStatElements() {
    const elements = {
        '.card:nth-child(2) .stat-number': detectionCount,
        '.card:nth-child(3) .stat-number': unknownCount,
        '.card:nth-child(4) .stat-number': recognizedCount
    };

    Object.entries(elements).forEach(([selector, value]) => {
        const element = document.querySelector(selector);
        if (element) element.textContent = value;
    });
}

function createDetectionEntry(detection, cameraIndex) {
    const timeStr = detection.timestamp ?
        new Date(detection.timestamp).toLocaleTimeString() :
        new Date().toLocaleTimeString();

    const detectionItem = document.createElement('div');
    detectionItem.className = 'det-item';
    detectionItem.innerHTML = `
        <div class="det-item-name">
            <i class="fas fa-video det-icon"></i>
            <div>
                <strong>${detection.label || 'Unknown'}</strong>
                <small>Camera ${cameraIndex + 1} - ${timeStr}</small>
            </div>
        </div>
        <div class="det-det">
            <div class="det-status ${detection.status || 'unknown'}">
                ${(detection.status || 'Unknown').toUpperCase()}
            </div>
            <small>${Math.round(detection.confidence * 100)}% match</small>
        </div>
    `;

    detectionsContainer.prepend(detectionItem);
    maintainDetectionListLimit();
}

function maintainDetectionListLimit() {
    while (detectionsContainer.children.length > 80) {
        detectionsContainer.removeChild(detectionsContainer.lastChild);
    }
}

async function initializeCameraSystems() {
    try {
        const cameras = document.querySelectorAll('.cam-card');
        for (const [index, cameraCard] of cameras.entries()) {
            const videoElement = cameraCard.querySelector('video');
            const canvasElement = cameraCard.querySelector('canvas');
            const toggleButton = cameraCard.querySelector('.toggle-camera');

            if (videoElement && canvasElement && toggleButton) {
                new CameraSystem(videoElement, canvasElement, toggleButton, index);
            }
        }
    } catch (error) {
        console.error('Camera initialization error:', error);
        alert('Error initializing cameras. Please check your devices and permissions.');
    }
}

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    initializeCameraSystems();

    // Clear detections button
    document.querySelector('.clear-detections')?.addEventListener('click', () => {
        detectionsContainer.innerHTML = '';
        detectionCount = 0;
        recognizedCount = 0;
        unknownCount = 0;
        trackHistory.clear();
        updateStatElements();
    });
});