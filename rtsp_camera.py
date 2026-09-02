"""
rtsp_camera.py
==============
Server-side RTSP camera manager for Hikvision IP cameras.

Architecture
------------
Each RTSPCamera instance runs two background threads:

  1. _reader_thread  — opens the RTSP stream via OpenCV (which delegates to
                       FFmpeg under the hood), reads frames continuously, and
                       stores the latest frame in self._frame.  Auto-reconnects
                       on stream loss with configurable back-off.

  2. _inference_thread — pulls the latest frame at RTSP_INFERENCE_INTERVAL
                         seconds, sends it through the InsightFace analyzer and
                         CatBoost classifier, and stores results in
                         self._detections for the Flask routes to serve.

The RTSPCameraManager singleton owns all RTSPCamera instances and is the
single access point for app.py.

Usage (in app.py)
-----------------
    from rtsp_camera import camera_manager
    camera_manager.start()           # starts enabled cameras
    camera_manager.get_jpeg(cam_id)  # latest MJPEG frame bytes
    camera_manager.get_status()      # list of status dicts
"""

import cv2
import time
import threading
import logging
import numpy as np
from datetime import datetime

from config import (
    CAMERAS,
    RTSP_TRANSPORT,
    RTSP_RECONNECT_DELAY,
    RTSP_MAX_FAILURES,
    MJPEG_QUALITY,
    RTSP_INFERENCE_INTERVAL,
    STATUS_COLORS,
)
from utils import rgb_to_hex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — analyzer / classifier are initialised in app.py.
# We grab them at runtime to avoid circular imports.
# ---------------------------------------------------------------------------
def _get_analyzer():
    try:
        import app as _app
        return getattr(_app, 'analyzer', None)
    except Exception:
        return None

def _get_classifier():
    try:
        import app as _app
        return getattr(_app, 'classifier', None)
    except Exception:
        return None

def _save_record(label, conf, bbox, embedding, ts):
    try:
        from database import save_face_record
        save_face_record(label, float(conf), bbox, embedding, ts)
    except Exception as e:
        logger.warning(f"DB save failed: {e}")


# ---------------------------------------------------------------------------
# Single RTSP camera
# ---------------------------------------------------------------------------
class RTSPCamera:
    """
    Manages one RTSP camera stream.

    Public attributes (all read-only from outside):
        cam_id      str   — e.g. "CAM-01"
        name        str   — human label
        rtsp_url    str   — full rtsp:// URL (credentials embedded)
        enabled     bool
        connected   bool  — True while the reader thread has a live capture
        frame_count int   — total frames received since start
    """

    def __init__(self, cfg: dict):
        self.id          = cfg['id']
        self.cam_id      = cfg['cam_id']
        self.name        = cfg['name']
        self.rtsp_url    = cfg['rtsp_url']
        self.enabled     = cfg.get('enabled', True)
        self.stream_type = cfg.get('stream_type', 'rtsp')

        # State
        self.connected   = False
        self.frame_count = 0
        self._running    = False
        self._failures   = 0

        # Latest raw frame (BGR numpy array) and its lock
        self._frame      = None
        self._frame_lock = threading.Lock()

        # Latest JPEG bytes (for MJPEG streaming) and its lock
        self._jpeg       = None
        self._jpeg_lock  = threading.Lock()

        # Latest AI detections list
        self._detections      = []
        self._detections_lock = threading.Lock()

        # Timestamps
        self._last_frame_time     = 0.0
        self._last_inference_time = 0.0

        # Threads
        self._reader_thread    = None
        self._inference_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running or not self.enabled:
            return
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"rtsp-reader-{self.cam_id}",
            daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"rtsp-infer-{self.cam_id}",
            daemon=True
        )
        self._reader_thread.start()
        self._inference_thread.start()
        logger.info(f"[{self.cam_id}] Started — {self._safe_url()}")

    def stop(self):
        self._running = False
        self.connected = False
        logger.info(f"[{self.cam_id}] Stopped.")

    def get_jpeg(self) -> bytes | None:
        """Return the latest JPEG frame bytes, or None if no frame yet."""
        with self._jpeg_lock:
            return self._jpeg

    def get_frame(self) -> np.ndarray | None:
        """Return the latest raw BGR frame, or None."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_detections(self) -> list:
        """Return the latest detection results list."""
        with self._detections_lock:
            return list(self._detections)

    def status(self) -> dict:
        return {
            'id':          self.id,
            'cam_id':      self.cam_id,
            'name':        self.name,
            'stream_type': self.stream_type,
            'enabled':     self.enabled,
            'connected':   self.connected,
            'frame_count': self.frame_count,
            'rtsp_url':    self._safe_url(),
        }

    def update_url(self, rtsp_url: str):
        """Hot-update the RTSP URL and reconnect."""
        was_running = self._running
        self.stop()
        self.rtsp_url = rtsp_url
        if was_running:
            time.sleep(0.5)
            self.start()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_url(self) -> str:
        """Return URL with password replaced by ***."""
        import re
        return re.sub(r'(rtsp://[^:]+:)[^@]+(@)', r'\1***\2', self.rtsp_url)

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Open the RTSP stream via FFmpeg backend with TCP transport."""
        # Tell OpenCV/FFmpeg to use TCP for RTSP and increase buffer size
        # to handle Hikvision cameras that sometimes send large NAL units.
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS',
                              f'rtsp_transport;{RTSP_TRANSPORT}|'
                              'buffer_size;4194304|'
                              'stimeout;5000000')   # 5 s socket timeout

        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return None

        # Set internal buffer to 1 frame so we always get the freshest image
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ------------------------------------------------------------------
    # Reader thread — owns the VideoCapture object
    # ------------------------------------------------------------------

    def _reader_loop(self):
        import os  # needed inside thread
        while self._running:
            cap = None
            try:
                logger.info(f"[{self.cam_id}] Connecting to {self._safe_url()} …")
                cap = self._open_capture()

                if cap is None or not cap.isOpened():
                    raise ConnectionError("VideoCapture failed to open")

                self.connected  = True
                self._failures  = 0
                logger.info(f"[{self.cam_id}] Connected ✓")

                while self._running:
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        self._failures += 1
                        if self._failures >= RTSP_MAX_FAILURES:
                            logger.warning(
                                f"[{self.cam_id}] {self._failures} consecutive "
                                "read failures — reconnecting."
                            )
                            break
                        time.sleep(0.05)
                        continue

                    self._failures = 0
                    self.frame_count += 1
                    self._last_frame_time = time.time()

                    # Store raw frame
                    with self._frame_lock:
                        self._frame = frame

                    # Encode JPEG for MJPEG stream
                    ok, jpeg_buf = cv2.imencode(
                        '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
                    )
                    if ok:
                        with self._jpeg_lock:
                            self._jpeg = jpeg_buf.tobytes()

            except Exception as exc:
                logger.error(f"[{self.cam_id}] Reader error: {exc}")
            finally:
                self.connected = False
                if cap:
                    cap.release()

            if self._running:
                logger.info(
                    f"[{self.cam_id}] Reconnecting in {RTSP_RECONNECT_DELAY} s …"
                )
                time.sleep(RTSP_RECONNECT_DELAY)

    # ------------------------------------------------------------------
    # Inference thread — reads latest frame, runs AI pipeline
    # ------------------------------------------------------------------

    def _inference_loop(self):
        while self._running:
            time.sleep(RTSP_INFERENCE_INTERVAL)

            if not self.connected:
                continue

            frame = self.get_frame()
            if frame is None:
                continue

            analyzer   = _get_analyzer()
            classifier = _get_classifier()

            if analyzer is None:
                continue

            try:
                faces   = analyzer.analyze(frame)
                results = []

                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = (float(v) for v in face.bbox)
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    if classifier:
                        try:
                            status, label, conf = classifier.classify(face.embedding)
                            if status == 'recognized':
                                _save_record(
                                    label, conf,
                                    [x1, y1, x2, y2],
                                    face.embedding, ts
                                )
                        except Exception as ce:
                            logger.debug(f"[{self.cam_id}] Classify err: {ce}")
                            status, label, conf = 'unknown', 'Unknown', 0.0
                    else:
                        status, label, conf = 'unknown', 'No Classifier', 0.0

                    results.append({
                        'bbox':       [x1, y1, x2, y2],
                        'status':     status,
                        'label':      label,
                        'confidence': float(conf),
                        'timestamp':  ts,
                        'color':      rgb_to_hex(
                                          STATUS_COLORS.get(status, (255, 193, 7))
                                      ),
                        'track_id':   i + 1,
                        'cam_id':     self.cam_id,
                    })

                with self._detections_lock:
                    self._detections = results

                # Draw bounding boxes onto the stored JPEG
                if results:
                    self._annotate_frame(frame, results)

            except Exception as exc:
                logger.error(f"[{self.cam_id}] Inference error: {exc}")

    def _annotate_frame(self, frame: np.ndarray, detections: list):
        """Draw bounding boxes + labels on frame, then re-encode JPEG."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det['bbox'])
            color_hex = det.get('color', '#FFFF00')
            # Convert hex to BGR
            color_hex = color_hex.lstrip('#')
            r, g, b = (int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            bgr = (b, g, r)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            label_text = (
                f"{det['label']}  "
                f"{int(det['confidence'] * 100)}%"
            )
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            ty = max(y1 - 10, th + 4)
            cv2.rectangle(annotated,
                          (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                          bgr, cv2.FILLED)
            cv2.putText(annotated, label_text,
                        (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

        ok, jpeg_buf = cv2.imencode(
            '.jpg', annotated,
            [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
        )
        if ok:
            with self._jpeg_lock:
                self._jpeg = jpeg_buf.tobytes()


# ---------------------------------------------------------------------------
# Camera Manager — singleton that owns all RTSPCamera instances
# ---------------------------------------------------------------------------
import os  # needed for _open_capture env var


class RTSPCameraManager:
    """
    Singleton that manages all configured cameras.

    Usage:
        from rtsp_camera import camera_manager
        camera_manager.start()
    """

    def __init__(self):
        self._cameras: dict[int, RTSPCamera] = {}
        self._lock = threading.Lock()
        self._initialised = False

    def _build_cameras(self):
        for cfg in CAMERAS:
            if cfg.get('stream_type', 'rtsp') == 'rtsp':
                cam = RTSPCamera(cfg)
                self._cameras[cfg['id']] = cam

    def start(self):
        """Start all enabled RTSP cameras. Safe to call multiple times."""
        with self._lock:
            if not self._initialised:
                self._build_cameras()
                self._initialised = True
            for cam in self._cameras.values():
                if cam.enabled and not cam._running:
                    cam.start()
        logger.info(
            f"RTSPCameraManager: {len(self._cameras)} camera(s) initialised."
        )

    def stop(self):
        """Stop all cameras."""
        with self._lock:
            for cam in self._cameras.values():
                cam.stop()

    # ------------------------------------------------------------------
    # Frame / detection access
    # ------------------------------------------------------------------

    def get_jpeg(self, cam_id: int) -> bytes | None:
        cam = self._cameras.get(cam_id)
        return cam.get_jpeg() if cam else None

    def get_detections(self, cam_id: int) -> list:
        cam = self._cameras.get(cam_id)
        return cam.get_detections() if cam else []

    def get_status(self) -> list:
        return [cam.status() for cam in self._cameras.values()]

    def get_camera(self, cam_id: int) -> RTSPCamera | None:
        return self._cameras.get(cam_id)

    def start_camera(self, cam_id: int) -> bool:
        cam = self._cameras.get(cam_id)
        if cam:
            cam.enabled = True
            cam.start()
            return True
        return False

    def stop_camera(self, cam_id: int) -> bool:
        cam = self._cameras.get(cam_id)
        if cam:
            cam.stop()
            return True
        return False

    def update_camera_url(self, cam_id: int, rtsp_url: str) -> bool:
        cam = self._cameras.get(cam_id)
        if cam:
            cam.update_url(rtsp_url)
            return True
        return False

    # ------------------------------------------------------------------
    # MJPEG frame generator — used by Flask streaming response
    # ------------------------------------------------------------------

    def mjpeg_generator(self, cam_id: int):
        """
        Yields multipart MJPEG frames for a Flask streaming response.

        Example Flask route:
            from flask import Response
            from rtsp_camera import camera_manager

            @app.route('/api/cameras/stream/<int:cam_id>')
            def stream(cam_id):
                return Response(
                    camera_manager.mjpeg_generator(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
        """
        cam = self._cameras.get(cam_id)
        if cam is None:
            return

        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        blank    = self._blank_frame()

        while cam._running:
            jpeg = cam.get_jpeg()
            if jpeg is None:
                jpeg = blank
                time.sleep(0.1)

            yield boundary + jpeg + b'\r\n'
            time.sleep(0.033)   # ~30 fps display cap

    @staticmethod
    def _blank_frame() -> bytes:
        """Return a small grey 'No Signal' JPEG for offline cameras."""
        img = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.putText(
            img, 'NO SIGNAL',
            (70, 130), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (180, 180, 180), 2, cv2.LINE_AA
        )
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()


# ---------------------------------------------------------------------------
# Module-level singleton — import this in app.py
# ---------------------------------------------------------------------------
camera_manager = RTSPCameraManager()
