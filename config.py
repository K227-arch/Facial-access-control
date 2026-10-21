# config.py

import os

# ---------------------------------------------------------------------------
# RTSP / Hikvision Camera Configuration
# ---------------------------------------------------------------------------
# Hikvision RTSP URL format:
#   rtsp://<user>:<password>@<ip>:<port>/Streaming/Channels/<channel>
#
# channel codes:
#   101 = main stream (1080p)  |  102 = sub-stream (640×480, lower CPU)
#
# All cameras share the same credentials unless overridden per-camera.
# Set environment variables to avoid storing credentials in source code:
#   export HIK_USER=admin  HIK_PASS=yourpassword
#
# The LAN network should have cameras reachable on the same subnet as the
# server (e.g. 192.168.1.x) through the Hikvision PoE/managed switch.
# ---------------------------------------------------------------------------

HIK_USER     = os.getenv('HIK_USER', 'admin')
HIK_PASS     = os.getenv('HIK_PASS', 'admin123')
HIK_PORT     = int(os.getenv('HIK_PORT', '554'))   # standard RTSP port

# Convenience helper — build a Hikvision RTSP URL
def hik_rtsp(ip, channel=101, user=None, password=None, port=None):
    u = user     or HIK_USER
    p = password or HIK_PASS
    pt = port    or HIK_PORT
    return f"rtsp://{u}:{p}@{ip}:{pt}/Streaming/Channels/{channel}"

# Per-camera definitions — edit the IP addresses to match your LAN layout.
# stream_type: "rtsp" uses the server-side RTSP reader (Hikvision / IP cam)
#              "webcam" uses the browser getUserMedia webcam
CAMERAS = [
    {
        "id": 0,
        "name": "Webcam — Local",
        "cam_id": "CAM-01",
        "stream_type": "webcam",   # ← browser getUserMedia → posts to /getdata/
        "enabled": True,
    },
    {
        "id": 1,
        "name": "Area B - Side Entrance",
        "cam_id": "CAM-02",
        "stream_type": "rtsp",
        "rtsp_url": hik_rtsp("192.168.1.65"),
        "channel": 101,
        "enabled": False,          # disable until RTSP cameras are on the LAN
    },
    {
        "id": 2,
        "name": "Area C - Parking",
        "cam_id": "CAM-03",
        "stream_type": "rtsp",
        "rtsp_url": hik_rtsp("192.168.1.66"),
        "channel": 101,
        "enabled": False,
    },
    {
        "id": 3,
        "name": "Area D - Back Exit",
        "cam_id": "CAM-04",
        "stream_type": "rtsp",
        "rtsp_url": hik_rtsp("192.168.1.67"),
        "channel": 101,
        "enabled": False,
    },
]

# OpenCV / FFmpeg RTSP transport options.
# "tcp" is more reliable on a LAN; "udp" has lower latency but packet loss.
RTSP_TRANSPORT   = os.getenv('RTSP_TRANSPORT', 'tcp')

# Seconds to wait before attempting a reconnect after stream drop
RTSP_RECONNECT_DELAY = int(os.getenv('RTSP_RECONNECT_DELAY', '5'))

# Max consecutive read failures before triggering a full reconnect
RTSP_MAX_FAILURES = int(os.getenv('RTSP_MAX_FAILURES', '10'))

# MJPEG stream JPEG quality (1-100) — lower = less bandwidth
MJPEG_QUALITY = int(os.getenv('MJPEG_QUALITY', '75'))

# Face-detection polling interval in seconds for RTSP cameras
# (every Nth frame is sent to the AI; 0.2 s ≈ 5 fps inference)
RTSP_INFERENCE_INTERVAL = float(os.getenv('RTSP_INFERENCE_INTERVAL', '0.2'))

# ---------------------------------------------------------------------------
# Model paths
MODEL_DIR = "./Models"
DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "scrfd_10g_bnkps.onnx")
RECOGNITION_MODEL_PATH = os.path.join(MODEL_DIR, "glintr100.onnx")
CLASSIFIER_DIR =  "./Models"

# SHAPE_PREDICTOR_PATH = os.path.join(CLASSIFIER_DIR, "shape_predictor_68_face_landmarks.dat")
# FACE_RECOGNITION_MODEL_PATH = os.path.join(CLASSIFIER_DIR,"dlib_face_recognition_resnet_model_v1.dat")
# ML Models
# ISOLATION_FOREST_PATH = os.path.join(CLASSIFIER_DIR, "dlib_isolation_forest.pkl")
# SVM_MODEL_PATH = os.path.join(CLASSIFIER_DIR, "dlib_lgbm_model.pkl")
# LABEL_ENCODER_PATH = os.path.join(CLASSIFIER_DIR, "dlib_label_encoder.pkl")

ISOLATION_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca.pkl")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.9

# Status colors
STATUS_COLORS = {
    "recognized": (52, 201, 36),   # Green
    "unknown": (220, 53, 69),      # Red
    "processing": (255, 193, 7)    # Yellow
}

# Database configuration
DB_CONFIG = {
    'server': 'localhost',
    'database': 'FaceRecognitionDB',
    'username': 'sa',
    'password': 'YourStrongPassword123'
}

class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET', 'a-very-secret-key')

    # Flask-Mail (using Gmail SMTP)
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    # NOTE: os.getenv() takes the NAME of the env var, not its value.
    # Previously this called os.getenv('admin@gmail.com') which always
    # returned None (no env var literally named that).
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')   # e.g. your.email@gmail.com
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')    # your Gmail app password