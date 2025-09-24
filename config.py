# config.py

import os

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
    MAIL_USERNAME = os.getenv('admin@gmail.com')      # e.g. your.email@gmail.com
    MAIL_PASSWORD = os.getenv('admin123')      # your Gmail app password