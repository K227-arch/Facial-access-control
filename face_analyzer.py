
# face_analyzer.p
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from config import DETECTION_MODEL_PATH, RECOGNITION_MODEL_PATH
from utils import rgb_to_hex
import logging

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self):
        self.app = FaceAnalysis()
        self._load_models()
        self._prepare()

    def _load_models(self):
        """Load InsightFace detection and recognition models"""
        logger.info("Loading InsightFace models...")
        
        try:
            # Try to load models if they exist
            if os.path.exists(DETECTION_MODEL_PATH):
                det_model = model_zoo.get_model(DETECTION_MODEL_PATH, task='detection')
                self.app.det_model = det_model
                logger.info(f"Detection model loaded from {DETECTION_MODEL_PATH}")
            else:
                logger.warning(f"Detection model not found at {DETECTION_MODEL_PATH}, using default")
                
            if os.path.exists(RECOGNITION_MODEL_PATH):
                rec_model = model_zoo.get_model(RECOGNITION_MODEL_PATH, task='recognition')
                self.app.rec_model = rec_model
                logger.info(f"Recognition model loaded from {RECOGNITION_MODEL_PATH}")
            else:
                logger.warning(f"Recognition model not found at {RECOGNITION_MODEL_PATH}, using default")
                
        except Exception as e:
            logger.error(f"Error loading custom models: {e}, falling back to defaults")

    def _prepare(self):
        """Prepare the model for inference"""
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Face analyzer ready.")

    def analyze(self, frame):
        """Run full analysis on input frame"""
        return self.app.get(frame)