# classifier.py
import pickle
import logging
import numpy as np
from config import ISOLATION_FOREST_PATH, SVM_MODEL_PATH, LABEL_ENCODER_PATH, CONFIDENCE_THRESHOLD
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from catboost import CatBoostClassifier
import pandas as pd
import cv2
import os

logger = logging.getLogger(__name__)

class FaceClassifier:
    def __init__(self):
        logger.info("Loading classifier models...")
        self.iso_forest = None
        self.model = None
        self.le = None
        
        try:
            if os.path.exists(ISOLATION_FOREST_PATH):
                with open(ISOLATION_FOREST_PATH, 'rb') as f:
                    self.iso_forest = pickle.load(f)
                logger.info("Isolation forest model loaded.")
            else:
                logger.warning(f"Isolation forest model not found at {ISOLATION_FOREST_PATH}")
                
            if os.path.exists(SVM_MODEL_PATH):
                with open(SVM_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Classification model loaded.")
            else:
                logger.warning(f"Classification model not found at {SVM_MODEL_PATH}")
                
            if os.path.exists(LABEL_ENCODER_PATH):
                with open(LABEL_ENCODER_PATH, 'rb') as f:
                    self.le = pickle.load(f)
                logger.info("Label encoder loaded.")
            else:
                logger.warning(f"Label encoder not found at {LABEL_ENCODER_PATH}")
                
        except Exception as e:
            logger.error(f"Error loading classifier models: {e}")

    def classify(self, embedding):
        """Classify a face embedding"""
        if not all([self.iso_forest, self.model, self.le]):
            return "unknown", "Models not loaded", 0.0
            
        try:
            is_outlier = self.iso_forest.predict([embedding])[0]
            if is_outlier == -1:
                return "unknown", "Unknown", 0.0

            pred = self.model.predict([embedding])[0]
            label = self.le.inverse_transform([pred])[0]
            confidence = float(self.model.predict_proba([embedding]).max())

            if confidence >= CONFIDENCE_THRESHOLD:
                return "recognized", label, confidence

            return "processing", "Process...", confidence
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown", "Classification Error", 0.0



def retrain_model(analyzer):
    image_path = "./facedata"  # Update with your path

    # Get embeddings and labels
    embeddings, labels = [], []

    for person in tqdm(os.listdir(image_path), desc="Persons", unit="person"):
        person_dir = os.path.join(image_path, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in tqdm(os.listdir(person_dir),
                             desc=f"Images ({person})", unit="img", leave=False):
            img_path = os.path.join(person_dir, img_name)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Detect & align & embed in one go
            faces = analyzer.analyze(rgb)
            if not faces:
                continue

            # Take the first face
            embedding = faces[0].embedding  # 512-dim embedding
            embeddings.append(embedding)
            labels.append(person)

    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Create DataFrame
    data = pd.DataFrame(embeddings)
    combined = pd.concat([data, pd.Series(labels, name='label')], axis=1)

    # Shuffle data
    from sklearn.utils import shuffle

    combined = shuffle(combined).reset_index(drop=True)

    # Train Isolation Forest for outlier detection
    iso_forest_model = IsolationForest(contamination=0.3, random_state=42)
    iso_forest_model.fit(embeddings)

    # Save model
    with open('./Models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest_model, f)

    # Label encoding for classification
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)


    model = CatBoostClassifier(
        iterations=1000,  # number of trees
        learning_rate=0.01,  # how fast to learn
        depth=6,  # depth of trees
        verbose=True,  # set verbose>0 if you want training logs
        task_type="GPU"

    )
    model.fit(embeddings, y_encoded)

    # Save models
    with open('./Models/catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('./Models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open('./Models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest_model, f)
