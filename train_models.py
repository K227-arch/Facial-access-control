#!/usr/bin/env python3
"""
Train face recognition models using the existing face data
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import pandas as pd

# Add current directory to path to import our modules
sys.path.append('.')

from face_analyzer import FaceAnalyzer
from config import ISOLATION_FOREST_PATH, SVM_MODEL_PATH, LABEL_ENCODER_PATH

def train_models():
    print("🚀 Starting model training...")
    
    # Initialize face analyzer
    print("📊 Loading face analyzer...")
    analyzer = FaceAnalyzer()
    
    # Paths
    image_path = "./facedata"
    
    # Get embeddings and labels
    embeddings, labels = [], []
    
    print("🔍 Processing face images...")
    
    for person in tqdm(os.listdir(image_path), desc="Processing people"):
        person_dir = os.path.join(image_path, person)
        if not os.path.isdir(person_dir):
            continue
            
        person_embeddings = []
        
        for img_name in tqdm(os.listdir(person_dir), 
                           desc=f"Processing {person}", 
                           leave=False):
            img_path = os.path.join(person_dir, img_name)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"⚠️  Could not read image: {img_path}")
                continue
                
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Detect & extract face embedding
            faces = analyzer.analyze(rgb)
            if not faces:
                print(f"⚠️  No face found in: {img_path}")
                continue
                
            # Take the first (largest) face
            embedding = faces[0].embedding
            person_embeddings.append(embedding)
            
        print(f"✅ {person}: {len(person_embeddings)} face embeddings extracted")
        
        # Add all embeddings for this person
        for embedding in person_embeddings:
            embeddings.append(embedding)
            labels.append(person)
    
    if not embeddings:
        print("❌ No face embeddings found! Please check your image data.")
        return False
        
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"📈 Training data: {len(embeddings)} embeddings from {len(set(labels))} people")
    
    # Create Models directory if it doesn't exist
    os.makedirs('./Models', exist_ok=True)
    
    # 1. Train Isolation Forest for outlier detection
    print("🌲 Training Isolation Forest...")
    iso_forest_model = IsolationForest(
        contamination=0.1,  # Expect 10% outliers
        random_state=42,
        n_estimators=100
    )
    iso_forest_model.fit(embeddings)
    
    # Save Isolation Forest
    with open(ISOLATION_FOREST_PATH, 'wb') as f:
        pickle.dump(iso_forest_model, f)
    print(f"✅ Isolation Forest saved to {ISOLATION_FOREST_PATH}")
    
    # 2. Train Label Encoder
    print("🏷️  Training Label Encoder...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    # Save Label Encoder
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"✅ Label Encoder saved to {LABEL_ENCODER_PATH}")
    
    # 3. Train CatBoost Classifier
    print("🐱 Training CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=False,  # Reduce output
        task_type="CPU",  # Use CPU since GPU might not be available
        random_seed=42
    )
    
    model.fit(embeddings, y_encoded)
    
    # Save CatBoost model
    with open(SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ CatBoost model saved to {SVM_MODEL_PATH}")
    
    # Print training summary
    print("\n🎉 Training completed successfully!")
    print(f"📊 Training Summary:")
    print(f"   - Total embeddings: {len(embeddings)}")
    print(f"   - Unique people: {len(set(labels))}")
    print(f"   - People: {', '.join(sorted(set(labels)))}")
    
    return True

if __name__ == "__main__":
    success = train_models()
    if success:
        print("\n✅ Models are ready! Restart the Flask app to use the new models.")
    else:
        print("\n❌ Training failed!")