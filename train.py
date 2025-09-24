import os
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from tqdm import tqdm
import pandas as pd
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# Define Data Augmentation Function
def augment_image(image, num_aug=5):
    augmented_images = []
    for _ in range(num_aug):
        aug = image.copy()

        # Brightness and contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-30, 30)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # Horizontal flip
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)

        # Rotation
        angle = np.random.uniform(-10, 10)
        center = (aug.shape[1] // 2, aug.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug = cv2.warpAffine(aug, M, (aug.shape[1], aug.shape[0]),
                             borderMode=cv2.BORDER_REFLECT_101)

        # Scaling
        scale = np.random.uniform(0.8, 1.2)
        new_size = (int(aug.shape[1] * scale), int(aug.shape[0] * scale))
        aug = cv2.resize(aug, new_size, interpolation=cv2.INTER_LINEAR)
        aug = cv2.resize(aug, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        augmented_images.append(aug)
    return augmented_images

# Initialize FaceAnalysis with custom models
det_model = model_zoo.get_model(
    "./Models/scrfd_10g_bnkps.onnx", task='detection')
rec_model = model_zoo.get_model(
    "./Models/glintr100.onnx", task='recognition')

app = FaceAnalysis()
app.det_model = det_model
app.rec_model = rec_model
app.models = {'detection': det_model, 'recognition': rec_model}
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to dataset
image_path = "./facedata"

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

        # Generate 5 augmented images
        augmented_images = augment_image(rgb, num_aug=1)

        # Process original image
        faces = app.get(rgb)
        if faces:
            embedding = faces[0].embedding
            embeddings.append(embedding)
            labels.append(person)

        # Process augmented images
        for aug_img in augmented_images:
            faces_aug = app.get(aug_img)
            if faces_aug:
                embedding_aug = faces_aug[0].embedding
                embeddings.append(embedding_aug)
                labels.append(person)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Create DataFrame
data = pd.DataFrame(embeddings)
combined = pd.concat([data, pd.Series(labels, name='label')], axis=1)

# Shuffle data
combined = shuffle(combined).reset_index(drop=True)

# Train Isolation Forest for outlier detection
iso_forest_model = IsolationForest(contamination=0.3, random_state=42)
iso_forest_model.fit(embeddings)

# Label encoding for classification
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

# Train CatBoost classifier
model = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth    = 6, verbose=0, task_type='GPU')
model.fit(embeddings, y_encoded)

# Save models
with open('./Models/catboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./Models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('./Models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest_model, f)