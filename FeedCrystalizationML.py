# ==========================================================
# 🧪 Starch Gelatinization Stage Classifier
# ==========================================================
# Requirements:
# pip install opencv-python scikit-learn numpy matplotlib
# ==========================================================

import cv2
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # for saving/loading model

# --------------------------
# 1. Image Preprocessing
# --------------------------
def preprocess_image(path, size=(224, 224)):
    """Read, resize, and normalize image."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img / 255.0
    return img

# --------------------------
# 2. Feature Extraction
# --------------------------
def extract_features(img):
    """Extract grayscale histogram and texture features."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture feature: local binary pattern (LBP)
    lbp = (gray > cv2.GaussianBlur(gray, (5,5), 0)).astype(np.uint8)
    lbp_hist = cv2.calcHist([lbp], [0], None, [64], [0,256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()

    return np.concatenate([hist, lbp_hist])

# --------------------------
# 3. Load Dataset
# --------------------------
def load_dataset(dataset_path="dataset"):
    X, y = [], []
    for stage in range(1, 6):
        stage_path = os.path.join(dataset_path, f"stage{stage}")
        for file in glob.glob(f"{stage_path}/*.jpg"):
            img = preprocess_image(file)
            feat = extract_features(img)
            X.append(feat)
            y.append(stage)
    return np.array(X), np.array(y)

# --------------------------
# 4. Train Model
# --------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

# --------------------------
# 5. Predict Unknown Sample
# --------------------------
def predict_sample(model, image_path):
    img = preprocess_image(image_path)
    feat = extract_features(img)
    pred = model.predict([feat])[0]
    probs = model.predict_proba([feat])[0]
    print(f"\n🧩 Predicted Stage: {pred}")
    print(f"Confidence Scores: {np.round(probs, 2)}")
    return pred

# --------------------------
# 6. Save/Load Model
# --------------------------
def save_model(model, filename="starch_stage_model.pkl"):
    joblib.dump(model, filename)
    print(f"\n💾 Model saved as {filename}")

def load_model(filename="starch_stage_model.pkl"):
    return joblib.load(filename)

# --------------------------
# 7. Main Routine
# --------------------------
if __name__ == "__main__":
    # Step 1: Load dataset
    X, y = load_dataset("dataset")  # ensure folder structure dataset/stage1,...,stage5 exists
    print(f"Loaded {len(X)} images for training.")

    # Step 2: Train model
    model = train_model(X, y)

    # Step 3: Save model
    save_model(model)

    # Step 4: Predict unknown sample
    # Change 'unknown/sampleA.jpg' to your test image path
    unknown_image = "unknown/sampleA.jpg"
    if os.path.exists(unknown_image):
        predict_sample(model, unknown_image)
    else:
        print("⚠️ No unknown sample found for prediction.")