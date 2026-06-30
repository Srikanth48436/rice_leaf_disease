"""
Rice Leaf Disease Detection — Flask REST API
==============================================
Serves real-time predictions from the trained model.

Tries to load the Keras CNN model (rice_leaf_disease_model.h5) first —
this is the production model trained by train_cnn.py. If TensorFlow or
the .h5 file isn't available in the current environment, it falls back
to the scikit-learn model (rice_leaf_sklearn_model.joblib) trained by
train_sklearn_model.py, so the API always has something to serve.

Endpoints
---------
GET  /                      -> Web UI
POST /api/predict           -> multipart/form-data image upload -> prediction JSON
GET  /api/health            -> health check + which model is loaded
GET  /api/model-info        -> metadata: classes, accuracy, model type
"""

import os
import io
import json
import time
import uuid

import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from preprocessing import extract_handcrafted_features, CLASS_NAMES, IMG_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Model loading — CNN preferred, sklearn fallback
# ---------------------------------------------------------------------------
MODEL_TYPE = None
CNN_MODEL = None
SKLEARN_MODEL = None
SCALER = None
CLASS_LABELS = CLASS_NAMES
MODEL_META = {}

cnn_path = os.path.join(MODELS_DIR, "rice_leaf_disease_model.h5")
sklearn_path = os.path.join(MODELS_DIR, "rice_leaf_sklearn_model.joblib")
scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
labels_path = os.path.join(MODELS_DIR, "class_labels.json")
meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")

if os.path.exists(labels_path):
    with open(labels_path) as f:
        CLASS_LABELS = json.load(f)

if os.path.exists(meta_path):
    with open(meta_path) as f:
        MODEL_META = json.load(f)

try:
    if os.path.exists(cnn_path):
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        CNN_MODEL = load_model(cnn_path)
        MODEL_TYPE = "cnn"
        print(f"Loaded Keras CNN model from {cnn_path}")
except Exception as e:
    print(f"CNN model not available ({e}); will use sklearn fallback.")

if MODEL_TYPE is None and os.path.exists(sklearn_path):
    import joblib
    SKLEARN_MODEL = joblib.load(sklearn_path)
    SCALER = joblib.load(scaler_path)
    MODEL_TYPE = "sklearn"
    print(f"Loaded scikit-learn model from {sklearn_path}")

if MODEL_TYPE is None:
    print("WARNING: No trained model found. Run train_sklearn_model.py or train_cnn.py first.")


DISEASE_INFO = {
    "Bacterial_leaf_blight": {
        "description": "A bacterial disease causing yellow-to-white water-soaked streaks along leaf veins, often starting at leaf tips or margins.",
        "recommendation": "Use resistant varieties, avoid excess nitrogen, ensure field drainage, and apply copper-based bactericides if severe.",
    },
    "Brown_spot": {
        "description": "A fungal disease producing small circular brown lesions with dark margins and light centers, mainly on older leaves.",
        "recommendation": "Apply balanced fertilization (especially potassium), treat seeds with fungicide, and remove infected crop residue.",
    },
    "Leaf_smut": {
        "description": "A fungal disease causing fine black speckles/specks scattered across the leaf surface, reducing photosynthesis.",
        "recommendation": "Improve field sanitation, apply recommended fungicides, and avoid dense planting to improve airflow.",
    },
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_cnn(image_path):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    probs = CNN_MODEL.predict(image)[0]
    return probs


def predict_sklearn(image_path):
    feats = extract_handcrafted_features(image_path).reshape(1, -1)
    feats_scaled = SCALER.transform(feats)
    probs = SKLEARN_MODEL.predict_proba(feats_scaled)[0]
    return probs


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_type=MODEL_TYPE,
        classes=CLASS_LABELS,
        meta=MODEL_META,
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL_TYPE else "no_model",
        "model_type": MODEL_TYPE,
        "classes": CLASS_LABELS,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": MODEL_TYPE,
        "classes": CLASS_LABELS,
        "metadata": MODEL_META,
        "disease_info": DISEASE_INFO,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if MODEL_TYPE is None:
        return jsonify({"error": "No trained model available on the server."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    # Validate it's actually a readable image
    try:
        img_bytes = file.read()
        Image.open(io.BytesIO(img_bytes)).verify()
    except Exception:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    try:
        if MODEL_TYPE == "cnn":
            probs = predict_cnn(save_path)
        else:
            probs = predict_sklearn(save_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
    info = DISEASE_INFO.get(pred_class, {})

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        "prediction": pred_class,
        "prediction_display": pred_class.replace("_", " "),
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "description": info.get("description", ""),
        "recommendation": info.get("recommendation", ""),
        "model_type": MODEL_TYPE,
        "image_url": url_for("static", filename=f"uploads/{filename}"),
        "inference_time_ms": elapsed_ms,
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size is 8MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
