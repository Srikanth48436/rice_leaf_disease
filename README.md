# Paddy Scan — Rice Leaf Disease Detection

End-to-end machine learning web application that detects three rice leaf
diseases — **Bacterial leaf blight**, **Brown spot**, and **Leaf smut** —
from a photo, built on the approach in `Rice_Leaf_prediction.ipynb`.

## What's included

| Piece | File | Purpose |
|---|---|---|
| Data preprocessing & feature engineering | `preprocessing.py` | Image normalization for CNN input + handcrafted color/texture feature extraction for classical ML |
| Deep learning training (production) | `train_cnn.py` | Trains & compares a baseline CNN, MobileNetV2, and ResNet50 (transfer learning), matching the notebook's architecture. Requires TensorFlow. |
| Classical ML training (live demo) | `train_sklearn_model.py` | Trains & compares RandomForest / SVM / GradientBoosting on engineered features, then optimizes the best one with GridSearchCV. Runs anywhere — no TensorFlow needed. |
| REST API | `app.py` | Flask API serving predictions. Loads the CNN `.h5` model if present, otherwise falls back to the sklearn model automatically. |
| Web UI | `templates/index.html`, `static/` | Drag-and-drop upload, real-time prediction, confidence + per-class probability breakdown, agronomic recommendations. |
| Synthetic dataset generator | `generate_synthetic_dataset.py` | Generates placeholder leaf images with class-distinguishing visual patterns so the whole pipeline runs without the real dataset. **Swap in the real `Rice Leaf/` dataset folder and nothing else needs to change.** |

## Why two models?

The notebook's pipeline (CNN / MobileNetV2 / ResNet50) needs TensorFlow,
which wasn't available in the environment this was built in. So:

- `train_cnn.py` is the **real production training script** — run it
  wherever TensorFlow is installed (local machine, Colab, a GPU box) to
  produce `models/rice_leaf_disease_model.h5`.
- `train_sklearn_model.py` is a **fully working fallback** that trains
  in seconds on CPU with no extra dependencies, so the API and UI work
  end-to-end right now.

`app.py` automatically prefers the CNN model if it finds one, and falls
back to sklearn otherwise — no code changes needed when you bring the
real `.h5` file in.

## Setup

```bash
pip install -r requirements.txt

# Optional, only needed for the real CNN pipeline:
pip install tensorflow
```

## Using your real dataset

Replace the `dataset/` folder with your actual rice leaf images, keeping
the same structure (one subfolder per class):

```
dataset/
  Bacterial_leaf_blight/
    img1.jpg
    img2.jpg
    ...
  Brown_spot/
    ...
  Leaf_smut/
    ...
```

Then retrain:

```bash
# Classical ML (fast, runs anywhere)
python3 train_sklearn_model.py

# Deep learning (requires tensorflow, much higher accuracy on real photos)
python3 train_cnn.py --dataset dataset --epochs 15
```

## Running the app

```bash
python3 app.py
```

Visit `http://localhost:5000`. Upload a leaf photo and get a real-time
diagnosis with confidence scores.

## REST API

**`GET /api/health`** — health check
```json
{"status": "ok", "model_type": "sklearn", "classes": [...]}
```

**`GET /api/model-info`** — model metadata + disease descriptions

**`POST /api/predict`** — multipart form upload, field name `image`
```json
{
  "prediction": "Brown_spot",
  "prediction_display": "Brown spot",
  "confidence": 0.98,
  "all_probabilities": {"Bacterial_leaf_blight": 0.02, "Brown_spot": 0.98, "Leaf_smut": 0.0},
  "description": "...",
  "recommendation": "...",
  "model_type": "sklearn",
  "inference_time_ms": 19.4
}
```

Example with curl:
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/predict
```

## Model evaluation artifacts

After training, check `models/` for:
- `*_model_comparison.csv` / `.png` — accuracy comparison across models
- `confusion_matrix_*.png` — confusion matrix for the best model
- `*classification_report.json` — precision/recall/F1 per class
- `best_model_meta.json` — which model won and why

## Notes on the synthetic dataset

`generate_synthetic_dataset.py` creates 180 placeholder leaf images (60
per class) with procedurally generated, class-distinguishing visual
patterns (yellow-brown streaks for blight, dark-centered round spots for
brown spot, fine black speckling for smut) — close enough to the real
disease signatures that both training pipelines learn a meaningful
decision boundary, while making it explicit this is a stand-in for real
field photography. Swap in the real dataset for production accuracy.
