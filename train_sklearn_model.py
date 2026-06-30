"""
Rice Leaf Disease Detection — Classical ML Training (Live Demo Model)
=======================================================================
A scikit-learn pipeline that trains on handcrafted color/texture features
(see preprocessing.py). This is what actually powers the Flask API's live
prediction endpoint in this sandbox, since TensorFlow isn't available here.

It mirrors the notebook's spirit (model evaluation + comparison + best
model selection) but with classical ML so it runs end-to-end right now:

  1. Feature engineering from images
  2. Train/test split + StandardScaler
  3. Train + compare RandomForest, SVM, GradientBoosting
  4. Evaluate (accuracy, classification report, confusion matrix)
  5. Hyperparameter optimization (GridSearchCV) on the best performer
  6. Persist the final pipeline (scaler + model) for the API to load
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

from preprocessing import build_dataset, FEATURE_NAMES, CLASS_NAMES

DATASET_PATH = "dataset"
OUT_DIR = "models"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Step 1/6 — Extracting handcrafted features from images...")
    X, y, class_labels = build_dataset(DATASET_PATH)
    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Classes: {class_labels}")

    print("\nStep 2/6 — Train/test split + scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  Train: {X_train_s.shape}, Test: {X_test_s.shape}")

    print("\nStep 3/6 — Training & comparing candidate models...")
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=4)
        results[name] = {
            "test_accuracy": acc,
            "macro_f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        fitted_models[name] = model
        print(f"  {name}: test_acc={acc:.3f}  macro_f1={f1:.3f}  cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUT_DIR, "sklearn_model_comparison.csv"))

    plt.figure(figsize=(7, 5))
    sns.barplot(x=results_df.index, y=results_df["test_accuracy"])
    plt.title("Model Comparison — Test Accuracy")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sklearn_model_comparison.png"))
    plt.close()

    best_name = results_df["test_accuracy"].astype(float).idxmax()
    print(f"\nBest baseline model: {best_name}")

    print("\nStep 4/6 — Hyperparameter optimization on best model...")
    if best_name == "RandomForest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 4],
        }
        base_model = RandomForestClassifier(random_state=42)
    elif best_name == "SVM":
        param_grid = {
            "C": [0.5, 1, 5, 10],
            "gamma": ["scale", "auto"],
        }
        base_model = SVC(kernel="rbf", probability=True, random_state=42)
    else:
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
        }
        base_model = GradientBoostingClassifier(random_state=42)

    grid = GridSearchCV(base_model, param_grid, cv=4, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_s, y_train)
    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")

    final_preds = best_model.predict(X_test_s)
    final_acc = accuracy_score(y_test, final_preds)
    print(f"  Optimized test accuracy: {final_acc:.3f} (was {results[best_name]['test_accuracy']:.3f})")

    print("\nStep 5/6 — Final evaluation report...")
    report = classification_report(y_test, final_preds, target_names=class_labels, output_dict=True)
    print(classification_report(y_test, final_preds, target_names=class_labels))

    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name} (optimized)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_sklearn.png"))
    plt.close()

    with open(os.path.join(OUT_DIR, "sklearn_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nStep 6/6 — Saving model artifacts...")
    joblib.dump(best_model, os.path.join(OUT_DIR, "rice_leaf_sklearn_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
    with open(os.path.join(OUT_DIR, "class_labels.json"), "w") as f:
        json.dump(class_labels, f)
    with open(os.path.join(OUT_DIR, "best_model_meta.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "best_params": grid.best_params_,
            "baseline_accuracy": results[best_name]["test_accuracy"],
            "optimized_accuracy": final_acc,
            "feature_names": FEATURE_NAMES,
            "all_baseline_results": results,
        }, f, indent=2)

    print(f"\nDone. Optimized {best_name} saved with test accuracy {final_acc:.3f}")


if __name__ == "__main__":
    main()
