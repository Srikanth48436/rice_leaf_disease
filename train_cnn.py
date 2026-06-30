"""
Rice Leaf Disease Detection — CNN Training Pipeline
=====================================================
End-to-end deep learning training script, built on the architecture from
the project notebook (Rice_Leaf_prediction.ipynb):

  1. Data loading + augmentation (ImageDataGenerator)
  2. Baseline CNN (3 conv blocks)
  3. Transfer learning with MobileNetV2 (frozen base)
  4. Transfer learning with ResNet50 (frozen base)
  5. Model comparison + selection of the best performer
  6. Evaluation: confusion matrix, classification report
  7. Export to rice_leaf_disease_model.h5 for the Flask API to serve

Requires TensorFlow/Keras. Run with:
    pip install tensorflow
    python train_cnn.py --dataset dataset --epochs 15

NOTE: this script was authored in an environment without TensorFlow
installed, so it has not been executed here — it is provided as the
production-grade training pipeline to run in your TF-enabled environment
(local GPU/CPU, Colab, etc). The live demo API in this project instead
serves a scikit-learn model (see train_sklearn_model.py) trained on the
same preprocessing module, so the app works end-to-end right now.
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 16


def build_data_generators(dataset_path):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
    )

    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    return train_data, val_data


def build_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transfer_model(base_class, num_classes, fine_tune_last_n=0):
    base = base_class(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    if fine_tune_last_n:
        for layer in base.layers[-fine_tune_last_n:]:
            layer.trainable = True

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                   loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
    ]


def evaluate_and_report(model, val_data, class_labels, out_dir, model_name):
    val_data.reset()
    predictions = model.predict(val_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_data.classes

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

    report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    with open(os.path.join(out_dir, f"classification_report_{model_name}.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    return report


def main():
    parser = argparse.ArgumentParser(description="Train rice leaf disease CNN models")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", default="models", help="Output directory for models/reports")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data with augmentation...")
    train_data, val_data = build_data_generators(args.dataset)
    class_labels = list(train_data.class_indices.keys())
    num_classes = len(class_labels)
    print("Classes:", train_data.class_indices)

    callbacks = get_callbacks()
    results = {}

    # ---- Baseline CNN ----
    print("\nTraining baseline CNN...")
    cnn_model = build_cnn_model(num_classes)
    cnn_model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)
    cnn_loss, cnn_acc = cnn_model.evaluate(val_data)
    results["CNN"] = cnn_acc
    print("CNN Accuracy:", cnn_acc)

    # ---- MobileNetV2 transfer learning ----
    print("\nTraining MobileNetV2 transfer model...")
    mobilenet_model = build_transfer_model(MobileNetV2, num_classes)
    mobilenet_model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)
    mob_loss, mob_acc = mobilenet_model.evaluate(val_data)
    results["MobileNetV2"] = mob_acc
    print("MobileNetV2 Accuracy:", mob_acc)

    # ---- ResNet50 transfer learning ----
    print("\nTraining ResNet50 transfer model...")
    resnet_model = build_transfer_model(ResNet50, num_classes)
    resnet_model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)
    res_loss, res_acc = resnet_model.evaluate(val_data)
    results["ResNet50"] = res_acc
    print("ResNet50 Accuracy:", res_acc)

    # ---- Compare and select best ----
    results_df = pd.DataFrame({"Model": list(results.keys()), "Accuracy": list(results.values())})
    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("\nModel comparison:\n", results_df)

    plt.figure(figsize=(7, 5))
    sns.barplot(x="Model", y="Accuracy", data=results_df)
    plt.title("Model Comparison — Validation Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "model_comparison.png"))
    plt.close()

    best_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    best_model = {"CNN": cnn_model, "MobileNetV2": mobilenet_model, "ResNet50": resnet_model}[best_name]
    print(f"\nBest model: {best_name} (val accuracy={results[best_name]:.4f})")

    evaluate_and_report(best_model, val_data, class_labels, args.out, best_name)

    # ---- Save artifacts ----
    model_path = os.path.join(args.out, "rice_leaf_disease_model.h5")
    best_model.save(model_path)

    with open(os.path.join(args.out, "class_labels.json"), "w") as f:
        json.dump(class_labels, f)

    with open(os.path.join(args.out, "best_model_meta.json"), "w") as f:
        json.dump({"best_model": best_name, "val_accuracy": float(results[best_name]),
                    "img_size": IMG_SIZE, "all_results": results}, f, indent=2)

    print(f"\nSaved best model to {model_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
