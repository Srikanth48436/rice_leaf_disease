"""
Data Preprocessing & Feature Engineering
=========================================
Shared preprocessing utilities used by both the CNN (Keras) training
pipeline and the lightweight sklearn pipeline used for the live demo API.

Two preprocessing paths are provided:
  1. preprocess_for_cnn()      -> normalized (224,224,3) array for deep models
  2. extract_handcrafted_features() -> color/texture feature vector for
                                        classical ML models (RandomForest/SVM)
"""

import os
import numpy as np
import cv2

IMG_SIZE = 224
CLASS_NAMES = ["Bacterial_leaf_blight", "Brown_spot", "Leaf_smut"]


# ---------------------------------------------------------------------------
# Path 1: CNN-ready preprocessing (matches the notebook's pipeline)
# ---------------------------------------------------------------------------
def preprocess_for_cnn(image_path, img_size=IMG_SIZE):
    """Read, resize, color-correct and normalize an image for CNN input."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype("float32") / 255.0
    return image


# ---------------------------------------------------------------------------
# Path 2: Handcrafted feature engineering for classical ML
# ---------------------------------------------------------------------------
def extract_handcrafted_features(image_path, img_size=IMG_SIZE):
    """
    Engineer a feature vector capturing color distribution and lesion
    texture cues that are diagnostic for rice leaf diseases:

      - Mean/std of R, G, B and HSV channels (color disease signatures)
      - Dark-spot ratio (brown/black lesion pixel proportion)
      - Edge density via Canny (captures speckle/streak texture)
      - Local contrast (std of grayscale Laplacian)

    These mirror the kind of visual cues a plant pathologist would use:
    blight -> elongated yellow-brown streaks, brown spot -> circular
    dark-centered lesions, smut -> fine black speckling.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # Color channel statistics
    for channel in cv2.split(rgb):
        features.extend([channel.mean(), channel.std()])
    for channel in cv2.split(hsv):
        features.extend([channel.mean(), channel.std()])

    # Dark lesion ratio (brown/black spots — brown_spot & leaf_smut cue)
    dark_mask = gray < 80
    features.append(dark_mask.mean())

    # Yellow-brown streak ratio (bacterial blight cue) via HSV thresholding
    lower = np.array([15, 60, 100])
    upper = np.array([35, 255, 255])
    streak_mask = cv2.inRange(hsv, lower, upper)
    features.append((streak_mask > 0).mean())

    # Edge density (texture roughness from lesions/specks)
    edges = cv2.Canny(gray, 50, 150)
    features.append((edges > 0).mean())

    # Local contrast via Laplacian variance (focus/texture sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(laplacian.var())

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = (
    ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std",
     "H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std",
     "dark_lesion_ratio", "yellow_brown_streak_ratio",
     "edge_density", "laplacian_variance"]
)


def build_dataset(dataset_path, feature_fn=extract_handcrafted_features):
    """Walk the dataset directory and build (X, y, class_names)."""
    classes = sorted([c for c in os.listdir(dataset_path)
                       if not c.startswith(".") and
                       os.path.isdir(os.path.join(dataset_path, c))])
    X, y = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_path, cls)
        for fname in os.listdir(cls_dir):
            if fname.startswith("."):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                feats = feature_fn(fpath)
                X.append(feats)
                y.append(label_idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    return np.array(X), np.array(y), classes
