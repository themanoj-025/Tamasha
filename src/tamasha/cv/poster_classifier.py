"""Poster aesthetic classification module.

Proof-of-concept using TMDb poster images (~200) with hand-crafted
visual features (color histograms, brightness, edge density, face
count) and a Random Forest classifier for hit/flop prediction.

Limitations are explicitly documented — this is scoped as a small
demonstration, not a production CV pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tamasha.config import settings

# ── Face cascade (loaded once) ────────────────────────────────────────
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_FACE_CASCADE_PATH)


logger = logging.getLogger(__name__)


def load_poster_images(
    image_dir: Optional[Path] = None,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load poster images from disk with hit/flop labels.

    Images should be named ``hit_<title>_<year>.jpg`` or
    ``flop_<title>_<year>.jpg``.

    Parameters
    ----------
    image_dir : Path, optional
        Directory with poster images.  Defaults to
        ``settings.DATA_PROCESSED / "posters"``.
    target_size : tuple[int, int], default=(224, 224)
        Resize images to this size.

    Returns
    -------
    tuple[list[np.ndarray], list[int], list[str]]
        ``(images, labels, filenames)``.
    """
    image_dir = image_dir or settings.DATA_PROCESSED / "posters"

    if not image_dir.exists():
        logger.warning("Poster directory does not exist: %s", image_dir)
        return [], [], []

    images: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []

    for fpath in sorted(image_dir.glob("*.jpg")):
        label = 1 if fpath.name.startswith("hit_") else 0
        img = cv2.imread(str(fpath))
        if img is None:
            logger.debug("Could not read: %s", fpath.name)
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        images.append(img)
        labels.append(label)
        filenames.append(fpath.name)

    logger.info(
        "Loaded %d poster images (%d hits, %d flops) from %s",
        len(images),
        sum(labels),
        len(labels) - sum(labels),
        image_dir,
    )
    return images, labels, filenames


def extract_color_histogram(
    image: np.ndarray, bins: int = 16
) -> np.ndarray:
    """Extract HSV color histogram features.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.
    bins : int, default=16
        Number of bins per channel.

    Returns
    -------
    np.ndarray
        Concatenated histogram (3 * bins floats).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features: list[np.ndarray] = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    return np.concatenate(features)


def extract_brightness_features(image: np.ndarray) -> np.ndarray:
    """Extract brightness and contrast statistics.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.

    Returns
    -------
    np.ndarray
        ``[mean_brightness, std_brightness, mean_contrast, std_contrast]``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    # Laplacian variance as contrast measure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mean_contrast = float(np.mean(np.abs(laplacian)))
    std_contrast = float(np.std(laplacian))
    return np.array([mean_brightness, std_brightness, mean_contrast, std_contrast])


def extract_edge_density(image: np.ndarray) -> float:
    """Compute edge density using Canny edge detection.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.

    Returns
    -------
    float
        Fraction of edge pixels (0–1).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def detect_faces(image: np.ndarray) -> int:
    """Count faces in an image using Haar cascade.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.

    Returns
    -------
    int
        Number of faces detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces)


def extract_channel_statistics(image: np.ndarray) -> np.ndarray:
    """Extract per-channel mean, std, and skewness.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.

    Returns
    -------
    np.ndarray
        9 values (3 channels × 3 stats).
    """
    stats = []
    for channel in range(3):
        ch = image[:, :, channel].flatten().astype(float)
        stats.append(float(np.mean(ch)))
        stats.append(float(np.std(ch)))
        # Skewness approximation
        mean_ch = np.mean(ch)
        std_ch = np.std(ch) + 1e-8
        skew = float(np.mean(((ch - mean_ch) / std_ch) ** 3))
        stats.append(skew)
    return np.array(stats)


def extract_all_features(images: list[np.ndarray]) -> np.ndarray:
    """Extract all visual features from a list of images.

    Parameters
    ----------
    images : list[np.ndarray]
        List of RGB image arrays.

    Returns
    -------
    np.ndarray
        Feature matrix (n_samples x n_features).
    """
    feature_list: list[np.ndarray] = []
    for img in images:
        features = np.concatenate(
            [
                extract_color_histogram(img),       # 48 features
                extract_brightness_features(img),    # 4 features
                [extract_edge_density(img)],         # 1 feature
                [detect_faces(img)],                 # 1 feature
                extract_channel_statistics(img),     # 9 features
            ]
        )
        feature_list.append(features)
    return np.array(feature_list)


def train_poster_classifier(
    images: list[np.ndarray],
    labels: list[int],
    model_type: str = "simple_cnn",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a hit/flop classifier from poster images.

    Since TensorFlow is not available, uses hand-crafted features
    and a Random Forest classifier.

    Parameters
    ----------
    images : list[np.ndarray]
        List of image arrays (H, W, C).
    labels : list[int]
        Binary labels (1 = hit, 0 = flop).
    model_type : str, default=\"simple_cnn\"
        Ignored (only Random Forest supported without TensorFlow).
    test_size : float, default=0.3
        Test split fraction.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict[str, Any]
        Results with accuracy, baseline, confusion matrix, and report.
    """
    if len(images) < 20:
        logger.warning("Too few images (%d) for training.", len(images))
        return {
            "accuracy": None,
            "baseline": None,
            "n_samples": len(images),
            "message": f"Only {len(images)} images available — need at least 20.",
        }

    # Extract features
    logger.info("Extracting features from %d images...", len(images))
    X = extract_all_features(images)
    y = np.array(labels)

    # Check class balance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    majority_class_pct = max(n_pos, n_neg) / len(y) * 100
    logger.info(
        "Class distribution: hits=%d (%.1f%%), flops=%d (%.1f%%)",
        n_pos, n_pos / len(y) * 100,
        n_neg, n_neg / len(y) * 100,
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    logger.info("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Majority-class baseline
    majority_class = max(y_train.sum(), len(y_train) - y_train.sum())
    baseline = majority_class / len(y_train)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(
        "Poster classifier accuracy: %.2f%% (baseline: %.2f%%)",
        accuracy * 100, baseline * 100,
    )
    logger.info("Confusion matrix: %s", cm)

    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "n_samples": len(images),
        "n_features": X.shape[1],
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_names": [
            "color_histogram_48d",
            "brightness_mean",
            "brightness_std",
            "contrast_mean",
            "contrast_std",
            "edge_density",
            "face_count",
            "channel_stats_9d",
        ],
        "message": (
            f"Poster classifier accuracy: {accuracy*100:.1f}% "
            f"vs majority-class baseline: {baseline*100:.1f}%. "
            "Hand-crafted visual features (no deep learning)."
        ),
    }


def evaluate_poster_signal(
    image_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate whether poster aesthetics carry independent signal.

    Trains a classifier and reports accuracy vs. a majority-class
    baseline.  If signal is weak/null that is honestly reported.

    Parameters
    ----------
    image_dir : Path, optional
        Directory with poster images.

    Returns
    -------
    dict[str, Any]
        Results summary with accuracy, baseline, and interpretation.
    """
    images, labels, _ = load_poster_images(image_dir)

    if len(images) < 20:
        return {
            "accuracy": None,
            "baseline": None,
            "n_samples": len(images),
            "message": (
                f"Only {len(images)} poster images available. "
                "Need at least 20 for meaningful evaluation."
            ),
        }

    results = train_poster_classifier(images, labels)

    # Add interpretation
    if results["accuracy"] is not None:
        improvement = (results["accuracy"] - results["baseline"]) * 100
        if improvement > 15:
            results["interpretation"] = (
                f"Strong signal: {improvement:.1f}% above baseline. "
                "Poster aesthetics appear to carry independent signal."
            )
        elif improvement > 5:
            results["interpretation"] = (
                f"Moderate signal: {improvement:.1f}% above baseline. "
                "Poster features provide some predictive value."
            )
        elif improvement > 0:
            results["interpretation"] = (
                f"Weak signal: {improvement:.1f}% above baseline. "
                "Poster aesthetics have limited independent predictive power."
            )
        else:
            results["interpretation"] = (
                f"No signal: accuracy ({results['accuracy']*100:.1f}%) "
                "at or below baseline. "
                "Poster aesthetics do not predict box office success "
                "with these features."
            )

    return results
