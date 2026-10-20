
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torchvision

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

SEED = 42
TRAIN_SIZE = 10_000
VAL_SIZE = 2_000
N_CLASSES = 10
N_PIXELS = 28 * 28

# Keep the project root independent of the working directory.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_split_indices(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation indices into the original MNIST train set."""

    all_indices = np.arange(len(labels))

    # First choose 12,000 examples while preserving class proportions.
    selected, _ = train_test_split(
        all_indices,
        train_size=TRAIN_SIZE + VAL_SIZE,
        stratify=labels,
        random_state=SEED,
    )

    # Split those 12,000 into the exact requested sizes.
    train_idx, val_idx = train_test_split(
        selected,
        train_size=TRAIN_SIZE,
        test_size=VAL_SIZE,
        stratify=labels[selected],
        random_state=SEED,
    )

    assert len(train_idx) == TRAIN_SIZE
    assert len(val_idx) == VAL_SIZE
    assert len(np.intersect1d(train_idx, val_idx)) == 0

    return train_idx, val_idx, selected


def main() -> None:
    # --------------------------------------------------------
    # 2. Load the documented dataset
    # --------------------------------------------------------

    train_ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )
    test_ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )

    # uint8 arrays preserve the original pixel values.
    X_train_raw = train_ds.data.numpy()
    y_train_all = train_ds.targets.numpy().astype(np.int64)

    X_test_raw = test_ds.data.numpy()
    y_test = test_ds.targets.numpy().astype(np.int64)

    assert X_train_raw.shape == (60_000, 28, 28)
    assert X_test_raw.shape == (10_000, 28, 28)
    assert X_train_raw.dtype == np.uint8
    assert X_test_raw.dtype == np.uint8

    # --------------------------------------------------------
    # 3. Explore the original data
    # --------------------------------------------------------

    summary = {
        "source": "torchvision.datasets.MNIST",
        "original_train_shape": list(X_train_raw.shape),
        "original_test_shape": list(X_test_raw.shape),
        "image_height": 28,
        "image_width": 28,
        "flattened_features": N_PIXELS,
        "original_pixel_dtype": str(X_train_raw.dtype),
        "original_pixel_min": int(X_train_raw.min()),
        "original_pixel_max": int(X_train_raw.max()),
        "train_class_counts": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(y_train_all, return_counts=True)
            )
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in zip(
                *np.unique(y_test, return_counts=True)
            )
        },
    }
    save_json(OUTPUT_DIR / "data_summary.json", summary)

    # Plot the original training-set class distribution.
    classes, counts = np.unique(
        y_train_all, return_counts=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(classes, counts)
    ax.set(
        xlabel="Digit class",
        ylabel="Number of examples",
        title="MNIST original training-set class distribution",
        xticks=range(10),
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "class_distribution.png", dpi=150
    )
    plt.close(fig)

    # Show example images from all ten classes.
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for digit, ax in enumerate(axes.flat):
        idx = int(np.flatnonzero(y_train_all == digit)[0])
        ax.imshow(X_train_raw[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Label: {digit}")
        ax.axis("off")

    fig.suptitle("MNIST examples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_images.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # 4. Create deterministic, stratified partitions
    # --------------------------------------------------------

    train_idx, val_idx, selected_idx = make_split_indices(
        y_train_all
    )

    # Save indices into the original 60,000-example training set.
    split_manifest = {
        "seed": SEED,
        "method": "two-stage stratified train_test_split",
        "index_reference": (
            "Zero-based positions in torchvision MNIST train=True"
        ),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "selected_12000_indices": selected_idx.tolist(),
        "test_indices": None,
        "test_definition": (
            "All 10,000 examples from torchvision MNIST train=False"
        ),
        "sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(y_test)),
        },
    }
    save_json(
        OUTPUT_DIR / "split_manifest.json",
        split_manifest,
    )

    # --------------------------------------------------------
    # 5. Flatten images and scale consistently
    # --------------------------------------------------------

    # Keep only the selected examples for development.
    X_train = (
        X_train_raw[train_idx].reshape(-1, N_PIXELS).astype(np.float32)
        / 255.0
    )
    X_val = (
        X_train_raw[val_idx].reshape(-1, N_PIXELS).astype(np.float32)
        / 255.0
    )

    # The official test set is never used for fitting.
    X_test = (
        X_test_raw.reshape(-1, N_PIXELS).astype(np.float32)
        / 255.0
    )

    y_train = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    assert X_train.shape == (10_000, 784)
    assert X_val.shape == (2_000, 784)
    assert X_test.shape == (10_000, 784)

    assert np.isfinite(X_train).all()
    assert 0.0 <= X_train.min() <= X_train.max() <= 1.0
    assert np.array_equal(np.unique(y_train), np.arange(10))

    preprocessing = {
        "flattening": "row-major 28x28 -> 784",
        "pixel_transform": "float32(pixel) / 255.0",
        "fitted_statistics": False,
        "fit_on": "training partition only",
        "feature_count": N_PIXELS,
        "feature_order": "row-major pixel order",
    }
    save_json(
        OUTPUT_DIR / "preprocessing_metadata.json",
        preprocessing,
    )

    # --------------------------------------------------------
    # 6. Fit the all-pixel logistic regression baseline
    # --------------------------------------------------------

    # Fixed configuration: no test-set tuning.
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        random_state=SEED,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    if not model.n_iter_.max() < model.max_iter:
        print(
            "WARNING: at least one class optimisation may have "
            "reached max_iter. Inspect convergence warnings."
        )

    # --------------------------------------------------------
    # 7. Evaluate validation first, then final test
    # --------------------------------------------------------

    def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
        start = time.perf_counter()
        predictions = model.predict(X)
        elapsed = time.perf_counter() - start

        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(
                f1_score(y, predictions, average="macro")
            ),
            "inference_seconds_total": float(elapsed),
            "inference_microseconds_per_example": float(
                elapsed / len(y) * 1_000_000
            ),
            "n_examples": int(len(y)),
            "classification_report": classification_report(
                y,
                predictions,
                labels=np.arange(10),
                output_dict=True,
                zero_division=0,
            ),
        }

    validation_metrics = evaluate(X_val, y_val)

    # Test metrics are reported once, after model configuration
    # and the validation protocol have been fixed.
    test_metrics = evaluate(X_test, y_test)

    # Approximate parameter storage for NumPy model coefficients.
    parameter_bytes = (
        model.coef_.nbytes + model.intercept_.nbytes
    )

    metrics = {
        "experiment": "A1_full_pixel_logistic_regression",
        "seed": SEED,
        "status": "executed",
        "model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "max_iter": 300,
            "n_iter": model.n_iter_.tolist(),
            "n_features": int(X_train.shape[1]),
            "n_classes": int(len(model.classes_)),
        },
        "training": {
            "n_examples": int(len(y_train)),
            "seconds": float(training_seconds),
            "parameter_array_bytes": int(parameter_bytes),
            "coefficient_shape": list(model.coef_.shape),
            "intercept_shape": list(model.intercept_.shape),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }

    save_json(
        OUTPUT_DIR / "baseline_metrics.json",
        metrics,
    )

    print("\nData shapes")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nValidation")
    print(json.dumps({
        "accuracy": validation_metrics["accuracy"],
        "macro_f1": validation_metrics["macro_f1"],
    }, indent=2))

    print("\nOfficial test")
    print(json.dumps({
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }, indent=2))

    print("\nTraining seconds:", training_seconds)
    print("Parameter array bytes:", parameter_bytes)
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
