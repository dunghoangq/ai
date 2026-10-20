
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import sys
ROOT = Path.cwd().resolve().parents[2]
sys.path.append(str(ROOT))
from common.cur import cur_decompose, relative_frobenius_error


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

REQUIRED = [
    "split_manifest.json",
    "preprocessing_metadata.json",
    "data_summary.json",
    "baseline_metrics.json",
]

missing = [name for name in REQUIRED if not (OUT / name).exists()]
if missing:
    raise FileNotFoundError(
        "Required Step A1 artefacts are missing: "
        + ", ".join(missing)
        + ". Execute Step A1 first; do not create substitute splits."
    )

for name in REQUIRED:
    with (OUT / name).open() as f:
        json.load(f)

# These filenames must match the actual A1 exports.
array_names = [
    "X_train.npy", "X_val.npy",
    "y_train.npy", "y_val.npy",
]
missing = [name for name in array_names if not (OUT / name).exists()]
if missing:
    raise FileNotFoundError(
        "Expected arrays not found: "
        + ", ".join(missing)
        + ". Update the loader to use the actual A1 filenames."
    )

X_train = np.load(OUT / "X_train.npy")
X_val = np.load(OUT / "X_val.npy")
y_train = np.load(OUT / "y_train.npy")
y_val = np.load(OUT / "y_val.npy")

assert X_train.shape == (10000, 784)
assert X_val.shape == (2000, 784)
assert y_train.shape == (10000,)
assert y_val.shape == (2000,)
assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()

# Reconstruction uses training examples only.
grid = [
    (25, 100, "uniform"),
    (50, 200, "uniform"),
    (100, 400, "uniform"),
    (50, 200, "norm"),
]

reconstruction_records = []
for c, r, sampling in grid:
    start = time.perf_counter()
    result = cur_decompose(
        X_train, c=c, r=r, seed=42, sampling=sampling
    )
    elapsed = time.perf_counter() - start

    error = relative_frobenius_error(X_train, result.X_hat)

    reconstruction_records.append({
        "c": c,
        "r": r,
        "sampling": sampling,
        "seed": 42,
        "relative_frobenius_error": error,
        "runtime_seconds": elapsed,
        "C_shape": list(result.C.shape),
        "U_shape": list(result.U.shape),
        "R_shape": list(result.R.shape),
        "row_indices": result.row_indices.tolist(),
        "column_indices": result.column_indices.tolist(),
    })

# Pixel-feature classification is a separate experiment.
# Select columns using only the training partition.
feature_records = []
for c in (25, 50, 100):
    for sampling in ("uniform", "norm"):
        result = cur_decompose(
            X_train, c=c, r=min(200, len(X_train)),
            seed=42, sampling=sampling
        )

        columns = result.column_indices
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            random_state=42,
        )

        start = time.perf_counter()
        clf.fit(X_train[:, columns], y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(X_val[:, columns])
        predict_seconds = time.perf_counter() - start

        feature_records.append({
            "c": c,
            "sampling": sampling,
            "seed": 42,
            "selected_columns": columns.tolist(),
            "validation_accuracy": float(
                accuracy_score(y_val, pred)
            ),
            "validation_macro_f1": float(
                f1_score(y_val, pred, average="macro")
            ),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        })

with (OUT / "cur_reconstruction_metrics.json").open("w") as f:
    json.dump(reconstruction_records, f, indent=2)

with (OUT / "cur_validation_metrics.json").open("w") as f:
    json.dump(feature_records, f, indent=2)

print("Reconstruction experiments:")
for row in reconstruction_records:
    print(
        f"c={row['c']:3d} r={row['r']:3d} "
        f"sampling={row['sampling']:6s} "
        f"relative_error={row['relative_frobenius_error']:.6f} "
        f"runtime={row['runtime_seconds']:.3f}s"
    )

print("\nValidation classification:")
for row in feature_records:
    print(
        f"c={row['c']:3d} sampling={row['sampling']:6s} "
        f"accuracy={row['validation_accuracy']:.4f} "
        f"macro_f1={row['validation_macro_f1']:.4f}"
    )
