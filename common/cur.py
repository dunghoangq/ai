from dataclasses import dataclass
import numpy as np

@dataclass
class CURResult:
    C: np.ndarray
    U: np.ndarray
    R: np.ndarray
    row_indices: np.ndarray
    column_indices: np.ndarray
    X_hat: np.ndarray

def relative_frobenius_error(X: np.ndarray, X_hat:np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)

    assert X.shape == X_hat.shape
    assert np.isfinite(X).all()
    assert np.isfinite(X_hat).all()

    denominator = np.linalg.norm(X, ord="fro")
    numerator = np.linalg.norm(X - X_hat, ord="fro")

    if denominator == 0:
        if numerator == 0:
            return 0.0
        raise ValueError("Relative error is undefined for zero X.")

    return float(numerator / denominator)

def sample_indices(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unique indices without replacement."""
    weights = np.asarray(weights, dtype=np.float64)

    assert weights.ndim == 1
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() > 0
    assert 1 <= count <= len(weights)

    probabilities = weights / weights.sum()

    return rng.choice(
        len(weights),
        size=count,
        replace=False,
        p=probabilities,
    )


def cur_decompose(
    X: np.ndarray,
    c: int,
    r: int,
    seed: int = 42,
    sampling: str = "uniform",
) -> CURResult:
    """
    Select rows and columns from X and compute CUR.

    sampling:
      - "uniform": uniform sampling without replacement.
      - "norm": squared-norm-proportional sampling without replacement.
    """
    X = np.asarray(X, dtype=np.float64)

    assert X.ndim == 2
    assert np.isfinite(X).all()

    n, d = X.shape
    assert 1 <= c <= d
    assert 1 <= r <= n

    rng = np.random.default_rng(seed)

    if sampling == "uniform":
        column_indices = rng.choice(d, c, replace=False)
        row_indices = rng.choice(n, r, replace=False)

    elif sampling == "norm":
        # Larger column/row norms receive larger sampling probabilities.
        column_weights = np.sum(X * X, axis=0)
        row_weights = np.sum(X * X, axis=1)

        if column_weights.sum() == 0:
            column_weights = np.ones(d)
        if row_weights.sum() == 0:
            row_weights = np.ones(n)

        column_indices = sample_indices(
            column_weights, c, rng
        )
        row_indices = sample_indices(
            row_weights, r, rng
        )

    else:
        raise ValueError(
            "sampling must be 'uniform' or 'norm'"
        )

    assert len(np.unique(column_indices)) == c
    assert len(np.unique(row_indices)) == r
    assert np.all((column_indices >= 0) & (column_indices < d))
    assert np.all((row_indices >= 0) & (row_indices < n))

    # Actual columns and rows of the original matrix.
    C = X[:, column_indices]
    R = X[row_indices, :]

    # pinv handles rectangular and rank-deficient matrices.
    C_pinv = np.linalg.pinv(C)
    R_pinv = np.linalg.pinv(R)

    U = C_pinv @ X @ R_pinv
    X_hat = C @ U @ R

    assert C.shape == (n, c)
    assert U.shape == (c, r)
    assert R.shape == (r, d)
    assert X_hat.shape == X.shape
    assert np.isfinite(U).all()
    assert np.isfinite(X_hat).all()

    return CURResult(
        C=C,
        U=U,
        R=R,
        row_indices=row_indices,
        column_indices=column_indices,
        X_hat=X_hat,
    )