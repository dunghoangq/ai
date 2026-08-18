from __future__ import annotations
import numpy as np

class QuantileBinner:
    """
    Used for continuous skewed numerical data.
    """

    def __init__(self, max_bins: int = 256):
        if max_bins < 2:
            raise ValueError("max_bins must be at least 2.")

        self.max_bins = max_bins
        self.cut_points_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> QuantileBinner:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2-dim.")

        n_samples, n_features = X.shape

        self.cut_points_ = []

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            values = values[~np.isnan(values)] # ignore NaN rows

            if len(values) == 0:
                self.cut_points_.append(np.array([], dtype=np.float64))
                continue

            quantiles = np.linspace(0, 1, self.max_bins + 1)[1:-1]
            cuts = np.quantile(values, quantiles)
            cuts = np.unique(cuts)

            self.cut_points_.append(cuts)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.cut_points_ is None:
            raise ValueError("transform requires fit() first")
        if X.ndim != 2:
            raise ValueError("X must be 2-dim.")
        if np.isnan(X).any():
            raise ValueError("NaN handling is not implemented yet.")

        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_features != len(self.cut_points_):
            raise ValueError("X must have the same n_features with fitted data.")

        X_binned = np.empty((n_samples, n_features), dtype=np.int32)

        for feature_idx in range(n_features):
            cuts = self.cut_points_[feature_idx]
            values = X[:, feature_idx]

            X_binned[:, feature_idx] = np.searchsorted(cuts, values, side="right")

        return X_binned

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_threshold(self, feature_idx: int, bin_idx: int) -> float:
        if self.cut_points_ is None:
            raise ValueError("Binner must be fitted first.")
        
        cuts = self.cut_points_[feature_idx]

        if bin_idx >= len(cuts):
            raise IndexError("bin_idx exceed index range")
        return cuts[bin_idx]