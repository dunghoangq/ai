from __future__ import annotations
import numpy as np
import pandas as pd

class PCAReconstruction:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None


    def fit(self, X: np.ndarray) -> PCAReconstruction:
        self.mean_ = np.mean(X, axis=0)
        X_centred = X - self.mean_
        n_samples = X.shape[0]
        covariance = (X_centred.T @ X_centred) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Reverse order of eigenvalues
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_centred = X - self.mean_
        return X_centred @ self.components_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_.T + self.mean_

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X)
        return self.inverse_transform(Z)

    def anomaly_score(self, X):
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat)**2, axis=1)

    def explain_variance_ratio(self):
        total_variance = self.eigenvalues_.sum()
        return self.eigenvalues_ / total_variance

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Scaler must be fitted before transform().")