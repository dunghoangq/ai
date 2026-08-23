from __future__ import annotations
import numpy as np

from .csrmatrix import BinnedCSRMatrix

class GradientHistogram:
    """
    Store sum of gradients and hessians for each bin.
    """

    def __init__(self, n_features: int, max_bins: int):
        self.n_features = n_features
        self.max_bins = max_bins

        self.gradient_sum = np.zeros((n_features, max_bins), dtype=np.float64)
        self.hessian_sum = np.zeros((n_features, max_bins), dtype=np.float64)
        self.count = np.zeros((n_features, max_bins), dtype=np.int32)

    @classmethod
    def build(cls, X_binned: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, max_bins: int) -> GradientHistogram:
        n_samples, n_features = X_binned.shape

        if len(gradients) != n_samples:
            raise ValueError("gradients must have n_samples")
        if len(hessians) != n_samples:
            raise ValueError("hessians must have n_samples")

        hist = cls(n_features, max_bins)

        # for i in range(n_samples):
        #     g = gradients[i]
        #     h = hessians[i]

        #     for feature_idx in range(n_features):
        #         bin_idx = X_binned[i, feature_idx]

        #         hist.gradient_sum[feature_idx, bin_idx] += g
        #         hist.hessian_sum[feature_idx, bin_idx] += h
        #         hist.count[feature_idx, bin_idx] += 1

        for feature_idx in range(n_features):
            bins = X_binned[:, feature_idx]

            np.add.at(hist.gradient_sum[feature_idx], bins, gradients)
            np.add.at(hist.hessian_sum[feature_idx], bins, hessians)
            np.add.at(hist.count[feature_idx], bins, 1)

        return hist

    @classmethod
    def build_from_csrmatrix(cls, X_binned: BinnedCSRMatrix, gradients: np.ndarray, hessians: np.ndarray, max_bins: int) -> GradientHistogram:
        n_samples, n_features = X_binned.shape
        
        if len(gradients) != n_samples:
            raise ValueError("gradients must have n_samples")
        if len(hessians) != n_samples:
            raise ValueError("hessians must have n_samples")

        hist = cls(n_features, max_bins)


        for i in range(n_samples):
            start = X_binned.indptr[i]
            end = X_binned.indptr[i + 1]

            for j in range(start, end):

                feature_idx = X_binned.indices[j]
                bin_idx = X_binned.bin_ids[j]

                hist.gradient_sum[feature_idx, bin_idx] += gradients[i]
                hist.hessian_sum[feature_idx, bin_idx] += hessians[i]
                hist.count[feature_idx, bin_idx] += 1

        return hist


    def total_gradients(self, feature_idx: int) -> float:
        if feature_idx >= self.n_features:
            raise IndexError("feature_idx exceed gradients index range")

        return np.sum(self.gradient_sum[feature_idx])

    def total_hessians(self, feature_idx: int):
        if feature_idx >= self.n_features:
            raise IndexError("feature_idx exceed hessians index range")

        return np.sum(self.hessian_sum[feature_idx])