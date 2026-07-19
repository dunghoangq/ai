"""A small, NumPy-based loss library built from first principles.

Each loss has a PyTorch-like interface:
    loss = MSELoss(reduction="mean")
    value = loss(prediction, target)
    gradient = loss.backward(prediction, target)

`backward` returns d(loss) / d(prediction).  A later neural-network layer uses
that gradient to continue backpropagation through its own parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Base class defining the common loss-function contract."""

    _valid_reductions = {"none", "mean", "sum"}

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in self._valid_reductions:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return self.forward(prediction, target)

    def _reduce(self, values: np.ndarray) -> np.ndarray:
        if self.reduction == "none":
            return values
        if self.reduction == "sum":
            return np.sum(values)
        return np.mean(values)

    def _gradient_scale(self, size: int) -> float:
        """Account for whether the forward pass summed or averaged losses."""
        return 1.0 / size if self.reduction == "mean" else 1.0

    @abstractmethod
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the loss value."""

    @abstractmethod
    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the gradient with respect to `prediction`."""


class MSELoss(Loss):
    """Mean/sum elementwise squared error: (prediction - target)^2."""

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        return self._reduce(error ** 2)

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        return 2.0 * error * self._gradient_scale(error.size)


class MAELoss(Loss):
    """Mean/sum elementwise absolute error: |prediction - target|."""

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        return self._reduce(np.abs(error))

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        # At error = 0, |error| has no unique derivative.  We choose 0, a valid
        # subgradient and the conventional practical choice.
        return np.sign(error) * self._gradient_scale(error.size)


class HuberLoss(Loss):
    """Quadratic near zero and linear for large errors."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction)
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        absolute_error = np.abs(error)
        values = np.where(
            absolute_error <= self.delta,
            0.5 * error ** 2,
            self.delta * (absolute_error - 0.5 * self.delta),
        )
        return self._reduce(values)

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.asarray(prediction) - np.asarray(target)
        gradient = np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error))
        return gradient * self._gradient_scale(error.size)


class BCELoss(Loss):
    """Binary cross-entropy for probabilities already produced by sigmoid."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        super().__init__(reduction)
        self.eps = eps

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(prediction), self.eps, 1.0 - self.eps)
        target = np.asarray(target)
        values = -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        return self._reduce(values)

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(prediction), self.eps, 1.0 - self.eps)
        target = np.asarray(target)
        gradient = (probability - target) / (probability * (1.0 - probability))
        return gradient * self._gradient_scale(gradient.size)


class CrossEntropyLoss(Loss):
    """Stable multiclass cross-entropy from unnormalised logits.

    `prediction` has shape (batch_size, classes).  `target` is either integer
    class indices with shape (batch_size,) or one-hot targets of the same shape
    as prediction.
    """

    def _targets_as_probabilities(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        if logits.ndim != 2:
            raise ValueError("prediction must have shape (batch_size, classes)")
        target = np.asarray(target)
        if target.shape == logits.shape:
            return target.astype(float)
        if target.shape != (logits.shape[0],):
            raise ValueError("target must be class indices or one-hot labels")
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError("class-index targets must be integers")
        if np.any(target < 0) or np.any(target >= logits.shape[1]):
            raise ValueError("class index is outside the available classes")
        one_hot = np.zeros_like(logits, dtype=float)
        one_hot[np.arange(logits.shape[0]), target] = 1.0
        return one_hot

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        logits = np.asarray(prediction, dtype=float)
        targets = self._targets_as_probabilities(logits, target)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        per_example = -np.sum(targets * log_probabilities, axis=1)
        return self._reduce(per_example)

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        logits = np.asarray(prediction, dtype=float)
        targets = self._targets_as_probabilities(logits, target)
        gradient = self._softmax(logits) - targets
        # Cross entropy produces one loss per batch item, not per logit.
        return gradient * self._gradient_scale(logits.shape[0])
