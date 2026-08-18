from __future__ import annotations
import numpy as np
from tqdm import tqdm

from .tree import DecisionTree
from .binary_logistic_objective import BinaryLogisticObjective
from .quantile_binner import QuantileBinner

class XGBoostClassifier:

    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = .1,
            max_depth: int = 3,
            min_samples_split: int = 2,
            reg_lambda: float = 1.0,
            gamma: float = 0.0,
            min_child_weight: float = 0.0,
            max_bins: int = 256,
            objective: BinaryLogisticObjective | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_bins = max_bins
        self.objective = objective

        self.binner = QuantileBinner(max_bins=max_bins)
        self.trees = []
        self.base_score = None # F0
        self.training_loss_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostClassifier:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same n_samples.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Classifier requires binary labels 0 and 1.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X_binned = self.binner.fit_transform(X)

        # Fit trees -> corrections
        self.trees = []
        self.training_loss_ = []
        p = np.mean(y)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        self.base_score = np.log(p / (1 - p))
        raw_pred = np.full(len(y), self.base_score, dtype=np.float64)


        for _ in tqdm(range(self.n_estimators)):
            gradients = self.objective.gradient(y, raw_pred)
            hessians = self.objective.hessian(y, raw_pred)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                max_bins=self.max_bins,
            )

            tree.fit(X_binned, gradients, hessians)
            update = tree.predict(X_binned)
            raw_pred += self.learning_rate * update

            loss = self._log_loss(y, raw_pred)
            self.training_loss_.append(loss)

            self.trees.append(tree)

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_binned = self.binner.transform(X)

        raw_pred = np.full(X.shape[0], self.base_score, dtype=np.float64)

        for tree in self.trees:
            raw_pred += self.learning_rate * tree.predict(X_binned)

        return raw_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X = np.asarray(X, dtype=np.float64)
        # raw_pred = self.predict_raw(X)
        # return self._sigmoid(raw_pred)

        # sklearn style
        # Usage: model.predict_proba(X)[:, 1]
        X = np.asarray(X, dtype=np.float64)
        p1 = self._sigmoid(self.predict_raw(X))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold=.5) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int)

    def _log_loss(self, y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        p = self._sigmoid(raw_pred)
        eps = 1e-12

        return -np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))

    def _gradients(self, y_true: np.ndarray, raw_pred: np.ndarray):
        p = self._sigmoid(raw_pred)
        return p - y_true

    def _hessians(self, raw_pred: np.ndarray):
        p = self._sigmoid(raw_pred)
        return p * (1 - p)

    def _check_fitted(self):
        if self.base_score is None:
            raise RuntimeError("Model bust be fitted before prediction.")

    @staticmethod
    def _sigmoid(x: np.ndarray):
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))
