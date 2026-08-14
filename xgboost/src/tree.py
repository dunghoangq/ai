from __future__ import annotations
from typing import Optional
import numpy as np

from .node import Node
from .gradient_histogram import GradientHistogram
from .heap import MaxHeap
from .split_candidate import SplitCandidate

class DecisionTree:
    """
    Include methods:
        - fit()
        - _build_tree()
        - _find_best_split()
        - _split()
        - _leaf_value()
        - _score()
        - _gain()
        - predict()
        - _predict_one()
    """

    def __init__(
            self,
            max_depth: int = 3,
            min_samples_split: int = 2,
            reg_lambda: float = 1.0,
            gamma: float = 0.0,
            min_child_weight: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda    # lambda in the Omega regularisation of the Loss, and in score w
        self.gamma = gamma  # gamma in the Omega, penalising number of leaves
        self.min_child_weight = min_child_weight # compare to total Hessian in a child node, both left and right hessians >= min_child_weight, else don't split
        self.root = None

    # def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> DecisionTree:
    #     X = np.asarray(X, dtype=np.float64)
    #     gradients = np.asarray(gradients, dtype=np.float64)
    #     hessians = np.asarray(hessians, dtype=np.float64)

    #     if X.ndim != 2:
    #         raise ValueError("X must have shape (n_samples, n_features)")

    #     if len(gradients) != len(X):
    #         raise ValueError("Gradients must have the same n_samples as X")

    #     if len(hessians) != len(X):
    #         raise ValueError("Hessians must have the same n_samples as X")

    #     self.root = self._build_tree(X, gradients, hessians, depth=0)
    #     return self
    
    def _make_candidate(self, node: Node, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, depth: int) -> SplitCandidate | None:
        n_samples = len(X)

        if depth >= self.max_depth:
            return None
        if n_samples < self.min_samples_split:
            return None

        best_gain, best_feature, best_threshold, best_left_mask, best_right_mask = self._find_best_split(X, gradients, hessians)

        if best_feature is None:
            return None
        if best_gain <= 0:
            return None

        return SplitCandidate(
            node=node,
            X=X,
            gradients=gradients,
            hessians=hessians,
            depth=depth,
            gain=best_gain,
            feature_idx=best_feature,
            threshold=best_threshold,
            left_mask=best_left_mask,
            right_mask=best_right_mask
        )

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> DecisionTree:
        """
        Best-first with heap.
        """
        X = np.asarray(X, dtype=np.float64)
        gradients = np.asarray(gradients, dtype=np.float64)
        hessians = np.asarray(hessians, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dim")
        if len(gradients) != len(X):
            raise ValueError("gradients must have the same n_samples as X")
        if len(hessians) != len(X):
            raise ValueError("hessians must have the same n_samples as X")
        
        # Create root
        self.root = Node(value=self._leaf_value(gradients, hessians), n_samples=len(X))

        # Find root's best split
        heap = MaxHeap(key=lambda candidate: candidate.gain)

        root_candidate = self._make_candidate(node=self.root, X=X, gradients=gradients, hessians=hessians, depth=0)

        # Push candidate into heap
        if root_candidate is not None:
            heap.push(root_candidate)

        # While heap not empty
        while not heap.is_empty():
            # pop globally best candidate
            candidate = heap.pop()
            left_mask = candidate.left_mask
            right_mask = candidate.right_mask

            # split that node
            X_left = candidate.X[left_mask]
            g_left = candidate.gradients[left_mask]
            h_left = candidate.hessians[left_mask]

            X_right = candidate.X[right_mask]
            g_right = candidate.gradients[right_mask]
            h_right = candidate.hessians[right_mask]

            # evaluate left child
            # evaluate right child
            left_node = Node(value=self._leaf_value(g_left, h_left), n_samples=len(X_left))
            right_node = Node(value=self._leaf_value(g_right, h_right), n_samples=len(X_right))

            node = candidate.node

            node.feature_idx = candidate.feature_idx
            node.threshold = candidate.threshold
            node.gain = candidate.gain
            node.left = left_node
            node.right = right_node

            child_depth = candidate.depth + 1

            # push valid child candidates
            left_candidate = self._make_candidate(node=left_node, X=X_left, gradients=g_left, hessians=h_left, depth=child_depth)
            right_candidate = self._make_candidate(node=right_node, X=X_right, gradients=g_right, hessians=h_right, depth=child_depth)

            if left_candidate is not None:
                heap.push(left_candidate)
            if right_candidate is not None:
                heap.push(right_candidate)

        return self

    def _build_tree(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, depth: int = 0):
        n_samples = X.shape[0]

        if depth >= self.max_depth:
            return Node(value=self._leaf_value(gradients, hessians), n_samples=n_samples)

        if n_samples < self.min_samples_split:
            return Node(value=self._leaf_value(gradients, hessians), n_samples=n_samples)

        (best_gain, best_feature, best_threshold, best_left_mask, best_right_mask) = self._find_best_split(X, gradients, hessians)

        if best_feature is None or best_gain < 0:
            return Node(value=self._leaf_value(gradients, hessians), n_samples=n_samples)

        left_child = self._build_tree(X[best_left_mask], gradients[best_left_mask], hessians[best_left_mask], depth + 1)
        right_child = self._build_tree(X[best_right_mask], gradients[best_right_mask], hessians[best_right_mask], depth + 1)

        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gain=best_gain,
            n_samples=n_samples
        )

    def _leaf_value(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        G = np.sum(gradients)
        H = np.sum(hessians)

        return -G / (H + self.reg_lambda)

    def _find_best_split(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> tuple[float, Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]] | None:
        n_samples, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None

        G_total = np.sum(gradients)
        H_total = np.sum(hessians)

        parent_score = G_total**2 / (H_total + self.reg_lambda)

        for feature_idx in range(n_features):
            feature_val = X[:, feature_idx]
            order = np.argsort(feature_val)

            x_sorted = feature_val[order]
            g_sorted = gradients[order]
            h_sorted = hessians[order]

            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)

            valid_split_positions = np.where(x_sorted[:-1] != x_sorted[1:])[0]

            for i in valid_split_positions:
                G_left = G_prefix[i]
                H_left = H_prefix[i]

                G_right = G_total - G_left
                H_right = H_total - H_left

                if H_left < self.min_child_weight:
                    continue
                if H_right < self.min_child_weight:
                    continue

                left_score = G_left**2 / (H_left + self.reg_lambda)
                right_score = G_right**2 / (H_right + self.reg_lambda)

                gain = .5 * (left_score + right_score - parent_score) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (x_sorted[i] + x_sorted[i+1]) / 2

        if best_feature is not None:
            best_left_mask = X[:, best_feature] < best_threshold
            best_right_mask = ~best_left_mask
        
        return (
            best_gain,
            best_feature,
            best_threshold,
            best_left_mask,
            best_right_mask,
        )

    def _candidate_threshold(self, feature_values):
        values = np.unique(feature_values)

        if len(values) <= 1:
            return np.array([])

        return (values[:-1] + values[1:]) / 2

    def _split(self, X: np.ndarray, feature_idx: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask

        return left_mask, right_mask

    def _gain(self, g_left: np.ndarray, h_left: np.ndarray, g_right: np.ndarray, h_right: np.ndarray) -> float:
        G_left = np.sum(g_left)
        H_left = np.sum(h_left)

        G_right = np.sum(g_right)
        H_right = np.sum(h_right)

        G = G_left + G_right
        H = H_left + H_right

        gain = .5 * (G_left**2/(H_left + self.reg_lambda) + G_right**2/(H_right + self.reg_lambda) - G**2/(H + self.reg_lambda)) - self.gamma

        return gain

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] < node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _score(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """
        Compute score by plugging the optimal w* into the loss Gw + 0.5(H + lambda)w^2
        """
        G = np.sum(gradients)
        H = np.sum(hessians)

        return G**2 / (H + self.reg_lambda)

    def print_tree(self, node: Node | None = None, depth: int = 0):
        if node is None:
            node = self.root

        indent = "    " * depth

        if node.is_leaf:
            print(
                f"{indent}Leaf("
                f"value={node.value:.4f}, "
                f"n={node.n_samples})"
            )

            return

        print(
            f"{indent}"
            f"X[{node.feature_idx}] "
            f"< {node.threshold:.4f} "
            f"(gain={node.gain:.4f})"
        )

        self.print_tree(
            node.left,
            depth + 1
        )

        self.print_tree(
            node.right,
            depth + 1
        )

    def _find_best_split_histogram(self, histogram: GradientHistogram):
        """
        Bin-based best split
        """
        best_gain = -np.inf
        best_feature = None
        best_bin = None

        for feature_idx in range(histogram.n_features):
            G_total = np.sum(histogram.gradient_sum[feature_idx])
            H_total = np.sum(histogram.hessian_sum[feature_idx])

            G_left = 0.0
            H_left = 0.0

            for bin_idx in range(histogram.max_bins - 1):
                G_left += histogram.gradient_sum[feature_idx, bin_idx]
                H_left += histogram.hessian_sum[feature_idx, bin_idx]

                G_right = G_total - G_left
                H_right = H_total - H_left

                if H_left < self.min_child_weight:
                    continue
                if H_right < self.min_child_weight:
                    continue

                gain = self._gain(G_left, H_left, G_right, H_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_bin = bin_idx

        return best_feature, best_bin, best_gain