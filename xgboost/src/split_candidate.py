from dataclasses import dataclass
import numpy as np
from .node import Node

@dataclass
class SplitCandidate:
    node: Node
    gain: float
    feature_idx: int
    bin_idx: int

    X: np.ndarray
    gradients: np.ndarray
    hessians: np.ndarray

    depth: int
    threshold: float

    left_mask: np.ndarray
    right_mask: np.ndarray