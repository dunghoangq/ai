from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Node:
    feature_idx: int | None = None    # feature index
    threshold: float | None = None    # threshold of the split: < threshold -> go left, else -> go right
    value: float | None = None        # value if node is a leaf
    left: Node | None = None
    right: Node | None = None

    # For learning purpose, debugging
    gain: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self):
        return self.value is not None