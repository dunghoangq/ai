
import numpy as np

from src.step_a1 import make_split_indices


def test_split_sizes_and_disjointness():
    labels = np.repeat(np.arange(10), 1000)

    train_idx, val_idx, selected_idx = make_split_indices(labels)

    assert len(train_idx) == 10_000
    assert len(val_idx) == 2_000

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.unique(selected_idx)) == 12_000


def test_split_is_reproducible():
    labels = np.repeat(np.arange(10), 1000)

    first = make_split_indices(labels)
    second = make_split_indices(labels)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)
