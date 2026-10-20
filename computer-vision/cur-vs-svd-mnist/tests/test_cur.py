import os, sys
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1]
sys.path.append(str(ROOT))
import numpy as np
from common.cur import cur_decompose, relative_frobenius_error

def test_hand_calculated_error():
    X = np.array([[1., 2.], [3., 4.]])
    X_hat = np.array([[1., 2.], [3., 3.]])

    actual = relative_frobenius_error(X, X_hat)

    assert np.isclose(actual, 1 / np.sqrt(30))


def test_exact_reconstruction_error_is_zero():
    X = np.eye(3)
    assert relative_frobenius_error(X, X) == 0.0


def test_zero_matrix_error():
    X = np.zeros((2, 3))
    assert relative_frobenius_error(X, X) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        relative_frobenius_error(
            np.zeros((2, 3)),
            np.zeros((3, 2)),
        )


def test_cur_dimensions_and_finite_values():
    X = np.arange(60, dtype=float).reshape(10, 6)

    result = cur_decompose(X, c=3, r=4, seed=42)

    assert result.C.shape == (10, 3)
    assert result.U.shape == (3, 4)
    assert result.R.shape == (4, 6)
    assert result.X_hat.shape == X.shape
    assert np.isfinite(result.X_hat).all()


def test_sampling_is_reproducible():
    X = np.random.default_rng(1).normal(size=(20, 10))

    a = cur_decompose(X, c=4, r=5, seed=42)
    b = cur_decompose(X, c=4, r=5, seed=42)

    np.testing.assert_array_equal(
        a.row_indices, b.row_indices
    )
    np.testing.assert_array_equal(
        a.column_indices, b.column_indices
    )


def test_sampling_without_replacement():
    X = np.random.default_rng(1).normal(size=(20, 10))

    result = cur_decompose(X, c=4, r=5, seed=42)

    assert len(np.unique(result.column_indices)) == 4
    assert len(np.unique(result.row_indices)) == 5


def test_invalid_dimensions_raise():
    X = np.ones((5, 4))

    with pytest.raises(AssertionError):
        cur_decompose(X, c=5, r=2)


def test_norm_sampling():
    X = np.random.default_rng(7).normal(size=(20, 10))

    result = cur_decompose(
        X, c=4, r=5, seed=42, sampling="norm"
    )

    assert result.X_hat.shape == X.shape
    assert np.isfinite(result.U).all()