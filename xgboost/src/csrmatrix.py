from __future__ import annotations
import numpy as np

class CSRMatrix:
    """
    Only use on one-hot encoded categorical vars, because 0's represent structral sparsity.
    Don't use on numerical values where 0 has its meaning (e.g., txn_amount)
    """

    def __init__(
            self,
            data: np.ndarray,
            indices: np.ndarray,
            indptr: np.ndarray,
            shape: tuple[int, int],
    ):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.shape = shape

    @classmethod
    def from_dense(cls, X: np.ndarray) -> CSRMatrix:
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be 2-dim")

        n_rows, n_cols = X.shape

        data = []
        indices = []
        indptr = [0]

        for i in range(n_rows):
            for j in range(n_cols):
                if X[i][j] != 0:
                    data.append(X[i][j])
                    indices.append(j)
            indptr.append(len(data))

        return cls(
            data=np.asarray(data, dtype=np.float64),
            indices=np.asarray(indices, dtype=np.int32),
            indptr=np.asarray(indptr, dtype=np.int32),
            shape=(n_rows, n_cols),
        )

    def get_row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        start = self.indptr[row]
        end = self.indptr[row + 1]

        return (
            self.indices[start:end],
            self.data[start:end],
        )

    def to_dense(self) -> np.ndarray:
        n_rows, n_cols = self.shape

        X = np.zeros((n_rows, n_cols), dtype=self.data.dtype)

        for row in range(n_rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]

            cols = self.indices[start:end]
            values = self.data[start:end]
            X[row, cols] = values

        return X

    @property
    def nnz(self):
        """
        Number of Non-zero elements
        """
        return len(self.data)


class BinnedCSRMatrix:
    def __init__(
        self,
        bin_ids: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
    ):
        self.bin_ids = np.asarray(bin_ids, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.shape = shape