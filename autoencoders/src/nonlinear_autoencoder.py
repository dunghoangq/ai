from __future__ import annotations
import numpy as np

class NonlinearAutoencoder:

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        rng = np.random.default_rng(seed)

        self.W1_ = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2 / input_dim) # He (Kaiming) initialisation
        self.b1_ = np.zeros((1, hidden_dim))

        self.W2_ = rng.standard_normal((hidden_dim, latent_dim)) * np.sqrt(2 / hidden_dim)
        self.b2_ = np.zeros((1, latent_dim))

        self.W3_ = rng.standard_normal((latent_dim, hidden_dim)) * np.sqrt(2 / latent_dim)
        self.b3_ = np.zeros((1, hidden_dim))

        self.W4_ = rng.standard_normal((hidden_dim, input_dim)) * np.sqrt(2 / hidden_dim)
        self.b4_ = np.zeros((1, input_dim))

        self.learning_rate = learning_rate
        self.seed = seed
        self.n_epochs_trained_ = 0

        self.cache = {}

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X = self._validate_X(X)

        # encode
        Z1 = X @ self.W1_ + self.b1_
        A1 = self.relu(Z1)

        Z2 = A1 @ self.W2_ + self.b2_
        H = self.relu(Z2)

        # decode
        Z3 = H @ self.W3_ + self.b3_
        A3 = self.relu(Z3)
        X_hat = A3 @ self.W4_ + self.b4_

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "H": H,
            "Z3": Z3,
            "A3": A3,
            "X_hat": X_hat
        }

        return X_hat, cache

    def backward(self, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        H = cache["H"]
        Z3 = cache["Z3"]
        A3 = cache["A3"]
        X_hat = cache["X_hat"]

        n, d = X.shape

        dZ4 = 2.0 * (X_hat - X) / (n*d)

        dW4 = A3.T @ dZ4
        db4 = np.sum(dZ4, axis=0, keepdims=True)
        dA3 = dZ4 @ self.W4_.T

        dZ3 = dA3 * self.relu_grad(Z3)

        dW3 = H.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dH = dZ3 @ self.W3_.T

        dZ2 = dH * self.relu_grad(Z2)

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2_.T

        dZ1 = dA1 * self.relu_grad(Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
            "dW3": dW3,
            "db3": db3,
            "dW4": dW4,
            "db4": db4
        }

    def step(self, grads: dict[str, np.ndarray]):
        self.W1_ -= self.learning_rate * grads["dW1"]
        self.b1_ -= self.learning_rate * grads["db1"]

        self.W2_ -= self.learning_rate * grads["dW2"]
        self.b2_ -= self.learning_rate * grads["db2"]

        self.W3_ -= self.learning_rate * grads["dW3"]
        self.b3_ -= self.learning_rate * grads["db3"]

        self.W4_ -= self.learning_rate * grads["dW4"]
        self.b4_ -= self.learning_rate * grads["db4"]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X_hat, _ = self.forward(X)
        return X_hat

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_grad(self, x):
        return (x > 0).astype(x.dtype)

    def reconstruction_loss(self, X: np.ndarray, X_hat: np.ndarray) -> float:
        X = self._validate_X(X)
        X_hat = self._validate_X(X_hat)
        return np.mean((X - X_hat)**2)

    def reconstruction_error(self, X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        X_hat = self._validate_X(X_hat)
        return np.mean((X - X_hat)**2, axis=1)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if len(X.shape) != 2:
            raise ValueError("X must have 2 dimensions.")

        if X.shape[0] == 0:
            raise ValueError("X must have at least 1 sample.")

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} dimensions, received {X.shape[1]}")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity.")

        return X

