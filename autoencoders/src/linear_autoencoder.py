from __future__ import annotations
import numpy as np


class LinearAutoencoder:

    def __init__(self, input_dim: int, latent_dim: int, learning_rate: float = 1e-3, seed: int = 42):
        self.input_dim = input_dim
        if latent_dim <= 0 or latent_dim >= self.input_dim:
            raise ValueError("Latent dimension must between 0 and input dimension.")
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.seed = seed

        rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / (input_dim + latent_dim))

        self.We_ = rng.normal(loc=.0, scale=scale, size=(input_dim, latent_dim))
        self.be_ = np.zeros((1, latent_dim))

        self.Wd_ = rng.normal(loc=.0, scale=scale, size=(latent_dim, input_dim))
        self.bd_ = np.zeros((1, input_dim))

        self.cache_ = {
            "train_loss": [],
            "val_loss": []
        }

        self.n_epochs_trained_ = 0

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        return X @ self.We_ + self.be_

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate_X(X)

        Z = X @ self.We_ + self.be_
        X_hat = Z @ self.Wd_ + self.bd_

        return Z, X_hat

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        _, X_hat = self.forward(X)
        return X_hat

    def reconstruction_loss(self, X: np.ndarray, X_hat: np.ndarray) -> float:
        X = self._validate_X(X)
        X_hat = self._validate_X(X_hat)

        return float(np.mean((X_hat - X)**2))

    def backward(self, X: np.ndarray, Z: np.ndarray, X_hat: np.ndarray) -> dict[str, np.ndarray]:
        X = self._validate_X(X)
        X_hat = self._validate_X(X_hat)

        batch_size, n_features = X.shape

        if Z.shape != (batch_size, self.latent_dim):
            raise ValueError(f"Expected Z shape {(batch_size, n_features)}, received {Z.shape}")

        if X_hat.shape != X.shape:
            raise ValueError(f"Expected X_hat shape {(batch_size, n_features)}, received {X_hat.shape}")

        dX_hat = 2.0 * (X_hat - X) / (batch_size * n_features)

        dWd = Z.T @ dX_hat
        dbd = np.sum(dX_hat, axis=0, keepdims=True)

        dZ = dX_hat @ self.Wd_.T

        dWe = X.T @ dZ
        dbe = np.sum(dZ, axis=0, keepdims=True)

        return {
            "We": dWe,
            "be": dbe,
            "Wd": dWd,
            "bd": dbd
        }

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        expected_shapes = {
            "We": self.We_.shape,
            "be": self.be_.shape,
            "Wd": self.Wd_.shape,
            "bd": self.bd_.shape
        }

        for name, expected_shape in expected_shapes.items():
            if name not in gradients:
                raise KeyError(f"Missing gradient for {name}")
            if gradients[name].shape != expected_shape:
                raise ValueError(f"Expected {name} gradient shape {expected_shape}, received {gradients[name].shape}")
            if not np.isfinite(gradients[name]).all():
                raise FloatingPointError(f"{name} gradient contains NaN or infinity")

            self.We_ -= self.learning_rate * gradients["We"]
            self.be_ -= self.learning_rate * gradients["be"]
            self.Wd_ -= self.learning_rate * gradients["Wd"]
            self.bd_ -= self.learning_rate * gradients["bd"]

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        X_hat = self.reconstruct(X)

        return np.mean((X_hat - X)**2, axis=1)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite")

        X = self._validate_X(X)
        scores = self.anomaly_score(X)

        return (scores >= threshold).astype(int)

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, epochs: int = 500, batch_size: int = 256, patience: int = 30, min_delta: float = 1e-7, verbose_every: int = 25) -> LinearAutoencoder:
        X_train = self._validate_X(X_train)
        X_val = self._validate_X(X_val)

        if epochs <= 0:
            raise ValueError("Epochs must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if patience <= 0:
            raise ValueError("Patience must be positive")

        self.cache_ = {
            "train_loss": [],
            "val_loss": []
        }

        best_params = self._copy_parameters()
        best_val_loss = np.inf
        epochs_without_improvement = 0

        n_train = X_train.shape[0]
        rng = np.random.default_rng(self.seed)

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n_train)
            total_batch_loss = 0.0
            samples_seen = 0

            for start in range(0, n_train, batch_size):
                batch_idx = idx[start : start + batch_size]
                X_batch = X_train[batch_idx]
                Z, X_hat = self.forward(X_batch)
                batch_loss = self.reconstruction_loss(X_batch, X_hat)
                gradients = self.backward(X_batch, Z, X_hat)

                self.step(gradients)
                current_batch_size = X_batch.shape[0]

                total_batch_loss += (batch_loss * current_batch_size)
                samples_seen += current_batch_size

            train_loss = total_batch_loss / samples_seen

            val_reconstruction = self.reconstruct(X_val)
            val_loss = self.reconstruction_loss(X_val, val_reconstruction)

            self.cache_["train_loss"].append(train_loss)
            self.cache_["val_loss"].append(val_loss)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_params = self._copy_parameters()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (verbose_every > 0 and (epoch == 1 or epoch % verbose_every == 0)):
                print(
                    f"Epoch {epoch:4d} | "
                    f"train loss: {train_loss:.8f} | "
                    f"val loss: {val_loss:.8f}"
                )

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best validation loss: {best_val_loss:.8f}"
                )
                break

        # The last epoch is not necessarily the best epoch.
        self._restore_parameters(best_params)
        self.best_val_loss_ = best_val_loss
        self.n_epochs_trained_ = epoch

        return self
            

    def _copy_parameters(self) -> dict[str, np.ndarray]:
        return {
            "We": self.We_.copy(),
            "be": self.be_.copy(),
            "Wd": self.Wd_.copy(),
            "bd": self.bd_.copy(),
        }

    def _restore_parameters(self, parameters: dict[str, np.ndarray]) -> None:
        self.We_ = parameters["We"].copy()
        self.be_ = parameters["be"].copy()
        self.Wd_ = parameters["Wd"].copy()
        self.bd_ = parameters["bd"].copy()

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_sanmples, input_dim)")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {X.shape[1]}")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        return X