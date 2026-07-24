import numpy as np

class BatchNorm1D:

    def __init__(self, num_features, momentum=.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.mu_running = np.zeros(num_features)
        self.var_running = np.ones(num_features)

    def forward(self, X, training=True):
        """
        X (m, n)
        """
        if training:
            mu_batch = np.mean(X, axis=0) # (n,)
            var_batch = np.var(X, axis=0) # (n,)

            self.mu_running = self.momentum*self.mu_running + (1 - self.momentum)*mu_batch
            self.var_running = self.momentum*self.var_running + (1 - self.momentum)*var_batch

            mu = mu_batch
            var = var_batch

        else:
            mu = self.mu_running
            var = self.var_running

        X_hat = (X - mu) / np.sqrt(var + self.eps) # (m, n)
        out = self.gamma * X_hat + self.beta

        return out