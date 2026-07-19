import numpy as np

class MSELoss:

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target)**2)
    
    def backward(self):
        n = self.pred.size
        return 2 * (self.pred - self.target) / n