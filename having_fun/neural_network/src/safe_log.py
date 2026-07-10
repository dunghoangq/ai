import numpy as np

def safe_log(x, eps=1e-8):
    return np.log(x + eps)