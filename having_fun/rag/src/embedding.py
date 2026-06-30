import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))