import numpy as np

class Posting:

    def __init__(self):
        self.vector = []
        self.id = ""
        self.text = ""

class IVFIndex:

    def __init__(self):
        self.centroids = np.ndarray()   # (k, d)
        self.inverted_list = {}         # dict[int, list[Posting]]

    def fit(self):
        ...

    def search(self, query_vector, k, nprobe):
        ...