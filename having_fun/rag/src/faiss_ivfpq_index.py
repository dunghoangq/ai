from .vector_index import VectorIndex
import faiss

class IVFPQIndex(VectorIndex):
    
    def __init__(self, nlist=100, m=32, nbits=8, metric=faiss.METRIC_L2):
        super().__init__()
        self.nlist = nlist
        self.m = m          # number of subvectors
        self.nbits = nbits  # bits used per subvector, nbits = 8 means 2^8 = 256 centroids
        self.metric = metric

    def build(self, embeddings):
        """
        USAGE:
        index = IVFPQIndex()
        index.build(embeddings)
        """
        d = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits, self.metric)
        self.index.train(embeddings)
        self.add(embeddings)

    @property
    def dimension(self):
        return self.index.d