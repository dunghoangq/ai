from .vector_index import VectorIndex
import faiss

class IVFIndex(VectorIndex):
    
    def __init__(self, nlist=100, metric=faiss.METRIC_L2):
        super().__init__()
        self.nlist = nlist
        self.metric = metric # metric can be replaced with faiss.METRIC_INNER_PRODUCT
    
    def build(self, embeddings):
        """
        USAGE:
        index = IVFIndex()
        index.build(embeddings)
        """
        d = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, self.metric)
        self.index.train(embeddings)
        self.add(embeddings)

    @property
    def dimension(self):
        return self.index.d