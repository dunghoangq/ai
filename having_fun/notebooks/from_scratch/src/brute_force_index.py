from document import DocumentChunk
from embedding import cosine_similarity

class BruteForceVectorIndex:

    def __init__(self):
        self.documents = []
    
    def add(self, chunk: DocumentChunk):
        self.documents.append(chunk)
    
    def search(self, query_embedding):
        best_score = float("-inf")
        best_doc = None

        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)

            if score > best_score:
                best_score = score
                best_doc = doc
        
        return best_doc