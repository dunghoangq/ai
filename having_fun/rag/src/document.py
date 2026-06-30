class DocumentChunk:
    
    def __init__(self, chunk_id, text, embedding):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding