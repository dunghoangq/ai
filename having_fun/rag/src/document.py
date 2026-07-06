from dataclasses import dataclass

class DocumentChunk:
    
    def __init__(self, chunk_id, text, embedding):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding

@dataclass
class Document:
    id: int
    text: str
    matadata: dict
    score: float