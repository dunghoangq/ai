from .compressor import Compressor
from .document import Document
import re

class SentenceCompressor(Compressor):
    
    def __init__(self, max_sentence=3):
        self.max_sentence = max_sentence

    def compress(self, query, documents):
        compressed = []
        keywords = query.lower().split()

        for doc in documents:
            sentences = re.split(r"(?<=[.!?])\s+", doc.text)
            sentence_scores = []
            
            for i, sentence in enumerate(sentences):
                score = sum(word in sentence.lower() for word in keywords)
                if score > 0:
                    sentence_scores.append((score, i, sentence))
            
            if not sentence_scores:
                compressed.append(doc)
                continue
            
            sentence_scores.sort(reverse=True, key=lambda x: x[0])
            selected = sentence_scores[:self.max_sentence]
            selected.sort(key=lambda x: x[1])
            text = " ".join(s for _, _, s in selected)

            compressed.append(Document(
                id=doc.id,
                text=text,
                metadata=doc.metadata,
                score=doc.score
            ))
        
        return compressed