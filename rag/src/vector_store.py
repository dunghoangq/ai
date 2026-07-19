from langchain_text_splitters import RecursiveCharacterTextSplitter
from .chunk import Chunk
from pathlib import Path
import pickle
import json
from .vector_index import VectorIndex
from .search_result import SearchResult

class VectorStore:

    def __init__(self, embedding_model, index):
        self.embedding_model = embedding_model
        self.index = index
        self.chunks = []

    def index_documents(self, documents, chunk_size=800, chunk_overlap=100):
        texts = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        for chunk_id, chunk in enumerate(chunks):
            chunk_obj = Chunk(
                chunk_id,
                chunk.page_content,
                chunk.metadata
            )
            self.chunks.append(chunk_obj)
            texts.append(chunk_obj.text)
        
        vectors = self.embedding_model.embed(texts)
        self.index.build(vectors)


    def search(self, query, k):
        query_vector = self.embedding_model.embed_query(query)
        ids, scores = self.index.search(query_vector, k)
        results = []

        for idx, score in zip(ids, scores):
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=score
            ))
        
        return results

    def save(self, path: str):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self.index.save(directory / "index.bin")
        
        with open(directory / "chunks.pkl", "wb") as f: # pickle uses binary mode
            pickle.dump(self.chunks, f)
        
        configs = {
            "embedding_model": self.embedding_model.model_name,
            "index_type": type(self.index).__name__,
            "dimension": self.index.dimension # index.d only works for FAISS, but not others (e.g., HNSW), expose it in VectorIndex
        }
        with open(directory / "config.json", "w") as f:
            json.dump(configs, f, indent=4)
    
    @classmethod
    def load(cls, path: str, embedding_model, index_class: type[VectorIndex]):
        """
        USAGE: store = VectorStore().load(path, Model, HNSWIndex)
        HNSWIndex is a class, not an object (HNSWIndex()).
        """
        directory = Path(path)
        with open(directory / "config.json", "r") as f:
            config = json.load(f)
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        if config["index_type"] != index_class.__name__:
            raise ValueError(f"Expected {config['index_type']}, got {type(index).__name__}")
        index = index_class.load(directory / "index.bin")

        if config["embedding_model"] != embedding_model.model_name:
            raise ValueError(
                f"Expected embedding model "
                f"{config['embedding_model']}, "
                f"got {embedding_model.model_name}"
            )
        store = cls(embedding_model, index)
        store.chunks = chunks
        return store