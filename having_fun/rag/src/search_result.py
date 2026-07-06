from dataclasses import dataclass
from .chunk import Chunk
import numpy as np

@dataclass
class SearchResult:
    chunk: Chunk
    score: np.float32