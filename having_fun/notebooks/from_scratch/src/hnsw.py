'''
Include these methods:
- HNSWNode
- HNSWIndex
- random_level()
- insert()
- search()
'''

import numpy as np
from embedding import cosine_similarity

class HNSWNode:

    def __init__(self, chunk_id: int, text: str, embedding: np.ndarray, level: int):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding
        self.level = level
        self.neighbours = {} # now a dict {level: []}

class HNSWIndex:

    '''
    Two things:
    1. A graph
    2. An insertion/search algorithm
    '''

    def __init__(self, entry_point: HNSWNode):
        self.entry_point = entry_point
        self.max_level
        self.nodes
    
    def insert(self, node: HNSWNode):
        