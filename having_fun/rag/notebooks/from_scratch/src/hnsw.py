'''
Include these methods:
- HNSWNode
- HNSWIndex
- random_level()
- insert()
- search()
'''

import numpy as np
from .embedding import cosine_similarity
import random

class HNSWNode:

    def __init__(self, chunk_id: int, text: str, embedding: np.ndarray, level: int):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding
        self.level = level

        # now a dict {level: []}
        self.neighbours = {
            level: [] for level in range(level + 1)
        }

    # Helper to connect undirected graph
    def connect(self, node1, node2, level):
        node1.neighbours[level].append(node2)
        node2.neighbours[level].append(node1)


class HNSWIndex:

    '''
    Two things:
    1. A graph
    2. An insertion/search algorithm
    '''

    def __init__(self):
        self.entry_point = None
        self.max_level = -1
        self.nodes = []

    
    def random_level(self):
        level = 0

        while random.random() < 0.5:
            level += 1
        return level
    
    def insert(self, node: HNSWNode):
        if self.entry_point is None:
            node.level = self.random_level()
            self.entry_point = node
            self.max_level = node.level
            self.nodes.append(node)
            return
        else:
            level = self.random_level()
            node.level = level
            node.neighbours = {l: [] for l in range(level+1)}

            for layer in range(level + 1):
                neighbours = self.nearest_nodes(node.embedding, layer, k=5)

                for neighbour in neighbours:
                    self.connect(node, neighbour, layer)

            self.nodes.append(node)

            if level > self.max_level:
                self.max_level = level
                self.entry_point = node
    
    def nodes_at_level(self, level):
        return [
            node for node in self.nodes if node.level >= level
        ]
    
    def nearest_nodes(self, embedding, level, k=5):
        candidates = []

        for node in self.nodes_at_level(level):
            score = cosine_similarity(embedding, node.embedding)
            candidates.append((score, node))

        candidates.sort(reverse=True, key=lambda x: x[0])

        return [node for _, node in candidates[:k]]