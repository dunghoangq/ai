'''
Include these methods:
- HNSWNode
- HNSWIndex
- random_level()
- insert()
- search()
'''

import numpy as np
import random
import heapq

from .cosine_similarity import cosine_similarity

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
    
    # When inserting a node, we pressume the node already has level.
    def insert(self, node: HNSWNode):
        if self.entry_point is None:
            self.entry_point = node
            self.max_level = node.level
            self.nodes.append(node)
            return
        else:
            level = node.level
            node.neighbours = {l: [] for l in range(level+1)}

            # Brute force search
            # for layer in range(level + 1):
            #     neighbours = self.brute_force_nearest_nodes(node.embedding, layer, k=5)

            #     for neighbour in neighbours:
            #         self.connect(node, neighbour, layer)

            # Greedy + Beam
            current = self.entry_point

            for layer in range(self.max_level, level, -1):
                current = self.greedy_nearest_node(node.embedding, current, layer)

            for layer in range(level, -1, -1):
                neighbours = self.beam_nearest_nodes(node.embedding, current, level)
                for neighbour in neighbours:
                    self.connect(node, neighbour, layer)
                current = sorted(neighbours, key=lambda x: x[0])[-1]

            self.nodes.append(node)

            if level > self.max_level:
                self.max_level = level
                self.entry_point = node
    
    def nodes_at_level(self, level):
        return [
            node for node in self.nodes if node.level >= level
        ]
    
    # Use Greedy Search for previous layers, Beam Search for layer 0
    # for this method
    def brute_force_nearest_nodes(self, embedding, level, k=5):
        candidates = []

        for node in self.nodes_at_level(level):
            score = cosine_similarity(embedding, node.embedding)
            candidates.append((score, node))

        candidates.sort(reverse=True, key=lambda x: x[0])

        return [node for _, node in candidates[:k]]
    
    def greedy_nearest_node(query_embedding: np.ndarray, entry_node: HNSWNode, level: int):
        '''
        Greedy Search
        is used in searching nodes at level > 0 when inserting a node.
        '''

        current_node = entry_node
        current_score = cosine_similarity(query_embedding, current_node.embedding)

        while True:
            best_node = current_node
            best_score = current_score

            for neighbour in current_node.neighbours[level]:
                score = cosine_similarity(query_embedding, neighbour.embedding)
                if score > best_score:
                    best_node = neighbour
                    best_score = score
            
            if best_node == current_node:
                break

            current_node = best_node
            current_score = best_score
        
        return best_node


    def beam_nearest_nodes(query_embedding: np.ndarray, entry_node: HNSWNode, level: int, k=5):
        '''
        Beam Search
        is used to search nodes at level 0 when inserting a node.
        '''

        pq = []
        visited = set()
        candidates = []
        result = []

        heapq.heappush(pq, (cosine_similarity(query_embedding, entry_node.embedding), entry_node))

        while pq:
            _, node = heapq.heappop(pq)
            visited.append(node)
            heapq.heappush((score, node))

            for neighbour in node.neighbours[level]:
                if neighbour not in visited:
                    score = cosine_similarity(query_embedding, neighbour.embedding)
                    heapq.heappush(pq, (score, neighbour))

        candidates = heapq.nlargest(k, candidates)
        return [pair[1] for pair in candidates]
    
    # Helper to connect undirected graph
    def connect(self, node1, node2, level):
        node1.neighbours[level].append(node2)
        node2.neighbours[level].append(node1)