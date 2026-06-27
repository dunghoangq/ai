import numpy as np
from knn_graph import GraphNode
from embedding import cosine_similarity
from collections import deque

'''
This is not truly a greedy search.
In Greedy search, each time it find a layer of nodes, get the one with highest score.
'''

def greedy_search(query_embedding: np.ndarray, root: GraphNode, k: int) -> list[GraphNode]:
    similarities = []

    queue = deque()
    queue.append(root)

    while queue is not None:
        node = queue.popleft()
        similarities.append((cosine_similarity(query_embedding, node.embedding), node))

        for neighbour in node.neighbours:
            queue.append(neighbour)
    
    return similarities.sort(key=lambda x: x[0])[-2:]