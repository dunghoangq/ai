import numpy as np
from knn_graph import GraphNode
from embedding import cosine_similarity
import heapq # for priority queue

def beam_search(query_embedding: np.ndarray, root: GraphNode, k: int):
    queue = []
    visited = []
    result = []

    heapq.heappush(queue, (cosine_similarity(query_embedding, root.embedding), root))

    while queue is not None:
        _, node = queue.pop()
        visited.append(node)

        for neighbour in node.neighbours:
            if neighbour not in visited:
                score = cosine_similarity(root, neighbour.embedding)
                heapq.heappush(queue, (score, neighbour))

                result = heapq.nlargest(k, queue)
                heapq.heapify(result)
    
    return result