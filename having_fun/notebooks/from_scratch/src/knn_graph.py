from embedding import cosine_similarity

class GraphNode:

    def __init__(self, chunk_id, text, embedding):
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding
        self.neighbours = []
    
    def get_neighbours(self, graph: list, k: int):
        for node in graph:
            similarities = []

            for other_node in graph:
                if other_node == node:
                    continue

                score = cosine_similarity(node.embedding, other_node.embedding)
                similarities.append((score, other_node))
            
            sorted_similarities = sorted(similarities, key=lambda x: x[0])
            
            self.neighbours = [x[1] for x in sorted_similarities[-k:]]