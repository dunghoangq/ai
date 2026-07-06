class RankFusion:
    """
    Reciprocal Rank Fusion (RRF)

    RRF(d) = \sum_i \frac{1}{k + r_i}

        - r_i = rank
        - k = 60 usually
    """

    def __init__(self, k=60):
        self.k = k

    def fuse(self, *rankings):
        scores = {}
        documents = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                documents[doc.id] = doc
                scores.setdefault(doc.id, 0)
                scores[doc.id] += 1 / (self.k + rank + 1)

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Update doc's score with fusion score
        for doc_id, fusion_score in ordered:
            documents[doc_id].score = fusion_score

        return [documents[doc_id] for doc_id, _ in ordered]