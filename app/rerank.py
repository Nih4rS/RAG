from __future__ import annotations
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 6) -> List[Dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in ranked[:top_k]]
