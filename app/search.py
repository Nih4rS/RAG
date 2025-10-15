from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

class HybridSearcher:
    def __init__(self, docs: List[Dict], emb_model: str = "intfloat/e5-small-v2"):
        self.docs = docs
        self.corpus = [d["text"] for d in docs]
        self.bm25 = BM25Okapi([c.split() for c in self.corpus])
        self.model = SentenceTransformer(emb_model, device="cpu")
        self.emb = self.model.encode(self.corpus, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def bm25_search(self, query: str, k: int) -> List[int]:
        scores = self.bm25.get_scores(query.split())
        return list(np.argsort(scores)[::-1][:k])

    def vec_search(self, query: str, k: int) -> List[int]:
        q = self.model.encode([query], normalize_embeddings=True)
        _, idx = self.index.search(q, k)
        return idx[0].tolist()

    def hybrid(self, query: str, k_bm25: int = 10, k_vec: int = 10, k_merge: int = 12) -> List[Tuple[int, float]]:
        bm = self.bm25_search(query, k_bm25)
        ve = self.vec_search(query, k_vec)
        merged = list(dict.fromkeys(bm + ve))  # stable dedup
        # simple score: reciprocal rank from bm25 + vec position
        pos = {i: 1/(1+bm.index(i)) if i in bm else 0 for i in merged}
        for i in merged:
            if i in ve:
                pos[i] += 1/(1+ve.index(i))
        ranked = sorted([(i, pos[i]) for i in merged], key=lambda x: x[1], reverse=True)
        return ranked[:k_merge]
