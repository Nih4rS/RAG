"""Hybrid search using BM25 and FAISS."""

import numpy as np
import faiss
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridSearchEngine:
    """Hybrid search engine combining BM25 and FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the hybrid search engine."""
        self.model = SentenceTransformer(model_name)
        self.bm25 = None
        self.faiss_index = None
        self.chunks = []
        self.metadata = []
        self.tokenized_chunks = []
        
    def build_index(self, chunks: List[str], metadata: List[dict]) -> None:
        """Build BM25 and FAISS indices from chunks."""
        self.chunks = chunks
        self.metadata = metadata
        
        # Build BM25 index
        self.tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        # Build FAISS index
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def search_faiss(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS."""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and FAISS.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for BM25 (1-alpha for FAISS). Range [0, 1]
        
        Returns:
            List of search results with scores and metadata
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k * 2)
        faiss_results = self.search_faiss(query, top_k * 2)
        
        # Normalize scores
        bm25_scores = {idx: score for idx, score in bm25_results}
        faiss_scores = {idx: score for idx, score in faiss_results}
        
        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores
        combined_scores = {}
        all_indices = set(bm25_scores.keys()) | set(faiss_scores.keys())
        
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0)
            faiss_score = faiss_scores.get(idx, 0)
            combined_scores[idx] = alpha * bm25_score + (1 - alpha) * faiss_score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Prepare results
        results = []
        for idx, score in sorted_indices:
            results.append({
                'chunk': self.chunks[idx],
                'metadata': self.metadata[idx],
                'score': score,
                'index': idx
            })
        
        return results
