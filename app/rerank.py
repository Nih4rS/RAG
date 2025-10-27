"""Reranking using BGE reranker model."""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    """Reranker using BGE model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: The search query
            results: List of search results with 'chunk' field
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        # Prepare pairs for scoring
        pairs = [(query, result['chunk']) for result in results]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Add rerank scores to results
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
