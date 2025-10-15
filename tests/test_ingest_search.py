"""Tests for ingestion and search functionality."""

import os
import sys
import tempfile
import pytest
import numpy as np
from PyPDF2 import PdfWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import chunk_text, extract_text_from_pdf, process_pdf
from app.search import HybridSearchEngine


class TestIngest:
    """Test ingestion functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        
        # Should have multiple chunks
        assert len(chunks) > 1
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert len(chunks) == 0
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "short text"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestSearch:
    """Test search functionality."""
    
    @pytest.fixture
    def search_engine(self):
        """Create a search engine instance."""
        return HybridSearchEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text and speech.",
            "Computer vision enables machines to interpret images.",
            "Reinforcement learning learns through trial and error."
        ]
        metadata = [
            {'source': 'doc1.pdf', 'chunk_id': i, 'text': chunk}
            for i, chunk in enumerate(chunks)
        ]
        return chunks, metadata
    
    def test_build_index(self, search_engine, sample_data):
        """Test building search index."""
        chunks, metadata = sample_data
        search_engine.build_index(chunks, metadata)
        
        assert search_engine.bm25 is not None
        assert search_engine.faiss_index is not None
        assert len(search_engine.chunks) == len(chunks)
    
    def test_bm25_search(self, search_engine, sample_data):
        """Test BM25 search."""
        chunks, metadata = sample_data
        search_engine.build_index(chunks, metadata)
        
        results = search_engine.search_bm25("machine learning", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(idx, (int, np.integer)) for idx, _ in results)
        assert all(isinstance(score, (float, np.floating)) for _, score in results)
    
    def test_faiss_search(self, search_engine, sample_data):
        """Test FAISS search."""
        chunks, metadata = sample_data
        search_engine.build_index(chunks, metadata)
        
        results = search_engine.search_faiss("machine learning", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(score, float) for _, score in results)
    
    def test_hybrid_search(self, search_engine, sample_data):
        """Test hybrid search."""
        chunks, metadata = sample_data
        search_engine.build_index(chunks, metadata)
        
        results = search_engine.hybrid_search("machine learning", top_k=3, alpha=0.5)
        
        assert len(results) <= 3
        assert all('chunk' in result for result in results)
        assert all('score' in result for result in results)
        assert all('metadata' in result for result in results)
    
    def test_hybrid_search_alpha_values(self, search_engine, sample_data):
        """Test hybrid search with different alpha values."""
        chunks, metadata = sample_data
        search_engine.build_index(chunks, metadata)
        
        # Test with pure BM25 (alpha=1)
        results_bm25 = search_engine.hybrid_search("machine learning", top_k=3, alpha=1.0)
        assert len(results_bm25) > 0
        
        # Test with pure FAISS (alpha=0)
        results_faiss = search_engine.hybrid_search("machine learning", top_k=3, alpha=0.0)
        assert len(results_faiss) > 0
        
        # Test with balanced (alpha=0.5)
        results_hybrid = search_engine.hybrid_search("machine learning", top_k=3, alpha=0.5)
        assert len(results_hybrid) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
