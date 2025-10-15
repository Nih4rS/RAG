"""Small evaluation script for the RAG system."""

import os
import sys
from typing import List, Dict
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import process_directory
from app.search import HybridSearchEngine
from app.rerank import Reranker
from app.qa import ExtractiveQA


def evaluate_pipeline(
    data_dir: str = "data",
    test_queries: List[str] = None,
    top_k: int = 5
) -> Dict:
    """
    Evaluate the RAG pipeline on test queries.
    
    Args:
        data_dir: Directory containing PDF files
        test_queries: List of test queries
        top_k: Number of results to retrieve
    
    Returns:
        Dictionary with evaluation metrics
    """
    if test_queries is None:
        test_queries = [
            "What is machine learning?",
            "How does neural network work?",
            "What are the applications of AI?"
        ]
    
    print("=" * 60)
    print("RAG System Evaluation")
    print("=" * 60)
    
    # Check if data directory exists and has PDFs
    if not os.path.exists(data_dir):
        print(f"\nWarning: Data directory '{data_dir}' not found.")
        print("Please add PDF files to the data directory.")
        return {}
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"\nWarning: No PDF files found in '{data_dir}'.")
        print("Please add PDF files to the data directory.")
        return {}
    
    print(f"\nFound {len(pdf_files)} PDF file(s) in '{data_dir}'")
    
    # Process documents
    print("\n1. Processing documents...")
    start_time = time.time()
    chunks, metadata = process_directory(data_dir)
    process_time = time.time() - start_time
    print(f"   Processed {len(chunks)} chunks in {process_time:.2f}s")
    
    # Build search index
    print("\n2. Building search index...")
    start_time = time.time()
    search_engine = HybridSearchEngine()
    search_engine.build_index(chunks, metadata)
    index_time = time.time() - start_time
    print(f"   Built index in {index_time:.2f}s")
    
    # Initialize reranker and QA
    print("\n3. Initializing reranker and QA...")
    reranker = Reranker()
    qa = ExtractiveQA()
    
    # Evaluate queries
    print("\n4. Evaluating test queries...")
    print("=" * 60)
    
    results = {
        'num_chunks': len(chunks),
        'process_time': process_time,
        'index_time': index_time,
        'queries': []
    }
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 60)
        
        # Search
        search_start = time.time()
        search_results = search_engine.hybrid_search(query, top_k=10)
        search_time = time.time() - search_start
        
        # Rerank
        rerank_start = time.time()
        reranked = reranker.rerank(query, search_results, top_k=top_k)
        rerank_time = time.time() - rerank_start
        
        # QA
        qa_start = time.time()
        answers = qa.answer_with_search_results(query, reranked, top_k=1)
        qa_time = time.time() - qa_start
        
        print(f"\nSearch time: {search_time:.3f}s")
        print(f"Rerank time: {rerank_time:.3f}s")
        print(f"QA time: {qa_time:.3f}s")
        print(f"Total time: {search_time + rerank_time + qa_time:.3f}s")
        
        if answers:
            print(f"\nAnswer: {answers[0]['answer']}")
            print(f"Confidence: {answers[0]['score']:.3f}")
        else:
            print("\nNo answer found.")
        
        results['queries'].append({
            'query': query,
            'search_time': search_time,
            'rerank_time': rerank_time,
            'qa_time': qa_time,
            'answer': answers[0] if answers else None
        })
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    evaluate_pipeline()
