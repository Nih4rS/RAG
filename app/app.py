"""Streamlit UI for the RAG system."""

import os
import streamlit as st
from ingest import process_directory
from search import HybridSearchEngine
from rerank import Reranker
from qa import ExtractiveQA
from utils import save_object, load_object, ensure_dir


# Configuration
DATA_DIR = "data"
STORE_DIR = "store"
INDEX_PATH = os.path.join(STORE_DIR, "search_index.pkl")

ensure_dir(DATA_DIR)
ensure_dir(STORE_DIR)


@st.cache_resource
def load_models():
    """Load and cache models."""
    search_engine = HybridSearchEngine()
    reranker = Reranker()
    qa = ExtractiveQA()
    return search_engine, reranker, qa


def initialize_session_state():
    """Initialize session state variables."""
    if 'index_built' not in st.session_state:
        st.session_state.index_built = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG: Hybrid Search + Reranking + QA")
    st.markdown("*CPU-only Retrieval-Augmented Generation system*")
    
    initialize_session_state()
    search_engine, reranker, qa = load_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Document Processing")
        chunk_size = st.slider("Chunk size", 100, 1000, 500, 50)
        overlap = st.slider("Overlap", 0, 200, 50, 10)
        
        st.subheader("Search Settings")
        top_k_search = st.slider("Search results", 5, 50, 20, 5)
        alpha = st.slider("BM25 weight (α)", 0.0, 1.0, 0.5, 0.1)
        
        st.subheader("Reranking")
        top_k_rerank = st.slider("Rerank top-k", 3, 20, 5, 1)
        
        st.divider()
        
        # Index management
        st.header("📚 Index Management")
        
        pdf_files = []
        if os.path.exists(DATA_DIR):
            pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        
        st.write(f"PDFs in `{DATA_DIR}/`: **{len(pdf_files)}**")
        
        if st.button("🔨 Build/Rebuild Index", type="primary"):
            if not pdf_files:
                st.error(f"No PDF files found in `{DATA_DIR}/`")
            else:
                with st.spinner("Processing documents..."):
                    chunks, metadata = process_directory(DATA_DIR, chunk_size, overlap)
                    st.session_state.chunks = chunks
                    st.session_state.metadata = metadata
                    
                    search_engine.build_index(chunks, metadata)
                    st.session_state.index_built = True
                    
                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(pdf_files)} PDFs")
    
    # Main content area
    if not st.session_state.index_built:
        st.info("👈 Please build the index first using the sidebar.")
        st.markdown("""
        ### Getting Started
        
        1. Add PDF files to the `data/` directory
        2. Click **Build/Rebuild Index** in the sidebar
        3. Enter your query below
        
        ### Features
        
        - **Hybrid Search**: Combines BM25 (keyword) and FAISS (semantic) search
        - **Reranking**: Uses cross-encoder for better relevance
        - **Extractive QA**: Extracts precise answers from retrieved passages
        """)
        return
    
    st.success(f"✅ Index ready: {len(st.session_state.chunks)} chunks")
    
    # Query input
    st.subheader("🔎 Ask a Question")
    query = st.text_input(
        "Enter your question:",
        placeholder="What is machine learning?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        qa_button = st.button("💬 Search + QA", type="secondary")
    
    if query and (search_button or qa_button):
        # Perform search
        with st.spinner("Searching..."):
            search_results = search_engine.hybrid_search(
                query, 
                top_k=top_k_search, 
                alpha=alpha
            )
        
        # Rerank
        with st.spinner("Reranking..."):
            reranked = reranker.rerank(query, search_results, top_k=top_k_rerank)
        
        # Display results
        st.subheader(f"📄 Top {len(reranked)} Results")
        
        # QA if requested
        if qa_button:
            with st.spinner("Generating answer..."):
                answers = qa.answer_with_search_results(query, reranked, top_k=1)
            
            if answers:
                st.success("**Answer:**")
                answer_box = st.container()
                with answer_box:
                    st.markdown(f"### {answers[0]['answer']}")
                    st.caption(f"Confidence: {answers[0]['score']:.3f}")
                    
                    with st.expander("📖 Source Context"):
                        st.write(answers[0]['context'])
                        if 'metadata' in answers[0]:
                            st.json(answers[0]['metadata'])
                
                st.divider()
        
        # Display search results
        for i, result in enumerate(reranked):
            with st.expander(f"Result {i+1} - Score: {result.get('rerank_score', 0):.3f}"):
                st.write(result['chunk'])
                st.caption(f"Source: {result['metadata'].get('source', 'Unknown')}")
                st.caption(f"Chunk ID: {result['metadata'].get('chunk_id', 'N/A')}")
                if 'score' in result:
                    st.caption(f"Original search score: {result['score']:.3f}")


if __name__ == "__main__":
    main()
