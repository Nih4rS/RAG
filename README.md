# RAG: Hybrid Search, Reranking, and Extractive QA

A CPU-only Retrieval-Augmented Generation (RAG) system combining hybrid search (BM25 + FAISS), cross-encoder reranking, and extractive question answering.

## Features

- **PDF Ingestion**: Process PDF documents into searchable chunks
- **Hybrid Search**: Combines keyword-based (BM25) and semantic (FAISS) search
- **Reranking**: Cross-encoder model for improved relevance scoring
- **Extractive QA**: Precise answer extraction from retrieved passages
- **Streamlit UI**: User-friendly web interface
- **CPU-Only**: No GPU required

## Project Structure

```
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/               # user PDFs, sample docs
├─ store/              # saved indices 
├─ app/
│  ├─ app.py           # Streamlit UI
│  ├─ ingest.py        # PDF -> chunks pipeline
│  ├─ search.py        # BM25 + FAISS + hybrid merge
│  ├─ rerank.py        # bge reranker
│  ├─ qa.py            # extractive QA pipeline
│  ├─ eval_small.py    # evaluation script
│  └─ utils.py         # utility functions
├─ tests/
│  └─ test_ingest_search.py
└─ .streamlit/
   └─ config.toml
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nih4rS/RAG.git
cd RAG
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit UI

1. Add PDF files to the `data/` directory

2. Run the Streamlit app:
```bash
streamlit run app/app.py
```

3. Open your browser to `http://localhost:8501`

4. Click "Build/Rebuild Index" in the sidebar

5. Enter your query and click "Search" or "Search + QA"

### Evaluation Script

Run the evaluation script to test the pipeline:

```bash
python app/eval_small.py
```

### Programmatic Usage

```python
from app.ingest import process_directory
from app.search import HybridSearchEngine
from app.rerank import Reranker
from app.qa import ExtractiveQA

# Process documents
chunks, metadata = process_directory("data")

# Build search index
search_engine = HybridSearchEngine()
search_engine.build_index(chunks, metadata)

# Search
results = search_engine.hybrid_search("What is machine learning?", top_k=10)

# Rerank
reranker = Reranker()
reranked = reranker.rerank("What is machine learning?", results, top_k=5)

# Extract answer
qa = ExtractiveQA()
answers = qa.answer_with_search_results("What is machine learning?", reranked)
print(answers[0]['answer'])
```

## Testing

Run tests with pytest:

```bash
pytest tests/test_ingest_search.py -v
```

## Configuration

### Search Parameters

- **chunk_size**: Size of text chunks (default: 500 words)
- **overlap**: Overlap between chunks (default: 50 words)
- **alpha**: BM25 weight in hybrid search (0-1, default: 0.5)
- **top_k**: Number of results to retrieve

### Models

Default models (can be changed in code):
- **Embeddings**: `all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **QA**: `distilbert-base-cased-distilled-squad`

## How It Works

1. **Ingestion**: PDFs are parsed and split into overlapping chunks
2. **Indexing**: Chunks are indexed with both BM25 (keyword) and FAISS (semantic)
3. **Search**: Queries retrieve results using hybrid scoring (alpha * BM25 + (1-alpha) * FAISS)
4. **Reranking**: Cross-encoder model rescores top results for better relevance
5. **QA**: Extractive model extracts precise answers from top passages

## License

Apache License 2.0 - See LICENSE file for details
