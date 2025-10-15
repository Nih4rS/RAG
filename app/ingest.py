from __future__ import annotations
import os, hashlib, logging
from typing import List, Dict
import fitz  # pymupdf

logger = logging.getLogger(__name__)

def read_pdf(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(path)
    texts = []
    for page in doc:
        try:
            texts.append(page.get_text("text"))
        except Exception as e:
            logger.warning("Failed to parse page: %s", e)
    return "\n".join(texts).strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    tokens = text.split()
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start = end - overlap
        if start < 0: start = 0
        if end == len(tokens): break
    return chunks

def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def to_documents(chunks: List[str], source_path: str) -> List[Dict]:
    h = content_hash(source_path)
    docs = []
    for i, c in enumerate(chunks):
        docs.append({"id": f"{h}-{i:05d}", "text": c, "source": os.path.basename(source_path)})
    return docs
