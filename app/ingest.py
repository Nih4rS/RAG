"""PDF ingestion and chunking pipeline."""

import os
from typing import List, Tuple
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> Tuple[List[str], List[dict]]:
    """
    Process a PDF file into chunks with metadata.
    
    Returns:
        Tuple of (chunks, metadata_list)
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)
    
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            'source': os.path.basename(pdf_path),
            'chunk_id': i,
            'text': chunk
        })
    
    return chunks, metadata


def process_directory(data_dir: str, chunk_size: int = 500, overlap: int = 50) -> Tuple[List[str], List[dict]]:
    """
    Process all PDF files in a directory.
    
    Returns:
        Tuple of (all_chunks, all_metadata)
    """
    all_chunks = []
    all_metadata = []
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(data_dir, filename)
            chunks, metadata = process_pdf(pdf_path, chunk_size, overlap)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
    
    return all_chunks, all_metadata
