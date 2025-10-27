"""Utility functions for the RAG system."""

import os
import pickle
from typing import Any


def save_object(obj: Any, filepath: str) -> None:
    """Save a Python object to disk using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath: str) -> Any:
    """Load a Python object from disk using pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)
