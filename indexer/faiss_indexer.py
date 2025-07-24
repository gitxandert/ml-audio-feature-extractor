import numpy as np
import faiss
import os

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """build inner product FAISS index from L2-normalized embeddings"""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (num_samples, dim)")
    
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index

def save_faiss_index(index: faiss.Index, path: str):
    """save index to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    """load index from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)

def search_faiss_index(index: faiss.Index, query: np.ndarray, k: int = 5):
    """search index for top-k most similar embeddings"""
    if query.ndim == 1:
        query = query.reshape(1, -1)

    faiss.normalize_L2(query)
    scores, indices = index.search(query, k)
    return indices, scores