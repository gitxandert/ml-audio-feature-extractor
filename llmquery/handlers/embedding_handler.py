import numpy as np
from indexer.metadata import get_metadata_by_id
from indexer.faiss_indexer import load_faiss_index, search_faiss_index

class FAISSIndex:
    __index = None

    @staticmethod
    def get_index():
        if FAISSIndex.__index == None:
            FAISSIndex.__index = load_faiss_index("data/index.faiss")
        return FAISSIndex.__index

def handle_embedding_query(query: str, embeddings: np.ndarray):
    index = FAISSIndex.get_index()

    indices, _ = search_faiss_index(index, embeddings)

    results = [get_metadata_by_id("data/metadata.sqlite", int(i)) for i in indices[0]]

    files = [r['file_path'] for r in results]

    return "\n".join(files)