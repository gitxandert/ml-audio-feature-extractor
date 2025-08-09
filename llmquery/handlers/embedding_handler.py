import numpy as np
from collections import defaultdict
from indexer.metadata import get_metadata_by_id
from indexer.faiss_indexer import load_faiss_index, search_faiss_index
from langchain_core.messages import HumanMessage

class FAISSIndex:
    __index = None

    @staticmethod
    def get_index():
        if FAISSIndex.__index == None:
            FAISSIndex.__index = load_faiss_index("data/index.faiss")
        return FAISSIndex.__index

def handle_embedding_query(query: HumanMessage, embeddings: np.ndarray):
    index = FAISSIndex.get_index()

    indices, _ = search_faiss_index(index, embeddings)

    results = [get_metadata_by_id("data/metadata.sqlite", int(i)) for i in indices[0]]

    files = defaultdict(int)
    for r in results:
        files[r['file_path']] += 1

    files.sort(key=lambda x: x['file_path'], reverse=True)

    return "\n".join(files.keys())

""" So far, the query here is pointless; only the embeddings that are most similar to the provided one are being returned.
    To make this replete, I'd also have to make a new metadata entry for the provided embedding and instruct the LLM
    to determine whether the query asks for similarity or comparison. For example, if the user provides a file and asks,
    "Which files are louder than this one?", the program should compare the new file's metadata to the stored files' metadata.
    If the user asks, "Which files sound most like this one?", then the FAISS index can be queried and respond with the
    corresponding tracks' file paths; however, a distinction should be made between *files* and *chunks*. For example, if
    the user asks "Does anything in here sound like this file?", the program should group chunks by file_path, responding with
    something like "0'50"-1'20" in <file_path> and 2'15"-2'30" in <other_file_path> sound like the new audio".
"""