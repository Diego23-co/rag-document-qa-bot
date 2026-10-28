"""
Local text embeddings with all-MiniLM-L6-v2, run through ONNX Runtime.

ChromaDB ships this model in ONNX format, so we get the same vectors as
sentence-transformers without installing PyTorch (~1 GB of RAM saved).
The model (~80 MB) downloads to ~/.cache/chroma on first use.
"""

import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class Embedder:
    """Drop-in for SentenceTransformer("all-MiniLM-L6-v2"): encode(texts) returns a numpy array."""

    def __init__(self):
        self._model = ONNXMiniLM_L6_V2()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self._model(list(texts)), dtype=np.float32)
