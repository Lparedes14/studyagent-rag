"""Steps 3-5: embed chunks, then retrieve the most relevant ones for a query."""
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import chunk_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class NotesIndex:
    """An in-memory vector index over one document.

    Embeddings are L2-normalised, so a dot product equals cosine similarity.
    State lives on the instance (not in globals), so each user session
    can have its own index.
    """

    def __init__(self, model: SentenceTransformer | None = None):
        self.model = model or SentenceTransformer(EMBEDDING_MODEL)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    def build(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> int:
        """Chunk and embed text. Returns the number of chunks."""
        self.chunks = chunk_text(text, chunk_size, overlap)
        self.embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top_k chunks most similar to the query, best first."""
        if not self.ready:
            return []
        query_vec = self.model.encode(query, normalize_embeddings=True)
        scores = query_vec @ self.embeddings.T
        top = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top]
