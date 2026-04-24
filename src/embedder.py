import os
from sentence_transformers import SentenceTransformer

class Embedder:
    """Generates dense vector embeddings using HuggingFace sentence-transformers."""

    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[Embedder] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"[Embedder] Model loaded successfully.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings and return a list of float vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()
