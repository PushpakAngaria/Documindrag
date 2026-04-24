import os
import chromadb
from chromadb.config import Settings
from src.embedder import Embedder

VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_db")

class VectorStore:
    """Manages ChromaDB vector storage and retrieval."""

    def __init__(self, collection_name: str = "documind_collection", persist_directory: str = VECTOR_DB_PATH):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.embedder = Embedder()

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Connected to collection '{collection_name}' at '{persist_directory}'")

    def add_documents(self, documents: list, source_name: str = "unknown"):
        """
        Add LangChain Document objects to the vector store.
        Each document chunk gets a unique ID, its embedding, and metadata.
        """
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_texts(texts)

        ids = []
        metadatas = []
        for i, doc in enumerate(documents):
            uid = f"{source_name}_{i}"
            ids.append(uid)
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["source"] = source_name
            meta["chunk_index"] = i
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"[VectorStore] Added {len(documents)} chunks from '{source_name}'.")
        return len(documents)

    def query(self, query_text: str, top_k: int = 5):
        """
        Perform semantic similarity search. Returns list of result dicts with
        keys: document, metadata, distance.
        """
        top_k = int(os.getenv("TOP_K_RESULTS", top_k))
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"]
        )
        threshold = float(os.getenv("DISTANCE_THRESHOLD", 0.8))
        print(f"[VectorStore] Query: '{query_text}' | Threshold: {threshold}")
        formatted = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                print(f"  - Chunk from '{meta.get('source')}': Distance {dist:.4f}")
                if dist <= threshold:
                    formatted.append({
                        "document": doc,
                        "metadata": meta,
                        "distance": dist
                    })
        
        if not formatted:
            print(f"[VectorStore] No chunks met the distance threshold.")
        
        return formatted

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.collection.count()

    def clear(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[VectorStore] Collection cleared.")

    def list_sources(self) -> list[str]:
        """Return unique source filenames stored in the vector store."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
