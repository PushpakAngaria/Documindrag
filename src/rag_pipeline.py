import os
from src.document_loader import DocumentLoader
from src.ocr_processor import OCRProcessor
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.llm_chain import LLMChain

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

class RAGPipeline:
    """Orchestrates the full Retrieval-Augmented Generation pipeline."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.ocr = OCRProcessor()
        self.chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
        self.vector_store = VectorStore()
        self.llm = LLMChain()
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    def ingest_document(self, file_path: str, filename: str, use_ocr: bool = False) -> dict:
        """
        Full ingestion pipeline: load → chunk → embed → store.
        Returns a status dict.
        """
        try:
            # Step 1: Extract text
            if use_ocr:
                text = self.ocr.process_document(file_path, is_pdf=filename.lower().endswith(".pdf"))
            else:
                text = self.loader.process_document(file_path)

            if not text or not text.strip():
                return {"success": False, "error": "No text could be extracted from the document.", "chunks": 0}

            # Step 2: Chunk text
            docs = self.chunker.chunk_text(text, metadata={"source": filename})
            if not docs:
                return {"success": False, "error": "Document could not be chunked (empty after splitting).", "chunks": 0}

            # Step 3: Store in vector DB
            count = self.vector_store.add_documents(docs, source_name=filename)

            return {
                "success": True,
                "chunks": count,
                "characters": len(text),
                "filename": filename
            }
        except Exception as e:
            return {"success": False, "error": str(e), "chunks": 0}

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full query pipeline: embed query → retrieve → generate answer.
        """
        try:
            if self.vector_store.count() == 0:
                return {
                    "answer": "No documents have been uploaded yet. Please upload a PDF first.",
                    "sources": [],
                    "chunks_used": 0,
                    "retrieved_chunks": []
                }

            # Step 1: Retrieve relevant chunks
            retrieved = self.vector_store.query(question, top_k=top_k)

            if not retrieved:
                return {
                    "answer": "I couldn't find information about that in your uploaded documents. Please try a different question or upload more documents.",
                    "sources": [],
                    "chunks_used": 0,
                    "retrieved_chunks": []
                }

            # Step 2: Generate answer
            result = self.llm.answer(question, retrieved)
            result["retrieved_chunks"] = retrieved
            return result

        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "chunks_used": 0,
                "retrieved_chunks": []
            }

    def get_stats(self) -> dict:
        """Return current pipeline stats."""
        return {
            "total_chunks": self.vector_store.count(),
            "sources": self.vector_store.list_sources()
        }

    def clear_knowledge_base(self):
        """Clear all stored documents."""
        self.vector_store.clear()
        self.llm.reset_history()

    def reset_conversation(self):
        """Reset only conversation history, keep documents."""
        self.llm.reset_history()
