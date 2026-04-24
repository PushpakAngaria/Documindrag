from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
import os

def test_pipeline():
    pdf_path = "genai project report.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    print(f"Testing pipeline with {pdf_path}...")
    
    # 1. Load document
    print("\n--- 1. Document Loading ---")
    loader = DocumentLoader()
    text = loader.process_document(pdf_path)
    print(f"Extracted {len(text)} characters of text.")
    print(f"First 100 characters: {text[:100]}...")
    
    # 2. Chunk text
    print("\n--- 2. Text Chunking ---")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    docs = chunker.chunk_text(text, metadata={"source": pdf_path})
    print(f"Created {len(docs)} chunks.")
    
    if docs:
        print(f"First chunk preview: {docs[0].page_content[:100]}...")
        print(f"First chunk metadata: {docs[0].metadata}")

if __name__ == "__main__":
    test_pipeline()
