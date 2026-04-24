from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """Handles breaking down large text into smaller, overlapping chunks for embedding."""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # We use RecursiveCharacterTextSplitter as it tries to keep paragraphs/sentences together
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text, metadata=None):
        """
        Splits a single text string into a list of Document objects with metadata.
        """
        if not text.strip():
            return []
            
        # Optional: Add metadata (like source file name)
        metadatas = [metadata] if metadata else None
        
        # Create documents directly from text
        documents = self.text_splitter.create_documents([text], metadatas=metadatas)
        return documents

    def chunk_documents(self, documents):
        """
        Splits a list of Langchain Document objects.
        """
        return self.text_splitter.split_documents(documents)
