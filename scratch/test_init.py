import os
from dotenv import load_dotenv
load_dotenv()

print("Testing RAGPipeline initialization...")
try:
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    print("Pipeline initialized successfully!")
    stats = pipeline.get_stats()
    print(f"Stats: {stats}")
except Exception as e:
    import traceback
    print(f"Initialization failed: {e}")
    traceback.print_exc()
