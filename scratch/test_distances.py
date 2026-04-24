import os
from dotenv import load_dotenv
load_dotenv()

from src.vector_store import VectorStore

vs = VectorStore()
query = "Why should I use UML?"
print(f"\nTesting query: {query}")
results = vs.query(query)
print(f"\nFound {len(results)} results above threshold.")
for r in results:
    print(f"  - Source: {r['metadata']['source']} | Distance: {r['distance']:.4f}")
