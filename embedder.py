from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embedded = model.encode(chunks).tolist()
    print(f"✅ Embedded {len(embedded)} chunks.")
    return embedded