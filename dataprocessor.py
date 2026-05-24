from pdfreader import read_pdf
from Chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
from typing import List

pdf_path = "./resources/HRPolicy.pdf"

def run():
    pages = read_pdf(pdf_path)
    
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)
    
    embedded_chunks = embed_chunks(chunks)

    store_in_pinecone(chunks, embedded_chunks, namespace="")

if __name__ == "__main__":
    run()