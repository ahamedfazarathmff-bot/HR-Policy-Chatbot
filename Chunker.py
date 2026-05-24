from typing import List

def chunk_pages(pages: List[str], chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    chunks: List[str] = []

    for text in pages:
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end]
            last_period = chunk.rfind(". ")
            if last_period != -1 and end < n and (last_period > chunk_size * 0.5):
                end = start + last_period + 2
                chunk = text[start:end]
            chunks.append(chunk.strip())  # ✅ chunks, chunk இல்லை
            start = max(end - chunk_overlap, end)

    return chunks  # ✅ for loop வெளியே