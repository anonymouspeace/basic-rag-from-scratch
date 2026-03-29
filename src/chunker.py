import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunker(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    total_length = len(text)
    while start < total_length:
        end = min(start + chunk_size, total_length)
        chunks.append(text[start:end])
        if end == total_length:
            break
        start += chunk_size - overlap
    return chunks
