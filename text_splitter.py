"""
text_splitter.py
----------------
Splits long texts (pages or documents) into manageable, context-preserving chunks.
Uses LangChain's RecursiveCharacterTextSplitter for smart chunking.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Splits a single string into chunks using sentence and character boundaries.

    :param text: Text to split
    :param chunk_size: Max characters per chunk
    :param chunk_overlap: Overlap between chunks
    :return: List of chunked strings
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)
    return chunks

# Example usage
if __name__ == "__main__":
    demo_text = (
        "This is the first sentence. Here is another! "
        "Yet another? Yes. This text should be split correctly and with overlap."
    )
    result = split_text(demo_text, chunk_size=60, chunk_overlap=10)
    for idx, chunk in enumerate(result):
        print(f"Chunk {idx + 1}:\n{chunk}\n")
