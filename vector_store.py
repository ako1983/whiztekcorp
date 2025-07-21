"""
vector_store.py
---------------
Stores and retrieves text chunks using vector embeddings (FAISS + LangChain).
Loads OpenAI API key from a .env file for security.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List

# --- Load environment variables from .env ---
from dotenv import load_dotenv
load_dotenv()  # This will automatically find and load .env if it's in the same directory

# You can swap this for any other embeddings model if you prefer (e.g., HuggingFaceEmbeddings)
def build_vector_store(chunks: List[str], openai_api_key: str = None) -> FAISS:
    """
    Builds a FAISS vector store from a list of text chunks.

    :param chunks: List of strings (text chunks)
    :param openai_api_key: Your OpenAI API key (for embeddings). Defaults to env var.
    :return: FAISS vector store object
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or function argument.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    from langchain_core.documents import Document
    docs = [Document(page_content=chunk) for chunk in chunks]
    store = FAISS.from_documents(docs, embeddings)
    return store

def search_vector_store(store: FAISS, query: str, k: int = 5) -> List[str]:
    """
    Search for the most relevant chunks to the query.

    :param store: FAISS vector store
    :param query: User's question or search string
    :param k: Number of top results to return
    :return: List of relevant chunk strings
    """
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Example usage
if __name__ == "__main__":
    chunks = [
        "Medicaid covers doctor visits and emergency care.",
        "Members can change their PCP anytime.",
        "You get one free dental cleaning per year."
    ]
    store = build_vector_store(chunks)  # openai_api_key loaded from .env automatically
    query = "How often can I get dental cleaning?"
    results = search_vector_store(store, query, k=2)
    for idx, chunk in enumerate(results):
        print(f"Match {idx+1}: {chunk}")
