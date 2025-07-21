"""
query_engine.py
---------------
Retrieves relevant context from your FAISS vector store and generates an LLM answer.
Uses OpenAI's ChatCompletion API with openai>=1.0.0.
"""

import os
from typing import List
from dotenv import load_dotenv
import openai

load_dotenv()  # Loads your .env with OPENAI_API_KEY

def ask_question(store, user_query: str, top_k: int = 5, model: str = "gpt-4o") -> str:
    """
    Retrieve relevant chunks and use OpenAI LLM to generate an answer.

    :param store: FAISS vector store (from vector_store.py)
    :param user_query: User's question
    :param top_k: Number of chunks to retrieve
    :param model: OpenAI Chat model to use (e.g., "gpt-4o")
    :return: Answer string
    """
    # Retrieve top-k relevant chunks
    context_chunks = store.similarity_search(user_query, k=top_k)
    context = "\n---\n".join([doc.page_content for doc in context_chunks])

    # Compose prompt
    prompt = f"""You are a helpful assistant answering questions using only the following document context.

Context:
{context}

Question: {user_query}
Answer:"""

    # New OpenAI API (1.0.0+)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    from vector_store import build_vector_store
    chunks = [
        "Medicaid covers doctor visits and emergency care.",
        "Members can change their PCP anytime.",
        "You get one free dental cleaning per year."
    ]
    store = build_vector_store(chunks)
    question = "How often can I get dental cleaning?"
    answer = ask_question(store, question, model="gpt-4o")
    print(f"Q: {question}\nA: {answer}")
