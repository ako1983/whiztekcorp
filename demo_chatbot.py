"""
demo_chatbot.py
---------------
Minimal all-in-one chatbot demo (terminal-based) using your modular pipeline.
Requires: langchain, langchain-community, langchain-openai, openai, python-dotenv
"""

import os
from dotenv import load_dotenv

# PDF Loader (LangChain)
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter (LangChain)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store (LangChain + FAISS)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# OpenAI Chat API (latest)
import openai

# ---- ENVIRONMENT ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "You must set OPENAI_API_KEY in your .env file."

def load_pdf_pages(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")
    return docs

def split_all_docs(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = splitter.split_text(doc.page_content)
        all_chunks.extend(chunks)
    print(f"Split into {len(all_chunks)} text chunks.")
    return all_chunks

def build_index(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    from langchain_core.documents import Document
    docs = [Document(page_content=chunk) for chunk in chunks]
    store = FAISS.from_documents(docs, embeddings)
    print("Vector index built.")
    return store

def ask(store, query, top_k=5, model="gpt-4o"):
    context_chunks = store.similarity_search(query, k=top_k)
    context = "\n---\n".join([doc.page_content for doc in context_chunks])
    prompt = f"""You are a helpful assistant. Answer only using the following document context.

Context:
{context}

Question: {query}
Answer:"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def main():
    print("=== PDF Chatbot Terminal Demo ===")
    pdf_path = input("Enter the PDF filename (default: 2025 Member Handbook for HealthChoice Illinois - ABHIL_Member_Handbook.pdf): ").strip()
    if not pdf_path:
        pdf_path = "2025 Member Handbook for HealthChoice Illinois - ABHIL_Member_Handbook.pdf"
    if not os.path.exists(pdf_path):
        print("File not found!")
        return
    docs = load_pdf_pages(pdf_path)
    all_chunks = split_all_docs(docs)
    store = build_index(all_chunks)

    print("\nAsk questions about the document! (Type 'exit' to quit)\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        answer = ask(store, q)
        print(f"\nAnswer:\n{answer}\n" + "-"*60)

if __name__ == "__main__":
    main()
