# 📘 PDF Chatbot

An intelligent, modular chatbot for interacting with PDF documents using vector search + OpenAI GPT.  
Now includes a simple all-in-one terminal demo (`demo_chatbot.py`) for quick end-to-end testing!

---

## 🚀 Features

- Upload and index PDF files
- Asks questions and retrieves grounded answers from document context (RAG)
- Uses LangChain + OpenAI for embeddings and LLM
- Simple terminal demo: No server required
- Modular: Re-usable loader, splitter, vector store, and query modules
- API-ready: FastAPI backend for web/app integration

---

## 🧱 Project Structure

```
pdf-chatbot/
│
├── pdf_loader.py           # Loads PDFs using LangChain
├── text_splitter.py        # Splits text into manageable chunks
├── vector_store.py         # Builds and searches vector DB (FAISS)
├── query_engine.py         # Runs OpenAI GPT on top-k retrieved chunks
├── api_server.py           # FastAPI backend (API for ask/upload/feedback)
├── demo_chatbot.py         # 🆕 All-in-one interactive terminal demo
├── requirements.txt        # All dependencies
├── .env.example            # Example .env file (copy and edit as .env)
└── README.md
```

---

## ⚡️ Quick Start (Demo)

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy and edit your `.env`:**
   ```bash
   cp .env.example .env
   # Edit and set your OpenAI API key
   ```

3. **Put your PDF in the same folder.**
   - Default: `2025 Member Handbook for HealthChoice Illinois - ABHIL_Member_Handbook.pdf`

4. **Run the terminal chatbot:**
   ```bash
   python demo_chatbot.py
   ```
   - Follow prompts to ask questions!

---

## 🖥️ FastAPI Server (Optional)

- Start the API:
  ```bash
  python api_server.py
  ```
- Go to `http://localhost:8000/docs` to upload a PDF and ask questions from your browser.

---

## 📦 Example `.env.example`

```env
OPENAI_API_KEY=sk-...
```

---

## 📝 Requirements

All required packages are in `requirements.txt` (includes LangChain, OpenAI, FAISS, dotenv, FastAPI, etc).
