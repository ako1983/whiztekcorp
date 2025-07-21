"""
api_server.py
-------------
FastAPI backend for PDF chatbot: exposes endpoints for question/answer, PDF upload, and feedback.
"""

import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from vector_store import build_vector_store, search_vector_store
from query_engine import ask_question
from pdf_loader import load_pdf
from text_splitter import split_text

from typing import List

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF Chatbot API")

# Singleton store for loaded chunks (for demo/prototype; use DB in prod)
vector_store = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    """
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    # Load & chunk PDF
    pages = load_pdf(temp_path)
    all_chunks = []
    for page in pages:
        all_chunks.extend(split_text(page["text_content"]))
    global vector_store
    vector_store = build_vector_store(all_chunks)
    os.remove(temp_path)
    return {"message": f"Processed {len(all_chunks)} text chunks from {file.filename}."}

@app.post("/ask")
async def ask_api(request: Request):
    """
    Answer a user question using the latest loaded document.
    """
    if vector_store is None:
        return JSONResponse({"error": "No PDF loaded. Upload a document first."}, status_code=400)
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse({"error": "Missing 'question' in request."}, status_code=400)
    answer = ask_question(vector_store, question, top_k=5, model="gpt-4o")
    return {"question": question, "answer": answer}

# Optional: Feedback endpoint
feedback_log = []

@app.post("/feedback")
async def submit_feedback(
    question: str = Form(...),
    answer: str = Form(...),
    rating: str = Form(...),  # 'up' or 'down'
    comment: str = Form(None)
):
    entry = {
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment
    }
    feedback_log.append(entry)
    print(f"Feedback received: {entry}")
    return {"message": "Feedback submitted"}

# For running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
