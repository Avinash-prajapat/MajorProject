import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Import RAG Functions
from rag_pipeline import create_vector_store, ask_question


# =====================================
# FASTAPI INITIALIZE
# =====================================

app = FastAPI(title="Hallucination-Resistant Research Assistant")


# =====================================
# CORS (Frontend Connection)
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# FOLDERS
# =====================================

UPLOAD_DIR = "data/uploaded_pdfs"
OFFLINE_DIR = "data/offline_dataset"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OFFLINE_DIR, exist_ok=True)


# =====================================
# HOME ROUTE
# =====================================

@app.get("/")
def home():
    return {"message": "Backend Running ✅"}


# =====================================
# UPLOAD PDF API
# =====================================

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("PDF Saved:", file.filename)

        # Create Vector DB for uploaded PDFs
        create_vector_store(UPLOAD_DIR, mode="uploaded")

        return {
            "message": f"{file.filename} Uploaded and Indexed ✅"
        }

    except Exception as e:
        return {
            "message": "Upload Failed ❌",
            "error": str(e)
        }


# =====================================
# BUILD OFFLINE DATASET (Optional API)
# =====================================

@app.get("/build_offline")
def build_offline():

    try:
        create_vector_store(OFFLINE_DIR, mode="offline")

        return {
            "message": "Offline Dataset Indexed Successfully ✅"
        }

    except Exception as e:
        return {
            "message": "Offline Indexing Failed ❌",
            "error": str(e)
        }


# =====================================
# ASK QUESTION API
# =====================================

@app.post("/ask")
async def ask(data: dict):

    try:
        question = data.get("question")
        mode = data.get("mode", "uploaded")  # uploaded / offline / hybrid

        if not question:
            return {
                "answer": "⚠ Please provide a question.",
                "citations": []
            }

        result = ask_question(question, mode)

        return {
            "answer": result.get("answer"),
            "citations": result.get("citations", [])
        }

    except Exception as e:
        return {
            "answer": "Error occurred ❌",
            "citations": [],
            "error": str(e)
        }