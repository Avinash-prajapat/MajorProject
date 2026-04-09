import os
import faiss
import pickle
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# =====================================
# EMBEDDING MODEL LOAD
# =====================================

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DB_BASE = "vector_db"


# =====================================
# CREATE VECTOR DATABASE
# =====================================

def create_vector_store(pdf_folder, mode):

    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    if not documents:
        print(f"No PDFs found in {pdf_folder}")
        return

    print("PDF Loaded:", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print("Chunks Created:", len(chunks))

    texts = [chunk.page_content for chunk in chunks]
    embeddings = MODEL.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    save_path = os.path.join(VECTOR_DB_BASE, mode)
    os.makedirs(save_path, exist_ok=True)

    faiss.write_index(index, f"{save_path}/faiss.index")

    with open(f"{save_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector DB created for {mode} ✅")


# =====================================
# SEARCH VECTOR DATABASE
# =====================================

def _search_db(question, selected_mode):

    db_path = os.path.join(VECTOR_DB_BASE, selected_mode)

    if not os.path.exists(f"{db_path}/faiss.index"):
        return "", []

    index = faiss.read_index(f"{db_path}/faiss.index")

    with open(f"{db_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    question_embedding = MODEL.encode([question])

    D, I = index.search(question_embedding, k=3)

    context = ""
    citations = []

    for i in I[0]:

        chunk = chunks[i]

        context += chunk.page_content + "\n\n"

        citations.append({
            "source": os.path.basename(
                chunk.metadata.get("source", "Unknown")
            ),
            "page": chunk.metadata.get("page", "N/A")
        })

    return context, citations


# =====================================
# NORMAL LLM (NO RETRIEVAL)
# =====================================

def normal_llm(question):

    try:

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        chat_completion = client.chat.completions.create(

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print("Groq Error:", e)
        return None


# =====================================
# LLM WITH CONTEXT
# =====================================

def generate_llm_answer(context, question):

    try:

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        chat_completion = client.chat.completions.create(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer strictly using the provided context. "
                        "If answer not present in context say "
                        "'Information not available in document.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],

            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        return chat_completion.choices[0].message.content

    except Exception as e:

        print("Groq Error:", e)

        return None


# =====================================
# MAIN QA FUNCTION
# =====================================

def ask_question(question, mode="rag_chunking"):

    # ----------------------
    # NORMAL LLM
    # ----------------------

    if mode == "normal_llm":

        answer = normal_llm(question)

        return {
            "answer": answer,
            "citations": []
        }

    # ----------------------
    # BASIC RAG
    # ----------------------

    elif mode == "basic_rag":

        context, citations = _search_db(question, "uploaded")

    # ----------------------
    # RAG + CHUNKING
    # ----------------------

    elif mode == "rag_chunking":

        context, citations = _search_db(question, "uploaded")

    # ----------------------
    # HYBRID RAG
    # ----------------------

    elif mode == "hybrid":

        context_u, citations_u = _search_db(question, "uploaded")
        context_o, citations_o = _search_db(question, "offline")

        context = context_u + "\n" + context_o
        citations = citations_u + citations_o

    else:

        return {
            "answer": "Invalid mode selected.",
            "citations": []
        }

    # ----------------------
    # NO CONTEXT FOUND
    # ----------------------

    if not context.strip():

        return {
            "answer": "⚠ No relevant content found.",
            "citations": []
        }

    # ----------------------
    # TRY LLM FIRST
    # ----------------------

    llm_answer = generate_llm_answer(context, question)

    if llm_answer:

        print("Groq LLM Used ✅")

        return {
            "answer": llm_answer.strip(),
            "citations": citations
        }

    # ----------------------
    # FALLBACK MODE
    # ----------------------

    print("Fallback Mode Activated ⚠")

    return {
        "answer": "⚠ LLM unavailable. Showing retrieved content:\n\n" + context[:1500],
        "citations": citations
    }