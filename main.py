import os
import secrets
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import cyborgdb_core as cyborgdb  # CyborgDB Embedded

import google.generativeai as genai  # pip install google-generativeai


# ============================================================
# Config
# ============================================================

INDEX_NAME = "patient_notes"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim embeddings

# 🔐 Cyborg API key (for hackathon demo – local embedded mode)
API_KEY = "cyborg_59e5f2a614674d548a25f34423d6f39b"

# Gemini config (you said you have Gemini Pro)
# Set this in your shell before running uvicorn:
#   export GEMINI_API_KEY="YOUR_GEMINI_KEY"
GEMINI_MODEL = "gemini-pro"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Secure Healthcare Assistant")
# Allow frontend (http://127.0.0.1:*) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for demo; in prod restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load local embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()


# ============================================================
# CyborgDB init
# ============================================================

def init_cyborg_index() -> cyborgdb.EncryptedIndex:
    """
    Initialize an in-memory encrypted index for the hackathon demo.
    """
    index_location = cyborgdb.DBConfig("memory")
    config_location = cyborgdb.DBConfig("memory")
    items_location = cyborgdb.DBConfig("memory")

    # Client: api_key must be first positional argument
    client = cyborgdb.Client(
        API_KEY,
        index_location=index_location,
        config_location=config_location,
        items_location=items_location,
    )

    # Fresh 32-byte key per run (OK for demo; persist for real use)
    index_key = secrets.token_bytes(32)
    index_config = cyborgdb.IndexIVFFlat(dimension=EMBED_DIM)

    index = client.create_index(
        index_name=INDEX_NAME,
        index_key=index_key,
        index_config=index_config,
    )

    return index


index = init_cyborg_index()


# ============================================================
# Pydantic models
# ============================================================

class Note(BaseModel):
    patient_id: str
    note_id: Optional[str] = None
    text: str


class IngestRequest(BaseModel):
    notes: List[Note]


class QueryRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    top_k: int = 3


# ============================================================
# Helper
# ============================================================

def embed_text(text: str) -> List[float]:
    vec = embed_model.encode(text)
    return [float(x) for x in vec]


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Secure Healthcare Assistant up & running 🔐",
        "index": INDEX_NAME,
        "embedding_dim": EMBED_DIM,
        "gemini_model": GEMINI_MODEL if GEMINI_API_KEY else "disabled (no GEMINI_API_KEY)",
    }


# --------------- Ingest synthetic notes ----------------------

@app.post("/ingest")
def ingest_notes(req: IngestRequest):
    """
    Take synthetic patient notes, embed them locally,
    and store encrypted vectors + contents in CyborgDB.
    """
    items: List[Dict[str, Any]] = []

    for n in req.notes:
        vec = embed_text(n.text)
        note_id = n.note_id or f"{n.patient_id}_{secrets.token_hex(4)}"

        items.append(
            {
                "id": note_id,
                "vector": vec,
                "contents": n.text,  # encrypted by CyborgDB internally
                "metadata": {
                    "patient_id": n.patient_id,
                    "note_type": "visit_note",
                },
            }
        )

    try:
        index.upsert(items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {e}")

    return {"status": "ok", "ingested": len(items)}


# --------------- Simple query endpoint -----------------------

@app.post("/query")
def query_notes(req: QueryRequest):
    """
    Lower-level endpoint:
    1) Embed the doctor's question
    2) Encrypted vector search in CyborgDB
    3) Decrypt and return notes + a simple summary
    """
    try:
        query_vec = embed_text(req.question)

        filters = {"patient_id": {"$eq": req.patient_id}} if req.patient_id else None

        raw_results = index.query(
            query_vectors=[query_vec],   # NOTE: query_vectors (plural)
            top_k=req.top_k,
            n_probes=5,
            filters=filters,
            include=["distance", "metadata"],
        )

        if not raw_results:
            return {
                "answer": "No matching notes found for this query.",
                "results": [],
            }

        # Some versions return [[hits]], others [hits]
        if isinstance(raw_results[0], list):
            hits = raw_results[0]
        else:
            hits = raw_results

        if not hits:
            return {
                "answer": "No matching notes found for this query.",
                "results": [],
            }

        ids = [h.get("id") for h in hits if "id" in h]
        if not ids:
            return {
                "answer": "No matching notes found for this query.",
                "results": [],
            }

        docs = index.get(ids=ids, include=["contents", "metadata"])

        if isinstance(docs, dict):
            doc_list = [docs]
        else:
            doc_list = docs

        contexts: List[str] = []
        results_out: List[Dict[str, Any]] = []

        for d in doc_list:
            meta = d.get("metadata", {}) or {}
            contents = d.get("contents", "") or ""
            pid = meta.get("patient_id", "Unknown")
            note_type = meta.get("note_type", "note")

            contexts.append(f"[{pid} / {note_type}] {contents}")
            results_out.append(
                {
                    "id": d.get("id"),
                    "metadata": meta,
                    "contents": contents,
                }
            )

        context_text = "\n\n".join(contexts)

        answer = (
            "Here is what I found in the encrypted patient notes:\n\n"
            f"{context_text}\n\n"
            "→ The last recommended treatment is described in the most relevant "
            "note(s) shown above for this patient."
        )

        return {
            "answer": answer,
            "results": results_out,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# --------------- RAG answer endpoint -------------------------

@app.post("/rag-answer")
def rag_answer(req: QueryRequest):
    """
    Higher-level endpoint:

    - Uses encrypted vector search (CyborgDB)
    - Then calls Gemini Pro to generate a doctor-style answer
    - If Gemini is missing or fails, falls back to extracting
      'Last recommended treatment:' directly from the latest note.
    """
    try:
        # If no Gemini key, just behave like /query with a note.
        if not GEMINI_API_KEY:
            base = query_notes(req)
            return {
                "question": req.question,
                "answer": (
                    "Gemini is disabled because GEMINI_API_KEY is not set. "
                    "Returning basic summary from encrypted notes instead.\n\n"
                    + base["answer"]
                ),
                "supporting_notes": [
                    r.get("contents", "") for r in base.get("results", [])
                ],
                "used_model": None,
            }

        # 1. Embed question
        query_vec = embed_text(req.question)

        # 2. Optional patient filter
        filters = {"patient_id": {"$eq": req.patient_id}} if req.patient_id else None

        # 3. Encrypted search in CyborgDB
        raw_results = index.query(
            query_vectors=[query_vec],
            top_k=req.top_k,
            n_probes=5,
            filters=filters,
            include=["distance", "metadata"],
        )

        if not raw_results:
            return {
                "question": req.question,
                "answer": "No matching notes found for this query.",
                "supporting_notes": [],
                "used_model": GEMINI_MODEL,
            }

        if isinstance(raw_results[0], list):
            hits = raw_results[0]
        else:
            hits = raw_results

        if not hits:
            return {
                "question": req.question,
                "answer": "No matching notes found for this query.",
                "supporting_notes": [],
                "used_model": GEMINI_MODEL,
            }

        ids = [h.get("id") for h in hits if "id" in h]
        if not ids:
            return {
                "question": req.question,
                "answer": "No matching notes found for this query.",
                "supporting_notes": [],
                "used_model": GEMINI_MODEL,
            }

        docs = index.get(ids=ids, include=["contents", "metadata"])

        if isinstance(docs, dict):
            doc_list = [docs]
        else:
            doc_list = docs

        notes_for_llm: List[str] = []
        for d in doc_list:
            meta = d.get("metadata", {}) or {}
            contents = d.get("contents", "") or ""
            pid = meta.get("patient_id", "Unknown")
            note_type = meta.get("note_type", "note")
            notes_for_llm.append(f"[Patient: {pid}, Type: {note_type}]\n{contents}")

        context_text = "\n\n---\n\n".join(notes_for_llm)

        system_prompt = (
            "You are an AI assistant helping doctors review encrypted patient notes. "
            "You will be given clinical notes as context. Answer clearly and briefly, "
            "only using the information in the notes. Do NOT invent treatments."
        )

        user_prompt = (
            f"Doctor's question: {req.question}\n\n"
            f"Relevant encrypted patient notes (already decrypted inside the app):\n\n"
            f"{context_text}\n\n"
            "Based only on these notes, answer the doctor's question. "
            "If the information is not present, say that it is not specified clearly."
        )

        # 4. Try Gemini Pro
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content([system_prompt, user_prompt])
            llm_answer = response.text if hasattr(response, "text") else str(response)

        except Exception:
            # 5. Nicer fallback: extract 'Last recommended treatment:' directly
            primary_note = notes_for_llm[0] if notes_for_llm else ""
            treatment_answer: Optional[str] = None

            marker = "Last recommended treatment:"
            if marker in primary_note:
                after = primary_note.split(marker, 1)[1].strip()
                # Stop at first newline or first period
                segment = after
                for sep in ["\n", "."]:
                    if sep in segment:
                        segment = segment.split(sep, 1)[0]
                        break
                treatment_answer = segment.strip(" .;:,")

            if treatment_answer:
                llm_answer = (
                    "Based on the latest encrypted note for this patient, "
                    f"the last recommended treatment was: {treatment_answer}."
                )
            else:
                llm_answer = (
                    "The encrypted search succeeded and returned the latest note, "
                    "but the external LLM is currently unavailable. "
                    "Please refer to the retrieved note text for full details:\n\n"
                    f"{context_text}"
                )

        return {
            "question": req.question,
            "answer": llm_answer,
            "supporting_notes": notes_for_llm,
            "used_model": GEMINI_MODEL,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG answer failed: {e}")
