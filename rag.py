
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from search import search_top_k
from generate import generate_answer
from chat_memory import (
    create_session,
    save_message,
    get_history,
    list_sessions,
)


# ============================================================
# 1. Fonction RAG avec mémoire conversationnelle
# ============================================================

def rag(
    question: str,
    history: list,
    k: int = 3
) -> str:

    # 1. Recherche documentaire dans Milvus
    chunks = search_top_k(
        query=question,
        k=k
    )

    # 2. Génération avec contexte documentaire et historique
    answer = generate_answer(
        question=question,
        chunks=chunks,
        history=history
    )

    return answer


# ============================================================
# 2. Création de l'API
# ============================================================

app = FastAPI(
    title="RAG API",
    description="API RAG avec mémoire conversationnelle SQLite",
    version="1.1.0"
)


# ============================================================
# 3. Modèles des requêtes
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    session_id: str
    k: int = 3


# ============================================================
# 4. Créer une nouvelle session
# ============================================================

@app.post("/sessions")
def new_session():

    session_id = str(uuid4())

    create_session(session_id)

    return {
        "session_id": session_id,
        "message": "Session créée"
    }


@app.get("/sessions")
def get_sessions():
    sessions = list_sessions()
    return {"sessions": sessions}

# ============================================================
# 5. Récupérer l'historique d'une session
# ============================================================

@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):

    history = get_history(session_id)

    return {
        "session_id": session_id,
        "messages": history
    }


# ============================================================
# 6. Poser une question
# ============================================================

@app.post("/ask")
def ask(request: QuestionRequest):

    # Vérifier que la session existe
    history = get_history(request.session_id)

    # Enregistrer la question utilisateur
    save_message(
        session_id=request.session_id,
        role="user",
        content=request.question
    )

    try:
        # Générer la réponse en utilisant l'historique précédent
        answer = rag(
            question=request.question,
            history=history,
            k=request.k
        )

        # Enregistrer la réponse de l'assistant
        save_message(
            session_id=request.session_id,
            role="assistant",
            content=answer
        )

        return {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la génération RAG : {str(error)}"
        )
