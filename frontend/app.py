import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Assistant RAG",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Assistant RAG")
st.caption("Pose tes questions à partir de tes documents.")


# --------------------------------------------------
# 1. Fonctions pour communiquer avec FastAPI
# --------------------------------------------------


def create_new_session():
    response = requests.post(
        f"{API_URL}/sessions",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["session_id"]


def fetch_sessions():
    response = requests.get(
        f"{API_URL}/sessions",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["sessions"]


def fetch_history(session_id):
    response = requests.get(
        f"{API_URL}/sessions/{session_id}/messages",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["messages"]


# --------------------------------------------------
# 2. Initialiser l'état de l'application
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 3. Sidebar : sélectionner une conversation
# --------------------------------------------------

with st.sidebar:
    st.header("💬 Mes conversations")

    if st.button("➕ Nouvelle conversation", use_container_width=True):
        # On prépare une nouvelle conversation sans créer
        # de session dans la base de données.
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    try:
        sessions = fetch_sessions()

        # Ne pas afficher les sessions sans aucun message.
        for session in sessions:
            session_id = session["session_id"]
            first_question = session.get("first_question")

            if not first_question:
                continue

            title = first_question[:45] + "..." if len(first_question) > 45 else first_question

            if session_id == st.session_state.session_id:
                label = f"🟢 {title}"
            else:
                label = f"💬 {title}"

            if st.button(
                label,
                key=f"session_{session_id}",
                use_container_width=True,
            ):
                try:
                    messages = fetch_history(session_id)

                    st.session_state.session_id = session_id
                    st.session_state.messages = [
                        {
                            "role": message["role"],
                            "content": message["content"],
                        }
                        for message in messages
                    ]

                    st.rerun()

                except requests.RequestException as error:
                    st.error(f"Impossible de charger la conversation : {error}")

        if not any(s.get("first_question") for s in sessions):
            st.caption("Aucune conversation sauvegardée.")

    except requests.RequestException as error:
        st.error(f"Impossible de récupérer les conversations : {error}")


# --------------------------------------------------
# 4. Afficher l'historique de la conversation active
# --------------------------------------------------

if st.session_state.session_id is not None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("Pose une question pour démarrer une nouvelle conversation.")


# --------------------------------------------------
# 5. Envoyer une question à FastAPI
# --------------------------------------------------

question = st.chat_input("Pose une question...")

if question:
    try:
        # Créer une session uniquement à la première question.
        if st.session_state.session_id is None:
            st.session_state.session_id = create_new_session()

        with st.chat_message("user"):
            st.markdown(question)

        # Envoyer la question à l'API RAG.
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=180,
                )
                response.raise_for_status()

                data = response.json()
                answer = data["answer"]

                st.markdown(answer)

        # Mettre à jour l'historique local après une réponse réussie.
        st.session_state.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

        # Relancer l'affichage pour actualiser la sidebar
        # et faire apparaître la nouvelle conversation.
        st.rerun()

    except requests.RequestException as error:
        st.error(f"Erreur lors de l'appel à l'API RAG : {error}")

    except (KeyError, ValueError):
        st.error("La réponse de l'API n'a pas le format attendu.")
