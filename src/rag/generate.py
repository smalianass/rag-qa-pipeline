
import os
import ollama

MODEL_NAME = "llama3.2:3b"

ollama_client = ollama.Client(
    host=os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434"
    )
)


def build_prompt(
    question: str,
    chunks: list,
    history: list | None = None
) -> str:

    # 1. Construire le contexte documentaire récupéré depuis Milvus
    context = "\n\n".join(
        f"""[Chunk {i + 1}]
Section: {chunk["entity"]["section"]}

{chunk["entity"]["content"]}"""
        for i, chunk in enumerate(chunks)
    )

    # 2. Construire l'historique conversationnel
    conversation = ""

    if history:
        conversation = "\n".join(
            f'{message["role"]}: {message["content"]}'
            for message in history
        )

    # 3. Construire le prompt complet
    prompt = f"""
Tu es un assistant spécialisé dans Docker.

Réponds à la question actuelle en utilisant uniquement
les informations présentes dans le contexte documentaire.

L'historique sert à comprendre les références aux questions
et réponses précédentes. Il ne constitue pas une source
documentaire supplémentaire.

Si la réponse ne se trouve pas dans le contexte, indique
que l'information n'est pas présente dans les documents.

Ne fais pas appel à des connaissances externes.

### HISTORIQUE DE LA CONVERSATION

{conversation if conversation else "Aucun échange précédent."}

### CONTEXTE DOCUMENTAIRE

{context if context else "Aucun document pertinent récupéré."}

### QUESTION ACTUELLE

{question}

### RÉPONSE
"""

    return prompt


def generate_answer(
    question: str,
    chunks: list,
    history: list | None = None
) -> str:

    prompt = build_prompt(
        question=question,
        chunks=chunks,
        history=history
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
