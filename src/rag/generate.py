import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant spécialisé dans Docker.

Réponds à la question actuelle en utilisant uniquement
les informations présentes dans le contexte documentaire.

L'historique sert à comprendre les références aux questions
et réponses précédentes. Il ne constitue pas une source
documentaire supplémentaire.

Si la réponse ne se trouve pas dans le contexte, indique
que l'information n'est pas présente dans les documents.

Ne fais pas appel à des connaissances externes.""",
        ),
        (
            "human",
            """### HISTORIQUE DE LA CONVERSATION

{conversation}

### CONTEXTE DOCUMENTAIRE

{context}

### QUESTION ACTUELLE

{question}

### RÉPONSE""",
        ),
    ]
)

# Chaîne LangChain : prompt -> modèle Ollama -> texte final
generation_chain = prompt | llm | StrOutputParser()


def build_prompt(
    question: str,
    chunks: list,
    history: list | None = None,
) -> str:
    """Construit un prompt lisible pour le diagnostic."""

    context = "\n\n".join(
        f"""[Chunk {i + 1}]
Section: {chunk["entity"]["section"]}

{chunk["entity"]["content"]}"""
        for i, chunk in enumerate(chunks)
    )

    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in (history or [])
    )

    formatted = prompt.format(
        question=question,
        context=context or "Aucun document pertinent récupéré.",
        conversation=conversation or "Aucun échange précédent.",
    )

    return formatted


def generate_answer(
    question: str,
    chunks: list,
    history: list | None = None,
) -> str:
    """Génère une réponse à partir des documents récupérés."""

    context = "\n\n".join(
        f"""[Chunk {i + 1}]
Section: {chunk["entity"]["section"]}

{chunk["entity"]["content"]}"""
        for i, chunk in enumerate(chunks)
    )

    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in (history or [])
    )

    return generation_chain.invoke(
        {
            "question": question,
            "context": context or "Aucun document pertinent récupéré.",
            "conversation": conversation or "Aucun échange précédent.",
        }
    )
