from FlagEmbedding import FlagReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langsmith import traceable

from .paths import DATA_DIR

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DB_PATH = str(DATA_DIR / "milvus.db")
COLLECTION_NAME = "docker_documents"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

CANDIDATE_K = 4
TOP_K = 2

# --------------------------------------------------
# 2. Chargement des embeddings
# --------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement du modèle d'embedding...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Chargement du modèle de reranking...")

reranker = FlagReranker(
    RERANKER_MODEL_NAME,
    use_fp16=False,
)

# --------------------------------------------------
# 3. Connexion à la collection Milvus existante
# --------------------------------------------------

vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    connection_args={"uri": DB_PATH},
    vector_field="embedding",
    primary_field="id",
    text_field="content",
    auto_id=False,
    drop_old=False,
)

print(f"Collection '{COLLECTION_NAME}' connectée via LangChain.")


# --------------------------------------------------
# 4. Recherche vectorielle + reranking
# --------------------------------------------------


@traceable(
    name="rag-search-and-rerank",
    run_type="retriever",
)
def search_top_k(query: str, k: int = TOP_K) -> list:
    """
    Recherche les chunks via LangChain/Milvus, puis les reclasse
    avec le reranker BGE.

    Le format des résultats reste compatible avec le pipeline existant.
    """

    if not query or not query.strip() or k <= 0:
        return []

    # Étape 1 : recherche vectorielle via LangChain
    documents_with_scores = vector_store.similarity_search_with_score(
        query,
        k=CANDIDATE_K,
    )

    if not documents_with_scores:
        return []

    # Étape 2 : reconstruire le format attendu par le reste du projet
    candidates = []

    for document, distance in documents_with_scores:
        candidates.append(
            {
                "id": document.metadata["id"],
                "distance": distance,
                "entity": {
                    "source": document.metadata.get("source", ""),
                    "section": document.metadata.get("section", ""),
                    "content": document.page_content,
                },
            }
        )

    # Étape 3 : préparer les paires question-document
    pairs = [[query, candidate["entity"]["content"]] for candidate in candidates]

    # Étape 4 : calculer les scores de reranking
    scores = reranker.compute_score(pairs, normalize=False)

    if isinstance(scores, (int, float)):
        scores = [scores]

    # Étape 5 : trier les résultats selon le reranker
    reranked_results = []

    for candidate, score in zip(candidates, scores):
        reranked_results.append(
            {
                "id": candidate["id"],
                "distance": candidate["distance"],
                "rerank_score": float(score),
                "entity": candidate["entity"],
            }
        )

    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    final_results = reranked_results[:k]

    print(
        f"Recherche terminée : {len(candidates)} candidats, "
        f"{len(final_results)} chunks après reranking."
    )

    return final_results
