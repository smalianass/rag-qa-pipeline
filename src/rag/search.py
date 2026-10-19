
from .paths import DATA_DIR

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DB_PATH = str(DATA_DIR / "milvus.db")
COLLECTION_NAME = "docker_documents"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

CANDIDATE_K = 4  # Nombre de chunks récupérés par Milvus
TOP_K = 2        # Nombre de chunks conservés après reranking


# --------------------------------------------------
# 2. Connexion à Milvus
# --------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)
client = MilvusClient(DB_PATH)

client.load_collection(
    collection_name=COLLECTION_NAME
)

print(f"Collection '{COLLECTION_NAME}' chargée.")


# --------------------------------------------------
# 3. Chargement des modèles
# --------------------------------------------------

print("Chargement du modèle d'embedding...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Chargement du modèle de reranking...")

reranker = FlagReranker(
    RERANKER_MODEL_NAME,
    use_fp16=False
)

print("Modèles chargés avec succès.")


# --------------------------------------------------
# 4. Recherche vectorielle + reranking
# --------------------------------------------------

def search_top_k(
    query: str,
    k: int = TOP_K
):
    """
    Recherche les chunks dans Milvus, puis les reclasse
    avec le reranker.

    Args:
        query: question de l'utilisateur.
        k: nombre final de chunks à retourner.

    Returns:
        Liste des chunks les plus pertinents après reranking.
    """

    if not query or not query.strip():
        return []

    if k <= 0:
        return []

    # ----------------------------------------------
    # Étape 1 : embedding de la question
    # ----------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    # ----------------------------------------------
    # Étape 2 : récupération des candidats par Milvus
    # ----------------------------------------------

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding.tolist()],
        limit=CANDIDATE_K,
        output_fields=[
            "source",
            "section",
            "content"
        ],
        search_params={
            "metric_type": "COSINE"
        }
    )

    candidates = results[0]

    if not candidates:
        return []

    # ----------------------------------------------
    # Étape 3 : préparation des paires question-chunk
    # ----------------------------------------------

    pairs = [
        [
            query,
            candidate["entity"]["content"]
        ]
        for candidate in candidates
    ]

    # ----------------------------------------------
    # Étape 4 : calcul des scores de reranking
    # ----------------------------------------------

    scores = reranker.compute_score(
        pairs,
        normalize=False
    )

    # compute_score renvoie un scalaire pour une seule paire
    # et une liste pour plusieurs paires.
    if isinstance(scores, (int, float)):
        scores = [scores]

    # ----------------------------------------------
    # Étape 5 : ajouter les scores et trier
    # ----------------------------------------------

    reranked_results = []

    for candidate, score in zip(candidates, scores):
        entity = candidate["entity"]

        reranked_results.append({
            "id": candidate["id"],
            "distance": candidate.get("distance"),
            "rerank_score": float(score),
            "entity": entity
        })

    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # ----------------------------------------------
    # Étape 6 : retourner les k meilleurs chunks
    # ----------------------------------------------

    final_results = reranked_results[:k]

    print(
        f"Recherche terminée : {len(candidates)} candidats, "
        f"{len(final_results)} chunks après reranking."
    )

    return final_results
