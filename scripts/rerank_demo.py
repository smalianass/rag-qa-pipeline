
from FlagEmbedding import FlagReranker

# 1. Charger le modèle de reranking
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

reranker = FlagReranker(
    MODEL_NAME,
    use_fp16=False
)

# 2. Définir une question
query = "Quelle est la différence entre une image Docker et un conteneur Docker ?"

# 3. Simuler trois chunks candidats
chunks = [
    "Une image Docker est un modèle en lecture seule qui contient "
    "les éléments nécessaires à la création d'un conteneur.",

    "Un conteneur Docker est une instance exécutable d'une image Docker.",

    "Pour installer Docker sur Linux, il faut configurer le dépôt "
    "et installer le moteur Docker."
]

# 4. Construire les paires question-document
pairs = [[query, chunk] for chunk in chunks]

# 5. Calculer les scores de pertinence
scores = reranker.compute_score(
    pairs,
    normalize=False
)

# 6. Afficher les résultats classés
results = sorted(
    zip(chunks, scores),
    key=lambda item: item[1],
    reverse=True
)

for rank, (chunk, score) in enumerate(results, start=1):
    print(f"\nRang {rank} | Score : {score:.4f}")
    print(chunk)
