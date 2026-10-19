import json
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

input_path = "data/chunks.json"
output_path = "data/chunks_embeddings.json"

model_name = "BAAI/bge-m3"


# ============================================================
# 1. CHARGER LES CHUNKS
# ============================================================

with open(input_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Nombre de chunks : {len(chunks)}")


# ============================================================
# 2. CHARGER BGE-M3
# ============================================================

print(f"Chargement du modèle : {model_name}")

model = SentenceTransformer(model_name)


# ============================================================
# 3. EXTRAIRE LE TEXTE DES CHUNKS
# ============================================================

texts = [
    chunk["content"]
    for chunk in chunks
]


# ============================================================
# 4. GÉNÉRER LES EMBEDDINGS
# ============================================================

print("Génération des embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# 5. AJOUTER L'EMBEDDING À CHAQUE CHUNK
# ============================================================

for chunk, embedding in zip(chunks, embeddings):

    chunk["embedding"] = embedding.tolist()


# ============================================================
# 6. SAUVEGARDER
# ============================================================

with open(output_path, "w", encoding="utf-8") as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 7. INFORMATIONS
# ============================================================

print()
print("=" * 70)
print("EMBEDDINGS TERMINÉS")
print("=" * 70)

print(f"Nombre de chunks : {len(chunks)}")
print(f"Dimension        : {embeddings.shape[1]}")
print(f"Fichier créé     : {output_path}")

print()
print("Premier chunk :")
print(chunks[0]["content"])

print()
print("10 premières valeurs de son embedding :")
print(chunks[0]["embedding"][:10])