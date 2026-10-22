import json

from rag.milvus import (
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_DIM,
    client,
    ensure_collection,
)


def main() -> None:
    input_path = DATA_DIR / "chunks_embeddings.json"
    with input_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(f"Le fichier {input_path} doit contenir une liste de chunks.")

    data = []
    for position, chunk in enumerate(chunks):
        try:
            embedding = chunk["embedding"]
            row = {
                "id": int(chunk["chunk_id"]),
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "embedding": embedding,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Chunk invalide à la position {position}: {error}") from error

        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding du chunk {row['id']} de dimension {len(embedding)}; "
                f"{EMBEDDING_DIM} valeurs attendues."
            )

        data.append(row)

    created = ensure_collection()
    print("Connexion à Milvus Lite réussie.")
    print(f"Collection créée : {'oui' if created else 'non, déjà existante'}")
    print(f"Nombre de chunks à insérer/mettre à jour : {len(data)}")

    result = client.upsert(
        collection_name=COLLECTION_NAME,
        data=data,
    )

    print()
    print("=" * 70)
    print("INSERTION TERMINÉE")
    print("=" * 70)
    print(f"Nombre de lignes insérées/mises à jour : {result['upsert_count']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
