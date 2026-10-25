import json
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
output_path = DATA_DIR / "chunks.json"

sources = [
    (DATA_DIR / "docker_cheatsheet.md", "docker_cheatsheet.pdf"),
    (DATA_DIR / "github_actions.md", "github_actions.md"),
]


# ============================================================
# 1. LECTURE DU MARKDOWN
# ============================================================

documents = []

for markdown_path, source_name in sources:
    with markdown_path.open("r", encoding="utf-8") as f:
        markdown = f.read()

    # ============================================================
    # 2. EXTRACTION DES SECTIONS
    # ============================================================

    sections = re.split(r"(?m)^## ", markdown)

    for section in sections:
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()

        title = lines[0].strip()

        content = "\n".join(lines[1:]).strip()

        # Supprimer les artefacts d'image Docling
        content = content.replace("<!-- image -->", "").strip()

        if not content:
            continue

        documents.append({"source": source_name, "section": title, "content": content})


# ============================================================
# 3. DÉTECTION DES COMMANDES
# ============================================================
def is_command(text):
    """
    Détermine si un bloc correspond à une commande Docker.
    """

    text = text.strip()

    return text.startswith("docker ")


# ============================================================
# 4. CRÉATION DES CHUNKS
# ============================================================

chunks = []

chunk_id = 0


for document in documents:
    source_name = document["source"]
    section = document["section"]
    content = document["content"]

    # --------------------------------------------------------
    # Séparer les blocs
    # --------------------------------------------------------

    blocks = re.split(r"\n\s*\n", content)

    blocks = [block.strip() for block in blocks if block.strip()]

    i = 0

    while i < len(blocks):
        current = blocks[i]

        # ----------------------------------------------------
        # DESCRIPTION + COMMANDE
        # ----------------------------------------------------

        if i + 1 < len(blocks) and not is_command(current) and is_command(blocks[i + 1]):
            combined_content = current + "\n\n" + blocks[i + 1]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "section": section,
                    "content": combined_content,
                }
            )

            chunk_id += 1

            i += 2

        # ----------------------------------------------------
        # BLOC SEUL
        # ----------------------------------------------------

        else:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "section": section,
                    "content": current,
                }
            )

            chunk_id += 1

            i += 1


# ============================================================
# 5. SAUVEGARDE JSON
# ============================================================

with output_path.open("w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)


# ============================================================
# 6. AFFICHAGE DU RÉSULTAT
# ============================================================

print("=" * 70)
print("CHUNKING TERMINÉ")
print("=" * 70)

print(f"Nombre de sections : {len(documents)}")
print(f"Nombre de chunks   : {len(chunks)}")
print(f"Fichier créé       : {output_path}")

print()


for chunk in chunks:
    print("=" * 70)

    print(f"CHUNK ID : {chunk['chunk_id']}")
    print(f"SOURCE   : {chunk['source']}")
    print(f"SECTION  : {chunk['section']}")

    print("-" * 70)

    print(chunk["content"])

    print()
