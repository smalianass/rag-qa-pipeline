# RAG QA Pipeline

Application de questions-réponses basée sur une recherche documentaire (RAG),
une API FastAPI, une interface Streamlit et une base vectorielle Milvus Lite.

## Structure

- `src/rag/` : API, recherche, génération, mémoire conversationnelle et accès Milvus.
- `frontend/` : interface Streamlit.
- `scripts/` : conversion PDF, découpage, embeddings et indexation.
- `tests/` : tests unitaires et d'intégration.
- `deploy/` : Dockerfiles et configuration Docker Compose.
- `.github/workflows/` : automatisation CI.
- `data/` : documents et bases locales, exclus de Git.

## Installation locale

Depuis la racine du dépôt, installe les dépendances puis le package en mode
éditable afin que les scripts puissent importer `rag` :

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-frontend.txt
python -m pip install -e .
```

Place le document source `docker_cheatsheet.pdf` dans `data/`, puis exécute le
pipeline depuis la racine :

```powershell
python scripts/pdftomd.py
python scripts/chunk.py
python scripts/embed.py
python scripts/insert.py
```

Lance l'API et l'interface dans deux terminaux :

```powershell
python -m uvicorn rag.api:app --reload
```

```powershell
streamlit run frontend/app.py
```

## Docker Compose

Depuis la racine du dépôt :

```powershell
docker compose -f deploy/compose.yaml up --build
```

Les documents, la base Milvus et l'historique SQLite sont conservés localement
dans `data/`.

## Tests

```powershell
python -m pytest
```
