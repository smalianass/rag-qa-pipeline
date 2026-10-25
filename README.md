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

L'image Docker de l'API utilise `requirements-api.txt` pour éviter d'installer
les dépendances du pipeline et des outils de test dans le conteneur. Comme le
service fonctionne sur CPU, elle installe également les roues CPU de PyTorch.
L'interface utilise uniquement `requirements-frontend.txt`. Les Dockerfiles
gardent le cache pip entre les builds et isolent l'installation de PyTorch :
une modification des dépendances API ne force donc pas son téléchargement à
nouveau.

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

Pour construire les images sans démarrer les services, en parallèle :

```powershell
docker compose --parallel 2 -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up
```

Le nom du projet Compose est fixé à `rag` afin que l'exécution depuis `deploy/`
réutilise le même projet :

```powershell
docker compose up -d --build
```

Les documents, la base Milvus et l'historique SQLite sont conservés localement
dans `data/`.

Compose transmet le fichier `.env` de la racine au conteneur API pour activer
les traces LangSmith. Vérifie qu'il contient `LANGSMITH_TRACING=true`,
`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` et, si nécessaire,
`LANGSMITH_ENDPOINT`. Après toute modification de ces variables, recrée l'API
avec `docker compose up -d --force-recreate rag-api`.

Par défaut, Ollama utilise `llama3.2:1b` pour réduire le temps de génération
sur une machine CPU. Pour choisir un autre modèle Ollama, définis
`OLLAMA_MODEL` avant de lancer Compose (par exemple
`$env:OLLAMA_MODEL = "llama3.2:3b"` dans PowerShell). La génération est aussi
limitée à 128 nouveaux tokens et à un contexte de 2048 tokens pour accélérer
les réponses ; le modèle 1B peut toutefois produire des réponses moins
détaillées que le modèle 3B.

## Tests

```powershell
python -m pytest
```
