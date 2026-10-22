import importlib
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def search_module(monkeypatch):
    """Importe rag.search avec Milvus et les modèles simulés."""

    fake_client = MagicMock()
    fake_embedding_model = MagicMock()
    fake_reranker = MagicMock()

    fake_pymilvus = MagicMock()
    fake_pymilvus.MilvusClient.return_value = fake_client

    fake_sentence_transformers = MagicMock()
    fake_sentence_transformers.SentenceTransformer.return_value = fake_embedding_model

    fake_flag_embedding = MagicMock()
    fake_flag_embedding.FlagReranker.return_value = fake_reranker

    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        fake_sentence_transformers,
    )
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding)

    # Force un nouvel import avec les dépendances simulées.
    sys.modules.pop("rag.search", None)
    module = importlib.import_module("rag.search")

    yield module, fake_client, fake_embedding_model, fake_reranker

    sys.modules.pop("rag.search", None)


def test_empty_query_returns_empty_list(search_module):
    module, _, _, _ = search_module

    assert module.search_top_k("") == []
    assert module.search_top_k("   ") == []


@pytest.mark.parametrize("k", [0, -1, -5])
def test_non_positive_k_returns_empty_list(search_module, k):
    module, _, _, _ = search_module

    assert module.search_top_k("Qu'est-ce que Docker ?", k=k) == []


def test_search_returns_empty_when_no_candidates(search_module):
    module, client, embedding_model, _ = search_module

    embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]
    client.search.return_value = [[]]

    result = module.search_top_k("Qu'est-ce que Docker ?")

    assert result == []


def test_reranking_sorts_by_score_and_limits_results(search_module):
    module, client, embedding_model, reranker = search_module

    embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2]

    client.search.return_value = [
        [
            {
                "id": 1,
                "distance": 0.9,
                "entity": {
                    "source": "docker.pdf",
                    "section": "Installation",
                    "content": "Installer Docker sur Linux.",
                },
            },
            {
                "id": 2,
                "distance": 0.8,
                "entity": {
                    "source": "docker.pdf",
                    "section": "Conteneurs",
                    "content": "Un conteneur est une instance exécutable d'une image.",
                },
            },
            {
                "id": 3,
                "distance": 0.7,
                "entity": {
                    "source": "docker.pdf",
                    "section": "Images",
                    "content": "Une image Docker est un modèle en lecture seule.",
                },
            },
        ]
    ]

    # Les scores sont volontairement dans un ordre différent
    # de celui de la recherche vectorielle.
    reranker.compute_score.return_value = [0.1, 0.9, 0.5]

    result = module.search_top_k("Qu'est-ce qu'un conteneur ?", k=2)

    assert len(result) == 2
    assert result[0]["id"] == 2
    assert result[1]["id"] == 3
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.5
    client.search.assert_called_once()
    reranker.compute_score.assert_called_once()
