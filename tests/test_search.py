import importlib
import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def search_module(monkeypatch):
    """Importe rag.search avec les composants LangChain simulés."""

    fake_embedding_model = MagicMock()
    fake_vector_store = MagicMock()
    fake_reranker = MagicMock()

    fake_huggingface = MagicMock()
    fake_huggingface.HuggingFaceEmbeddings.return_value = fake_embedding_model

    fake_langchain_milvus = MagicMock()
    fake_langchain_milvus.Milvus.return_value = fake_vector_store

    fake_flag_embedding = MagicMock()
    fake_flag_embedding.FlagReranker.return_value = fake_reranker

    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        fake_huggingface,
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_milvus",
        fake_langchain_milvus,
    )
    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        fake_flag_embedding,
    )

    sys.modules.pop("rag.search", None)
    module = importlib.import_module("rag.search")

    yield module, fake_vector_store, fake_embedding_model, fake_reranker

    sys.modules.pop("rag.search", None)


def test_empty_query_returns_empty_list(search_module):
    module, vector_store, _, _ = search_module

    assert module.COLLECTION_NAME == "devops_documents"
    assert module.search_top_k("") == []
    assert module.search_top_k("   ") == []

    vector_store.similarity_search_with_score.assert_not_called()


@pytest.mark.parametrize("k", [0, -1, -5])
def test_non_positive_k_returns_empty_list(search_module, k):
    module, vector_store, _, _ = search_module

    assert module.search_top_k("Qu'est-ce que Docker ?", k=k) == []

    vector_store.similarity_search_with_score.assert_not_called()


def test_search_returns_empty_when_no_candidates(search_module):
    module, vector_store, _, reranker = search_module

    vector_store.similarity_search_with_score.return_value = []

    result = module.search_top_k("Qu'est-ce que Docker ?")

    assert result == []
    reranker.compute_score.assert_not_called()


def test_reranking_sorts_by_score_and_limits_results(search_module):
    module, vector_store, _, reranker = search_module

    documents_with_scores = [
        (
            Document(
                page_content="Installer Docker sur Linux.",
                metadata={
                    "id": 1,
                    "source": "docker.pdf",
                    "section": "Installation",
                },
            ),
            0.9,
        ),
        (
            Document(
                page_content=("Un conteneur est une instance exécutable d'une image."),
                metadata={
                    "id": 2,
                    "source": "docker.pdf",
                    "section": "Conteneurs",
                },
            ),
            0.8,
        ),
        (
            Document(
                page_content=("Une image Docker est un modèle en lecture seule."),
                metadata={
                    "id": 3,
                    "source": "docker.pdf",
                    "section": "Images",
                },
            ),
            0.7,
        ),
    ]

    vector_store.similarity_search_with_score.return_value = documents_with_scores

    # Le reranker inverse volontairement l'ordre initial.
    reranker.compute_score.return_value = [0.1, 0.9, 0.5]

    result = module.search_top_k(
        "Qu'est-ce qu'un conteneur ?",
        k=2,
    )

    assert len(result) == 2
    assert result[0]["id"] == 2
    assert result[1]["id"] == 3
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.5

    vector_store.similarity_search_with_score.assert_called_once()
    reranker.compute_score.assert_called_once()
