"""
Tests for the retrieval module.

These tests use local FAISS and local embeddings.
No external LLM API is required.
"""

import pytest

from langchain_core.documents import Document

from src.embeddings.embedding_model import (
    get_embedding_model,
)

from src.retrieval.retriever import (
    create_retriever,
    retrieve_documents,
    retrieve_documents_with_scores,
    format_retrieved_context,
    extract_sources,
)

from src.retrieval.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    delete_vector_store,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for retrieval tests."""

    return [
        Document(
            page_content=(
                "Customers can return products "
                "within 30 days of delivery."
            ),
            metadata={
                "source": "return_policy.txt",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "Refunds are processed within "
                "5 to 7 business days."
            ),
            metadata={
                "source": "refund_policy.txt",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "Customers can pay using UPI, "
                "credit cards, and debit cards."
            ),
            metadata={
                "source": "payment_policy.txt",
                "page": 1,
            },
        ),
    ]


@pytest.fixture
def vector_store(
    sample_documents: list[Document],
):
    """Create a temporary FAISS vector store."""

    embedding_model = get_embedding_model()

    return create_vector_store(
        documents=sample_documents,
        embedding_model=embedding_model,
    )


@pytest.fixture
def retriever(vector_store):
    """Create a retriever."""

    return create_retriever(
        vector_store=vector_store,
        k=2,
    )


# ============================================================
# CREATE RETRIEVER TESTS
# ============================================================


def test_create_retriever(
    vector_store,
) -> None:
    """Test retriever creation."""

    retriever = create_retriever(
        vector_store=vector_store,
        k=2,
    )

    assert retriever is not None


def test_create_retriever_default_k(
    vector_store,
) -> None:
    """Test retriever creation with default k."""

    retriever = create_retriever(
        vector_store=vector_store,
    )

    assert retriever is not None


def test_create_retriever_invalid_k(
    vector_store,
) -> None:
    """Invalid k should raise ValueError."""

    with pytest.raises(ValueError):
        create_retriever(
            vector_store=vector_store,
            k=0,
        )


def test_create_retriever_negative_k(
    vector_store,
) -> None:
    """Negative k should raise ValueError."""

    with pytest.raises(ValueError):
        create_retriever(
            vector_store=vector_store,
            k=-1,
        )


def test_create_retriever_none() -> None:
    """None vector store should raise ValueError."""

    with pytest.raises(ValueError):
        create_retriever(
            vector_store=None,
        )


# ============================================================
# DOCUMENT RETRIEVAL TESTS
# ============================================================


def test_retrieve_documents(
    retriever,
) -> None:
    """Test document retrieval."""

    documents = retrieve_documents(
        retriever=retriever,
        query="What is the return policy?",
    )

    assert isinstance(
        documents,
        list,
    )

    assert len(documents) > 0

    assert all(
        isinstance(document, Document)
        for document in documents
    )


def test_retrieve_relevant_document(
    retriever,
) -> None:
    """Test that relevant documents are retrieved."""

    documents = retrieve_documents(
        retriever=retriever,
        query="How many days do I have to return a product?",
    )

    assert len(documents) > 0

    combined_text = " ".join(
        document.page_content
        for document in documents
    )

    assert "30 days" in combined_text


def test_retrieve_empty_query(
    retriever,
) -> None:
    """Empty query should raise ValueError."""

    with pytest.raises(ValueError):
        retrieve_documents(
            retriever=retriever,
            query="",
        )


def test_retrieve_whitespace_query(
    retriever,
) -> None:
    """Whitespace-only query should raise ValueError."""

    with pytest.raises(ValueError):
        retrieve_documents(
            retriever=retriever,
            query="   ",
        )


def test_retrieve_none_retriever() -> None:
    """None retriever should raise ValueError."""

    with pytest.raises(ValueError):
        retrieve_documents(
            retriever=None,
            query="test query",
        )


# ============================================================
# RETRIEVAL WITH SCORES
# ============================================================


def test_retrieve_documents_with_scores(
    vector_store,
) -> None:
    """Test similarity search with scores."""

    results = retrieve_documents_with_scores(
        vector_store=vector_store,
        query="return policy",
        k=2,
    )

    assert isinstance(
        results,
        list,
    )

    assert len(results) > 0

    for document, score in results:
        assert isinstance(
            document,
            Document,
        )

        assert isinstance(
            score,
            float,
        )


def test_retrieve_documents_with_scores_empty_query(
    vector_store,
) -> None:
    """Empty query should raise ValueError."""

    with pytest.raises(ValueError):
        retrieve_documents_with_scores(
            vector_store=vector_store,
            query="",
        )


def test_retrieve_documents_with_scores_none_store() -> None:
    """None vector store should raise ValueError."""

    with pytest.raises(ValueError):
        retrieve_documents_with_scores(
            vector_store=None,
            query="return policy",
        )


# ============================================================
# CONTEXT FORMATTER TESTS
# ============================================================


def test_format_retrieved_context(
    sample_documents: list[Document],
) -> None:
    """Test context formatting."""

    context = format_retrieved_context(
        sample_documents
    )

    assert isinstance(
        context,
        str,
    )

    assert "return_policy.txt" in context
    assert "refund_policy.txt" in context
    assert "30 days" in context


def test_format_retrieved_context_empty() -> None:
    """Empty document list should return empty string."""

    context = format_retrieved_context([])

    assert context == ""


def test_format_context_contains_page(
    sample_documents: list[Document],
) -> None:
    """Context should contain page information."""

    context = format_retrieved_context(
        sample_documents
    )

    assert "page 1" in context


# ============================================================
# SOURCE EXTRACTION TESTS
# ============================================================


def test_extract_sources(
    sample_documents: list[Document],
) -> None:
    """Test source extraction."""

    sources = extract_sources(
        sample_documents
    )

    assert isinstance(
        sources,
        list,
    )

    assert "return_policy.txt" in sources
    assert "refund_policy.txt" in sources
    assert "payment_policy.txt" in sources


def test_extract_sources_unique() -> None:
    """Duplicate sources should be removed."""

    documents = [
        Document(
            page_content="First chunk.",
            metadata={
                "source": "policy.txt"
            },
        ),
        Document(
            page_content="Second chunk.",
            metadata={
                "source": "policy.txt"
            },
        ),
    ]

    sources = extract_sources(
        documents
    )

    assert sources == [
        "policy.txt"
    ]


def test_extract_sources_empty() -> None:
    """Empty document list should return empty list."""

    sources = extract_sources([])

    assert sources == []


# ============================================================
# VECTOR STORE TESTS
# ============================================================


def test_create_vector_store(
    sample_documents: list[Document],
) -> None:
    """Test FAISS vector store creation."""

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        documents=sample_documents,
        embedding_model=embedding_model,
    )

    assert vector_store is not None

    assert vector_store.index.ntotal == len(
        sample_documents
    )


def test_create_vector_store_empty() -> None:
    """Empty documents should raise ValueError."""

    embedding_model = get_embedding_model()

    with pytest.raises(ValueError):
        create_vector_store(
            documents=[],
            embedding_model=embedding_model,
        )


# ============================================================
# VECTOR STORE SAVE / LOAD TESTS
# ============================================================


def test_vector_store_save_and_load(
    vector_store,
    tmp_path,
) -> None:
    """Test saving and loading a vector store."""

    save_path = tmp_path / "vectorstore"

    save_vector_store(
        vector_store=vector_store,
        path=save_path,
    )

    assert (
        save_path / "index.faiss"
    ).exists()

    assert (
        save_path / "index.pkl"
    ).exists()

    loaded_store = load_vector_store(
        path=save_path,
    )

    assert loaded_store is not None

    assert (
        loaded_store.index.ntotal
        == vector_store.index.ntotal
    )


def test_vector_store_exists(
    vector_store,
    tmp_path,
) -> None:
    """Test vector store existence check."""

    save_path = tmp_path / "vectorstore"

    assert (
        vector_store_exists(save_path)
        is False
    )

    save_vector_store(
        vector_store=vector_store,
        path=save_path,
    )

    assert (
        vector_store_exists(save_path)
        is True
    )


def test_delete_vector_store(
    vector_store,
    tmp_path,
) -> None:
    """Test vector store deletion."""

    save_path = tmp_path / "vectorstore"

    save_vector_store(
        vector_store=vector_store,
        path=save_path,
    )

    assert (
        vector_store_exists(save_path)
        is True
    )

    delete_vector_store(
        path=save_path,
    )

    assert (
        vector_store_exists(save_path)
        is False
    )