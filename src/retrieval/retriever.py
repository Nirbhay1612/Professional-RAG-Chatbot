"""
Document retrieval utilities for the Professional RAG Chatbot.

This module provides:
    - Retriever creation
    - Similarity-based document retrieval
    - Retrieval with similarity scores
    - Context formatting
    - Source extraction
"""

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.settings import RETRIEVAL_K


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

DEFAULT_SEARCH_TYPE = "similarity"


# =========================================================
# VALIDATION HELPERS
# =========================================================

def _validate_k(k: int) -> int:
    """
    Validate retrieval count.

    Args:
        k: Number of documents to retrieve.

    Returns:
        Validated retrieval count.

    Raises:
        ValueError: If k is not positive.
    """
    if k <= 0:
        raise ValueError(
            "Retrieval k must be greater than 0."
        )

    return k


def _validate_query(query: str) -> str:
    """
    Validate and normalize a user query.

    Args:
        query: User search query.

    Returns:
        Stripped query.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty.
    """
    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "Query cannot be empty."
        )

    return cleaned_query


# =========================================================
# CREATE RETRIEVER
# =========================================================

def create_retriever(
    vector_store: FAISS | None,
    k: int | None = None,
) -> BaseRetriever:
    """
    Create a LangChain retriever from a FAISS vector store.

    Args:
        vector_store:
            FAISS vector store.

        k:
            Number of documents to retrieve.
            Defaults to RETRIEVAL_K.

    Returns:
        Configured LangChain BaseRetriever.

    Raises:
        ValueError:
            If vector_store is None or k is invalid.
    """
    if vector_store is None:
        raise ValueError(
            "Vector store cannot be None."
        )

    retrieval_k = (
        RETRIEVAL_K
        if k is None
        else k
    )

    retrieval_k = _validate_k(
        retrieval_k
    )

    return vector_store.as_retriever(
        search_type=DEFAULT_SEARCH_TYPE,
        search_kwargs={
            "k": retrieval_k,
        },
    )


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(
    retriever: BaseRetriever,
    query: str,
) -> list[Document]:
    """
    Retrieve relevant documents for a query.

    Args:
        retriever:
            Configured LangChain retriever.

        query:
            User search query.

    Returns:
        List of relevant documents.

    Raises:
        ValueError:
            If retriever is None or query is empty.
        TypeError:
            If query is not a string.
    """
    if retriever is None:
        raise ValueError(
            "Retriever cannot be None."
        )

    cleaned_query = _validate_query(query)

    documents = retriever.invoke(
        cleaned_query
    )

    if not documents:
        return []

    return [
        document
        for document in documents
        if isinstance(document, Document)
    ]


# =========================================================
# RETRIEVE WITH SIMILARITY SCORES
# =========================================================

def retrieve_documents_with_scores(
    vector_store: FAISS | None,
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents along with FAISS similarity scores.

    Args:
        vector_store:
            FAISS vector store.

        query:
            User search query.

        k:
            Number of documents to retrieve.

    Returns:
        List of (Document, score) tuples.

    Raises:
        ValueError:
            If inputs are invalid.
        TypeError:
            If query is not a string.
    """
    if vector_store is None:
        raise ValueError(
            "Vector store cannot be None."
        )

    cleaned_query = _validate_query(query)

    retrieval_k = (
        RETRIEVAL_K
        if k is None
        else k
    )

    retrieval_k = _validate_k(
        retrieval_k
    )

    results = vector_store.similarity_search_with_score(
        cleaned_query,
        k=retrieval_k,
    )

    return [
        (document, float(score))
        for document, score in results
        if isinstance(document, Document)
    ]


# =========================================================
# FORMAT RETRIEVED CONTEXT
# =========================================================

def format_retrieved_context(
    documents: list[Document],
) -> str:
    """
    Convert retrieved documents into structured RAG context.

    Each source is clearly separated so the LLM can
    distinguish between retrieved documents.

    Args:
        documents:
            Retrieved LangChain documents.

    Returns:
        Formatted context string.
    """
    if not documents:
        return ""

    formatted_context: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        if not isinstance(document, Document):
            continue

        metadata = document.metadata or {}

        source = str(
            metadata.get(
                "source",
                "Unknown source",
            )
        )

        page = metadata.get("page")

        if page is not None:
            location = (
                f"{source}, page {page}"
            )
        else:
            location = source

        content = document.page_content.strip()

        if not content:
            continue

        formatted_context.append(
            f"[Source {index}: {location}]\n"
            f"{content}"
        )

    return "\n\n".join(
        formatted_context
    )


# =========================================================
# EXTRACT UNIQUE SOURCES
# =========================================================

def extract_sources(
    documents: list[Document],
) -> list[str]:
    """
    Extract unique source names from retrieved documents.

    Source order is preserved.

    Args:
        documents:
            Retrieved LangChain documents.

    Returns:
        List of unique source names.
    """
    sources: list[str] = []

    for document in documents:
        if not isinstance(document, Document):
            continue

        metadata = document.metadata or {}

        source = metadata.get("source")

        if not source:
            continue

        source_string = str(source)

        if source_string not in sources:
            sources.append(source_string)

    return sources