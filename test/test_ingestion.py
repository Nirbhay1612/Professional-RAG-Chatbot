# tests/test_ingestion.py

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.document_loader import load_document
from src.ingestion.text_cleaner import (
    clean_document,
    clean_documents,
    normalize_text,
)
from src.ingestion.chunker import (
    chunk_document,
    chunk_documents,
    get_chunk_statistics,
)


# =========================================================
# TEST DATA
# =========================================================

SAMPLE_TEXT = """
TechStore is an online technology company.

Customers can return eligible products within 30 days
of delivery.

Refunds are generally processed within 5-7 business days.
"""


# =========================================================
# DOCUMENT LOADER TESTS
# =========================================================

def test_load_txt_document(tmp_path: Path):
    """
    Test whether a TXT document can be loaded successfully.
    """

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        SAMPLE_TEXT,
        encoding="utf-8",
    )

    documents = load_document(file_path)

    assert documents
    assert len(documents) == 1

    assert (
        "TechStore"
        in documents[0].page_content
    )

    assert (
        documents[0].metadata["source"]
        == "sample.txt"
    )

    assert (
        documents[0].metadata["file_type"]
        == "txt"
    )


def test_load_unsupported_document(tmp_path: Path):
    """
    Test that unsupported file types raise ValueError.
    """

    file_path = tmp_path / "sample.xyz"

    file_path.write_text(
        "Unsupported document",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_document(file_path)


def test_load_missing_document(tmp_path: Path):
    """
    Test that a missing file raises FileNotFoundError.
    """

    file_path = (
        tmp_path / "does_not_exist.txt"
    )

    with pytest.raises(FileNotFoundError):
        load_document(file_path)


# =========================================================
# TEXT CLEANER TESTS
# =========================================================

def test_normalize_text():
    """
    Test whitespace and formatting normalization.
    """

    raw_text = (
        "Hello   world.\n\n\n"
        "This is   a test.\t\n"
    )

    cleaned = normalize_text(
        raw_text
    )

    assert cleaned == (
        "Hello world.\n\n"
        "This is a test."
    )


def test_normalize_text_removes_null_character():
    """
    Test removal of null characters.
    """

    text = "Hello\x00World"

    cleaned = normalize_text(
        text
    )

    assert "\x00" not in cleaned
    assert cleaned == "HelloWorld"


def test_clean_document():
    """
    Test cleaning of a single LangChain Document.
    """

    document = Document(
        page_content=(
            "Hello   world.\n\n\n"
            "This is a test."
        ),
        metadata={
            "source": "sample.txt"
        },
    )

    cleaned = clean_document(
        document
    )

    assert isinstance(
        cleaned,
        Document,
    )

    assert cleaned.page_content == (
        "Hello world.\n\n"
        "This is a test."
    )

    assert (
        cleaned.metadata["source"]
        == "sample.txt"
    )


def test_clean_documents_removes_empty_documents():
    """
    Test that empty documents are removed.
    """

    documents = [
        Document(
            page_content="Valid content",
        ),
        Document(
            page_content="   ",
        ),
        Document(
            page_content="Another valid document",
        ),
    ]

    cleaned_documents = clean_documents(
        documents
    )

    assert len(cleaned_documents) == 2

    assert all(
        document.page_content.strip()
        for document in cleaned_documents
    )


# =========================================================
# CHUNKER TESTS
# =========================================================

def test_chunk_document():
    """
    Test chunking of a single document.
    """

    document = Document(
        page_content=SAMPLE_TEXT,
        metadata={
            "source": "sample.txt"
        },
    )

    chunks = chunk_document(
        document
    )

    assert chunks
    assert len(chunks) >= 1

    assert all(
        isinstance(
            chunk,
            Document,
        )
        for chunk in chunks
    )


def test_chunk_documents():
    """
    Test chunking of multiple documents.
    """

    documents = [
        Document(
            page_content=SAMPLE_TEXT,
            metadata={
                "source": "document1.txt"
            },
        ),
        Document(
            page_content=(
                "TechStore accepts UPI, "
                "credit cards and debit cards."
            ),
            metadata={
                "source": "document2.txt"
            },
        ),
    ]

    chunks = chunk_documents(
        documents
    )

    assert chunks
    assert len(chunks) >= 2

    # Every chunk should have chunk metadata.
    for chunk in chunks:

        assert (
            "chunk_id"
            in chunk.metadata
        )

        assert (
            "chunk_size"
            in chunk.metadata
        )

        assert chunk.page_content.strip()


def test_chunk_metadata_is_preserved():
    """
    Test that original document metadata survives chunking.
    """

    document = Document(
        page_content=SAMPLE_TEXT,
        metadata={
            "source": "company_faq.txt",
            "file_type": "txt",
        },
    )

    chunks = chunk_documents(
        [document]
    )

    assert chunks

    for chunk in chunks:

        assert (
            chunk.metadata["source"]
            == "company_faq.txt"
        )

        assert (
            chunk.metadata["file_type"]
            == "txt"
        )


def test_chunk_statistics():
    """
    Test chunk statistics generation.
    """

    documents = [
        Document(
            page_content="A" * 100,
        ),
        Document(
            page_content="B" * 200,
        ),
        Document(
            page_content="C" * 300,
        ),
    ]

    chunks = chunk_documents(
        documents
    )

    statistics = get_chunk_statistics(
        chunks
    )

    assert (
        statistics["total_chunks"]
        > 0
    )

    assert (
        statistics["average_chunk_size"]
        > 0
    )

    assert (
        statistics["min_chunk_size"]
        > 0
    )

    assert (
        statistics["max_chunk_size"]
        >= statistics["min_chunk_size"]
    )


def test_empty_documents_return_empty_chunks():
    """
    Test that an empty document list produces no chunks.
    """

    chunks = chunk_documents([])

    assert chunks == []


def test_empty_document_is_not_chunked():
    """
    Test that an empty document produces no chunks.
    """

    document = Document(
        page_content="   "
    )

    chunks = chunk_document(
        document
    )

    assert chunks == []

