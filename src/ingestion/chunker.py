# src/ingestion/chunker.py

from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE


# =========================================================
# TEXT SPLITTER CONFIGURATION
# =========================================================

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the configured text splitter for the RAG pipeline.

    Returns:
        RecursiveCharacterTextSplitter:
            Configured text splitter instance.

    Raises:
        ValueError:
            If chunk configuration is invalid.
    """
    if CHUNK_SIZE <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than 0."
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        strip_whitespace=True,
        length_function=len,
    )


# =========================================================
# CHUNK SINGLE DOCUMENT
# =========================================================

def chunk_document(
    document: Document,
) -> List[Document]:
    """
    Split a single document into smaller RAG-friendly chunks.

    Args:
        document: LangChain Document object.

    Returns:
        List of document chunks.

    Raises:
        TypeError:
            If the input is not a LangChain Document.
    """
    if not isinstance(document, Document):
        raise TypeError(
            "Expected a LangChain Document object."
        )

    if not document.page_content.strip():
        return []

    splitter = get_text_splitter()

    chunks = splitter.split_documents(
        [document]
    )

    return _add_chunk_metadata(chunks)


# =========================================================
# CHUNK DOCUMENT COLLECTION
# =========================================================

def chunk_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split multiple documents into RAG-friendly chunks.

    Original document metadata is preserved.

    Additional metadata:
        - chunk_id
        - chunk_size

    Args:
        documents: List of LangChain Documents.

    Returns:
        List of non-empty document chunks.
    """
    if not documents:
        return []

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError(
                "All items must be LangChain Document objects."
            )

    # Remove empty documents before splitting.
    valid_documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not valid_documents:
        return []

    splitter = get_text_splitter()

    chunks = splitter.split_documents(
        valid_documents
    )

    # Remove empty chunks.
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    return _add_chunk_metadata(chunks)


# =========================================================
# CHUNK METADATA
# =========================================================

def _add_chunk_metadata(
    chunks: List[Document],
) -> List[Document]:
    """
    Add standardized metadata to document chunks.

    Args:
        chunks: List of document chunks.

    Returns:
        Chunks with metadata.
    """
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_size"] = len(
            chunk.page_content
        )

    return chunks


# =========================================================
# CHUNKING STATISTICS
# =========================================================

def get_chunk_statistics(
    chunks: List[Document],
) -> Dict[str, float]:
    """
    Generate statistics for document chunks.

    Args:
        chunks: List of document chunks.

    Returns:
        Dictionary containing:
            - total_chunks
            - average_chunk_size
            - min_chunk_size
            - max_chunk_size
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "average_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
        }

    sizes = [
        len(chunk.page_content)
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not sizes:
        return {
            "total_chunks": 0,
            "average_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
        }

    return {
        "total_chunks": len(sizes),
        "average_chunk_size": round(
            sum(sizes) / len(sizes),
            2,
        ),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
    }