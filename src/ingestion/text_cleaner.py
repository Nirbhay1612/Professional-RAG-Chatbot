# src/ingestion/text_cleaner.py

import re
import unicodedata
from typing import List

from langchain_core.documents import Document


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize and clean raw document text.

    Operations:
        - Unicode normalization
        - Line-ending normalization
        - Null-character removal
        - Tab replacement
        - Excessive-space cleanup
        - Blank-line cleanup
        - Trailing-space removal
        - Leading/trailing whitespace removal

    Args:
        text: Raw document text.

    Returns:
        Normalized text.
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    # Remove trailing whitespace from each line
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Remove leading/trailing whitespace
    return text.strip()


# =========================================================
# CLEAN SINGLE DOCUMENT
# =========================================================

def clean_document(document: Document) -> Document:
    """
    Clean the content of a single LangChain Document.

    Metadata is preserved.

    Args:
        document: LangChain Document object.

    Returns:
        A new cleaned Document object.

    Raises:
        TypeError:
            If the input is not a LangChain Document.
    """
    if not isinstance(document, Document):
        raise TypeError(
            "Expected a LangChain Document object."
        )

    cleaned_content = normalize_text(
        document.page_content
    )

    return Document(
        page_content=cleaned_content,
        metadata=document.metadata.copy(),
    )


# =========================================================
# CLEAN DOCUMENT COLLECTION
# =========================================================

def clean_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Clean a collection of LangChain Documents.

    Empty documents are automatically removed.

    Args:
        documents: List of LangChain Documents.

    Returns:
        Cleaned, non-empty Documents.
    """
    if not documents:
        return []

    cleaned_documents: List[Document] = []

    for document in documents:
        cleaned_document = clean_document(document)

        if cleaned_document.page_content.strip():
            cleaned_documents.append(cleaned_document)

    return cleaned_documents


# =========================================================
# CONTENT VALIDATION
# =========================================================

def has_meaningful_content(
    document: Document,
    minimum_length: int = 10,
) -> bool:
    """
    Check whether a document contains meaningful content.

    Args:
        document: LangChain Document.
        minimum_length: Minimum required character count.

    Returns:
        True if the document contains enough meaningful text,
        otherwise False.
    """
    if not isinstance(document, Document):
        return False

    if minimum_length < 0:
        raise ValueError(
            "minimum_length cannot be negative."
        )

    text = document.page_content.strip()

    return len(text) >= minimum_length