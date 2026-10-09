
# src/utils/validators.py

from pathlib import Path
from typing import Iterable


# =========================================================
# FILE CONFIGURATION
# =========================================================

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}

# 20 MB maximum upload size
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)


# =========================================================
# FILE EXTENSION VALIDATION
# =========================================================

def is_supported_file(
    file_path: str | Path,
) -> bool:
    """
    Check whether a file has a supported extension.

    Args:
        file_path:
            Path or filename.

    Returns:
        bool:
            True if the extension is supported.
    """

    path = Path(file_path)

    return (
        path.suffix.lower()
        in SUPPORTED_FILE_EXTENSIONS
    )


# =========================================================
# FILE SIZE VALIDATION
# =========================================================

def is_valid_file_size(
    file_path: str | Path,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> bool:
    """
    Check whether a file is within the allowed size limit.

    Args:
        file_path:
            Path to the file.

        max_size_bytes:
            Maximum allowed size in bytes.

    Returns:
        bool:
            True if the file size is valid.
    """

    path = Path(file_path)

    if not path.exists():
        return False

    if not path.is_file():
        return False

    if max_size_bytes <= 0:
        return False

    return (
        path.stat().st_size
        <= max_size_bytes
    )


# =========================================================
# FILE VALIDATION
# =========================================================

def validate_file(
    file_path: str | Path,
) -> tuple[bool, str]:
    """
    Validate a document before processing.

    Checks:
        - File exists
        - File is actually a file
        - Extension is supported
        - File size is within limit

    Args:
        file_path:
            Path to the document.

    Returns:
        tuple:
            (is_valid, message)
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # Existence
    # -----------------------------------------------------

    if not path.exists():
        return (
            False,
            "File does not exist.",
        )

    # -----------------------------------------------------
    # File check
    # -----------------------------------------------------

    if not path.is_file():
        return (
            False,
            "Provided path is not a file.",
        )

    # -----------------------------------------------------
    # Extension
    # -----------------------------------------------------

    if not is_supported_file(path):

        supported = ", ".join(
            sorted(SUPPORTED_FILE_EXTENSIONS)
        )

        return (
            False,
            f"Unsupported file type. "
            f"Supported formats: {supported}",
        )

    # -----------------------------------------------------
    # File size
    # -----------------------------------------------------

    if not is_valid_file_size(path):

        return (
            False,
            f"File exceeds the maximum size "
            f"of {MAX_FILE_SIZE_MB} MB.",
        )

    return (
        True,
        "File is valid.",
    )


# =========================================================
# TEXT VALIDATION
# =========================================================

def validate_text(
    text: str,
    min_length: int = 1,
) -> tuple[bool, str]:
    """
    Validate text input.

    Args:
        text:
            Text to validate.

        min_length:
            Minimum required length.

    Returns:
        tuple:
            (is_valid, message)
    """

    if not isinstance(text, str):
        return (
            False,
            "Text must be a string.",
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        return (
            False,
            "Text cannot be empty.",
        )

    if len(cleaned_text) < min_length:
        return (
            False,
            f"Text must contain at least "
            f"{min_length} characters.",
        )

    return (
        True,
        "Text is valid.",
    )


# =========================================================
# QUESTION VALIDATION
# =========================================================

def validate_question(
    question: str,
) -> tuple[bool, str]:
    """
    Validate a user's RAG question.

    Args:
        question:
            User's question.

    Returns:
        tuple:
            (is_valid, message)
    """

    return validate_text(
        question,
        min_length=2,
    )


# =========================================================
# CHUNK CONFIGURATION VALIDATION
# =========================================================

def validate_chunk_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[bool, str]:
    """
    Validate chunking configuration.

    Args:
        chunk_size:
            Maximum chunk size.

        chunk_overlap:
            Overlap between chunks.

    Returns:
        tuple:
            (is_valid, message)
    """

    if not isinstance(
        chunk_size,
        int,
    ):
        return (
            False,
            "Chunk size must be an integer.",
        )

    if not isinstance(
        chunk_overlap,
        int,
    ):
        return (
            False,
            "Chunk overlap must be an integer.",
        )

    if chunk_size <= 0:
        return (
            False,
            "Chunk size must be greater than 0.",
        )

    if chunk_overlap < 0:
        return (
            False,
            "Chunk overlap cannot be negative.",
        )

    if chunk_overlap >= chunk_size:
        return (
            False,
            "Chunk overlap must be smaller "
            "than chunk size.",
        )

    return (
        True,
        "Chunk configuration is valid.",
    )


# =========================================================
# RETRIEVAL CONFIGURATION VALIDATION
# =========================================================

def validate_retrieval_k(
    k: int,
) -> tuple[bool, str]:
    """
    Validate the number of retrieved chunks.

    Args:
        k:
            Number of documents to retrieve.

    Returns:
        tuple:
            (is_valid, message)
    """

    if not isinstance(k, int):
        return (
            False,
            "Retrieval k must be an integer.",
        )

    if k <= 0:
        return (
            False,
            "Retrieval k must be greater than 0.",
        )

    return (
        True,
        "Retrieval configuration is valid.",
    )


# =========================================================
# DOCUMENT COLLECTION VALIDATION
# =========================================================

def validate_document_collection(
    documents: Iterable,
) -> tuple[bool, str]:
    """
    Validate a document collection.

    Args:
        documents:
            Iterable of documents.

    Returns:
        tuple:
            (is_valid, message)
    """

    if documents is None:
        return (
            False,
            "Document collection cannot be None.",
        )

    documents = list(documents)

    if not documents:
        return (
            False,
            "Document collection is empty.",
        )

    return (
        True,
        "Document collection is valid.",
    )

