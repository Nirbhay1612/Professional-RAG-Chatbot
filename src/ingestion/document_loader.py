# src/ingestion/document_loader.py

from pathlib import Path
from typing import Callable, Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)


# =========================================================
# SUPPORTED FILE TYPES
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}


# =========================================================
# INDIVIDUAL DOCUMENT LOADERS
# =========================================================

def load_pdf(file_path: Path) -> List[Document]:
    """
    Load a PDF document using PyPDFLoader.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects.
    """
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_txt(file_path: Path) -> List[Document]:
    """
    Load a UTF-8 text document.

    Args:
        file_path: Path to the TXT file.

    Returns:
        List of LangChain Document objects.
    """
    loader = TextLoader(
        str(file_path),
        encoding="utf-8",
    )

    return loader.load()


def load_docx(file_path: Path) -> List[Document]:
    """
    Load a DOCX document.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        List of LangChain Document objects.
    """
    loader = Docx2txtLoader(str(file_path))
    return loader.load()


# =========================================================
# LOADER MAPPING
# =========================================================

LOADER_MAP: Dict[str, Callable[[Path], List[Document]]] = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".docx": load_docx,
}


# =========================================================
# VALIDATION
# =========================================================

def validate_file(file_path: Path) -> None:
    """
    Validate a document path before loading.

    Args:
        file_path: Path to the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or format is unsupported.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Provided path is not a file: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported file format: {extension or '[none]'}. "
            f"Supported formats: {supported}"
        )


# =========================================================
# METADATA
# =========================================================

def add_document_metadata(
    documents: List[Document],
    file_path: Path,
) -> List[Document]:
    """
    Add standardized metadata to loaded documents.

    Args:
        documents: Loaded LangChain documents.
        file_path: Original document path.

    Returns:
        Documents with enriched metadata.
    """
    extension = file_path.suffix.lower().lstrip(".")

    for document in documents:
        document.metadata.update(
            {
                "source": file_path.name,
                "file_name": file_path.name,
                "file_type": extension,
            }
        )

    return documents


# =========================================================
# MAIN DOCUMENT LOADER
# =========================================================

def load_document(file_path: Path) -> List[Document]:
    """
    Load a supported document and return LangChain documents.

    Supported formats:
        - PDF
        - TXT
        - DOCX

    Args:
        file_path: Path to the document.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError:
            If the document does not exist.

        ValueError:
            If the path or file format is invalid.

        RuntimeError:
            If document loading fails.
    """
    file_path = Path(file_path)

    # Validate input
    validate_file(file_path)

    extension = file_path.suffix.lower()

    # Select appropriate loader
    loader_function = LOADER_MAP.get(extension)

    if loader_function is None:
        raise ValueError(
            f"No loader available for extension: {extension}"
        )

    # Load document
    try:
        documents = loader_function(file_path)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load document '{file_path.name}': {exc}"
        ) from exc

    # Validate extracted content
    if not documents:
        raise ValueError(
            f"No content could be extracted from "
            f"'{file_path.name}'."
        )

    # Add standardized metadata
    return add_document_metadata(
        documents,
        file_path,
    )