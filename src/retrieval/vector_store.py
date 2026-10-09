# src/retrieval/vector_store.py

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import VECTORSTORE_PATH
from src.embeddings.embedding_model import get_embedding_model


# =========================================================
# VECTOR STORE CONFIGURATION
# =========================================================

VECTOR_STORE_PATH = Path(VECTORSTORE_PATH)

FAISS_INDEX_FILE = "index.faiss"
FAISS_METADATA_FILE = "index.pkl"


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(
    documents: List[Document],
    embedding_model=None,
) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Args:
        documents:
            List of LangChain Document chunks.

        embedding_model:
            Optional embedding model. If not provided,
            the configured embedding model is used.

    Returns:
        FAISS:
            Initialized FAISS vector store.

    Raises:
        ValueError:
            If documents are empty or invalid.
        RuntimeError:
            If FAISS creation fails.
    """
    if not documents:
        raise ValueError(
            "Cannot create vector store from empty documents."
        )

    if not all(
        isinstance(document, Document)
        for document in documents
    ):
        raise ValueError(
            "All items must be LangChain Document objects."
        )

    if embedding_model is None:
        embedding_model = get_embedding_model()

    try:
        return FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to create FAISS vector store: {exc}"
        ) from exc


# =========================================================
# SAVE VECTOR STORE
# =========================================================

def save_vector_store(
    vector_store: FAISS,
    path: Optional[Path] = None,
) -> Path:
    """
    Save a FAISS vector store locally.

    Args:
        vector_store:
            FAISS vector store to save.

        path:
            Target directory. Defaults to configured path.

    Returns:
        Path:
            Directory containing the saved vector store.

    Raises:
        ValueError:
            If vector_store is None.
        RuntimeError:
            If saving fails.
    """
    if vector_store is None:
        raise ValueError(
            "Vector store cannot be None."
        )

    save_path = Path(
        path or VECTOR_STORE_PATH
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        vector_store.save_local(
            str(save_path)
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to save FAISS vector store: {exc}"
        ) from exc

    return save_path


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store(
    path: Optional[Path] = None,
) -> FAISS:
    """
    Load an existing local FAISS vector store.

    The vector store must contain both:
        - index.faiss
        - index.pkl

    Args:
        path:
            Directory containing the vector store.

    Returns:
        FAISS:
            Loaded vector store.

    Raises:
        FileNotFoundError:
            If required FAISS files are missing.
        RuntimeError:
            If loading fails.
    """
    load_path = Path(
        path or VECTOR_STORE_PATH
    )

    index_file = load_path / FAISS_INDEX_FILE
    metadata_file = load_path / FAISS_METADATA_FILE

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at: {index_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found at: {metadata_file}"
        )

    embedding_model = get_embedding_model()

    try:
        return FAISS.load_local(
            str(load_path),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FAISS vector store "
            f"from '{load_path}': {exc}"
        ) from exc


# =========================================================
# CHECK VECTOR STORE
# =========================================================

def vector_store_exists(
    path: Optional[Path] = None,
) -> bool:
    """
    Check whether a complete FAISS vector store exists.

    Args:
        path:
            Directory containing the vector store.

    Returns:
        bool:
            True if both required FAISS files exist.
    """
    store_path = Path(
        path or VECTOR_STORE_PATH
    )

    return (
        (store_path / FAISS_INDEX_FILE).is_file()
        and
        (store_path / FAISS_METADATA_FILE).is_file()
    )


# =========================================================
# DELETE VECTOR STORE
# =========================================================

def delete_vector_store(
    path: Optional[Path] = None,
) -> bool:
    """
    Delete a saved FAISS vector store.

    Only known FAISS files are removed.

    Args:
        path:
            Directory containing the vector store.

    Returns:
        bool:
            True if at least one FAISS file was deleted.
    """
    store_path = Path(
        path or VECTOR_STORE_PATH
    )

    if not store_path.exists():
        return False

    deleted = False

    for file_name in (
        FAISS_INDEX_FILE,
        FAISS_METADATA_FILE,
    ):
        file_path = store_path / file_name

        if file_path.is_file():
            file_path.unlink()
            deleted = True

    return deleted


# =========================================================
# VECTOR STORE STATISTICS
# =========================================================

def get_vector_store_size(
    vector_store: FAISS,
) -> int:
    """
    Return the number of vectors stored in FAISS.

    Args:
        vector_store:
            FAISS vector store.

    Returns:
        int:
            Number of stored vectors.
    """
    if vector_store is None:
        return 0

    return int(vector_store.index.ntotal)