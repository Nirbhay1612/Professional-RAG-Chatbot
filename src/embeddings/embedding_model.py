# src/embeddings/embedding_model.py

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL


# =========================================================
# EMBEDDING CONFIGURATION
# =========================================================

EMBEDDING_DEVICE = "cpu"

NORMALIZE_EMBEDDINGS = True


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace embedding model.

    The model is cached to prevent unnecessary reloads,
    which is especially useful with Streamlit.

    Returns:
        HuggingFaceEmbeddings:
            Configured embedding model.

    Raises:
        RuntimeError:
            If the embedding model cannot be initialized.
    """
    if not EMBEDDING_MODEL.strip():
        raise ValueError(
            "EMBEDDING_MODEL cannot be empty."
        )

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": EMBEDDING_DEVICE,
            },
            encode_kwargs={
                "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model "
            f"'{EMBEDDING_MODEL}': {exc}"
        ) from exc


# =========================================================
# GENERATE SINGLE EMBEDDING
# =========================================================

def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text.

    Args:
        text: Text to embed.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        TypeError:
            If text is not a string.

        ValueError:
            If text is empty.
    """
    if not isinstance(text, str):
        raise TypeError(
            "Text must be a string."
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    model = get_embedding_model()

    try:
        return model.embed_query(
            cleaned_text
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate embedding: {exc}"
        ) from exc


# =========================================================
# GENERATE MULTIPLE EMBEDDINGS
# =========================================================

def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of text strings.

    Returns:
        List of embedding vectors.

    Raises:
        TypeError:
            If texts is not a list or contains non-string values.

        ValueError:
            If any text is empty.
    """
    if not isinstance(texts, list):
        raise TypeError(
            "texts must be a list of strings."
        )

    if not texts:
        return []

    if not all(
        isinstance(text, str)
        for text in texts
    ):
        raise TypeError(
            "All items in texts must be strings."
        )

    cleaned_texts = [
        text.strip()
        for text in texts
    ]

    if any(
        not text
        for text in cleaned_texts
    ):
        raise ValueError(
            "Texts cannot contain empty strings."
        )

    model = get_embedding_model()

    try:
        return model.embed_documents(
            cleaned_texts
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate embeddings: {exc}"
        ) from exc


# =========================================================
# EMBEDDING DIMENSION
# =========================================================

@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """
    Determine the dimensionality of the embedding model.

    Returns:
        Number of dimensions in the embedding vector.
    """
    embedding = generate_embedding("test")

    return len(embedding)