# config/settings.py

import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


# Create required directories
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_int_env(name: str, default: int) -> int:
    """
    Read an integer environment variable safely.

    Args:
        name: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        Integer value from the environment.
    """
    value = os.getenv(name, str(default))

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer. Got: {value!r}"
        ) from exc


# =========================================================
# GROQ CONFIGURATION
# =========================================================

# =========================================================
# GROQ CONFIGURATION
# =========================================================

def get_setting(name: str, default: str = "") -> str:
    """
    Read configuration from environment variables.

    Streamlit Cloud secrets are used as a fallback.
    """

    value = os.getenv(name)

    if value:
        return value.strip()

    try:
        import streamlit as st

        secret_value = st.secrets.get(name, default)

        if secret_value:
            return str(secret_value).strip()

    except (ImportError, FileNotFoundError):
        pass
    except Exception:
        pass

    return default


GROQ_API_KEY = get_setting("GROQ_API_KEY")

GROQ_MODEL = get_setting(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# =========================================================
# EMBEDDING CONFIGURATION
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# =========================================================
# RAG CONFIGURATION
# =========================================================

CHUNK_SIZE = get_int_env(
    "CHUNK_SIZE",
    1000,
)

CHUNK_OVERLAP = get_int_env(
    "CHUNK_OVERLAP",
    200,
)

RETRIEVAL_K = get_int_env(
    "RETRIEVAL_K",
    4,
)


# =========================================================
# VECTOR STORE CONFIGURATION
# =========================================================

VECTORSTORE_PATH = BASE_DIR / os.getenv(
    "VECTORSTORE_PATH",
    "data/vectorstore",
)


# =========================================================
# VALIDATION
# =========================================================

def validate_settings() -> None:
    """
    Validate application configuration.

    Raises:
        ValueError: If any required setting is invalid.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY to your .env file."
        )

    if not GROQ_MODEL.strip():
        raise ValueError(
            "GROQ_MODEL cannot be empty."
        )

    if not EMBEDDING_MODEL.strip():
        raise ValueError(
            "EMBEDDING_MODEL cannot be empty."
        )

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

    if RETRIEVAL_K <= 0:
        raise ValueError(
            "RETRIEVAL_K must be greater than 0."
        )


# Validate configuration when the application starts
validate_settings()