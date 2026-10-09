# src/chatbot/response_handler.py

"""
Response processing utilities for the Professional RAG Chatbot.

This module validates, cleans, normalizes, and formats
responses returned by the RAG pipeline.
"""

from typing import Any


# =========================================================
# DEFAULT FALLBACK MESSAGES
# =========================================================

INVALID_RESPONSE_MESSAGE = (
    "I couldn't generate a valid answer from the provided document."
)

NO_ANSWER_MESSAGE = (
    "I couldn't find the answer in the provided document."
)


# =========================================================
# RESPONSE VALIDATION
# =========================================================

def validate_response(response: Any) -> bool:
    """
    Validate whether a RAG response contains a usable answer.

    Supported response formats:
        - String
        - Dictionary containing an 'answer' field

    Args:
        response:
            Raw response returned by the RAG chain.

    Returns:
        True if a usable answer exists, otherwise False.
    """
    if response is None:
        return False

    if isinstance(response, str):
        return bool(response.strip())

    if isinstance(response, dict):
        answer = response.get("answer")

        return (
            isinstance(answer, str)
            and bool(answer.strip())
        )

    return False


# =========================================================
# CLEAN ANSWER
# =========================================================

def clean_answer(answer: str) -> str:
    """
    Clean generated answer text while preserving formatting.

    Operations:
        - Normalize line endings
        - Remove trailing whitespace
        - Remove duplicate blank lines
        - Remove leading/trailing whitespace

    Args:
        answer:
            Raw generated answer.

    Returns:
        Cleaned answer string.
    """
    if not isinstance(answer, str):
        return ""

    answer = answer.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = [
        line.rstrip()
        for line in answer.splitlines()
    ]

    cleaned_lines: list[str] = []

    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank and previous_blank:
            continue

        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(
        cleaned_lines
    ).strip()


# =========================================================
# EXTRACT ANSWER
# =========================================================

def extract_answer(response: Any) -> str:
    """
    Extract and clean the answer from a RAG response.

    Supported formats:

        String:
            "The answer is..."

        Dictionary:
            {
                "answer": "...",
                "sources": [...]
            }

    Args:
        response:
            Raw RAG response.

    Returns:
        Cleaned answer string.
    """
    if isinstance(response, str):
        return clean_answer(response)

    if isinstance(response, dict):
        answer = response.get(
            "answer",
            "",
        )

        if isinstance(answer, str):
            return clean_answer(answer)

    return ""


# =========================================================
# EXTRACT SOURCES
# =========================================================

def extract_sources(response: Any) -> list[str]:
    """
    Extract unique source names from a RAG response.

    Args:
        response:
            Raw RAG response.

    Returns:
        List of unique, cleaned source names.
    """
    if not isinstance(response, dict):
        return []

    sources = response.get(
        "sources",
        [],
    )

    if not isinstance(sources, list):
        return []

    cleaned_sources: list[str] = []

    for source in sources:
        if not isinstance(source, str):
            continue

        cleaned_source = source.strip()

        if (
            cleaned_source
            and cleaned_source not in cleaned_sources
        ):
            cleaned_sources.append(
                cleaned_source
            )

    return cleaned_sources


# =========================================================
# HANDLE RESPONSE
# =========================================================

def handle_response(response: Any) -> dict[str, Any]:
    """
    Normalize a raw RAG response into a consistent structure.

    Output format:

        {
            "answer": "...",
            "sources": [...]
        }

    Args:
        response:
            Raw response returned by the RAG chain.

    Returns:
        Normalized response dictionary.
    """
    if not validate_response(response):
        return {
            "answer": INVALID_RESPONSE_MESSAGE,
            "sources": [],
        }

    answer = extract_answer(response)
    sources = extract_sources(response)

    if not answer:
        answer = NO_ANSWER_MESSAGE

    return {
        "answer": answer,
        "sources": sources,
    }


# =========================================================
# FORMAT SOURCES FOR UI
# =========================================================

def format_sources(
    sources: list[str],
) -> str:
    """
    Format source names for Streamlit display.

    Args:
        sources:
            List of source names.

    Returns:
        Markdown-compatible source list.
    """
    if not sources:
        return ""

    formatted_sources = [
        f"{index}. {source}"
        for index, source in enumerate(
            sources,
            start=1,
        )
    ]

    return "\n".join(
        formatted_sources
    )