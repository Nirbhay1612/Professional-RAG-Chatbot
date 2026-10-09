"""
Tests for the Professional RAG Chatbot.

These tests do not call the real Groq API.
"""


from typing import Any

import pytest

from langchain_core.documents import Document

from src.chatbot.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    create_rag_prompt,
)

from src.chatbot.rag_chain import (
    RAGChain,
    extract_sources,
    format_documents,
)

from src.chatbot.response_handler import (
    clean_answer,
    extract_answer,
    extract_sources as extract_response_sources,
    format_sources,
    handle_response,
    validate_response,
)


# ============================================================
# FAKE LLM
# ============================================================


class FakeResponse:
    """Fake LLM response."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Fake LLM used instead of the real Groq API."""

    def __init__(
        self,
        response: str = "The return period is 30 days.",
    ) -> None:
        self.response = response
        self.last_input: Any = None

    def invoke(
        self,
        messages: Any,
    ) -> FakeResponse:
        """Return a fake response."""

        self.last_input = messages

        return FakeResponse(
            self.response
        )


# ============================================================
# FAKE RETRIEVER
# ============================================================


class FakeRetriever:
    """Fake retriever for unit tests."""

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents

    def invoke(
        self,
        query: str,
    ) -> list[Document]:
        """Return test documents."""

        return self.documents


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents."""

    return [
        Document(
            page_content=(
                "Customers can return products "
                "within 30 days of delivery."
            ),
            metadata={
                "source": "return_policy.txt",
                "page": 1,
            },
        ),
        Document(
            page_content=(
                "Refunds are processed within "
                "5 to 7 business days."
            ),
            metadata={
                "source": "refund_policy.txt",
                "page": 1,
            },
        ),
    ]


@pytest.fixture
def fake_llm() -> Any:
    """Create fake LLM."""

    return FakeLLM()


@pytest.fixture
def fake_retriever(
    sample_documents: list[Document],
) -> Any:
    """Create fake retriever."""

    return FakeRetriever(
        sample_documents
    )


@pytest.fixture
def rag_chain(
    fake_retriever: Any,
    fake_llm: Any,
) -> RAGChain:
    """Create RAG chain with fake components."""

    return RAGChain(
        retriever=fake_retriever,
        llm=fake_llm,
    )


# ============================================================
# PROMPT TESTS
# ============================================================


def test_create_rag_prompt() -> None:
    """Test RAG prompt creation."""

    prompt = create_rag_prompt()

    assert prompt is not None

    messages = prompt.format_messages(
        context="Test document context.",
        input="What is the return policy?",
    )

    assert len(messages) == 2


def test_system_prompt_contains_context() -> None:
    """Test context placeholder."""

    assert "{context}" in SYSTEM_PROMPT


def test_user_prompt_contains_input() -> None:
    """Test input placeholder."""

    assert "{input}" in USER_PROMPT


def test_prompt_contains_question() -> None:
    """Test user question appears in prompt."""

    prompt = create_rag_prompt()

    messages = prompt.format_messages(
        context="Return period is 30 days.",
        input="What is the return policy?",
    )

    text = "\n".join(
        str(message.content)
        for message in messages
    )

    assert "What is the return policy?" in text


def test_prompt_contains_context() -> None:
    """Test document context appears in prompt."""

    prompt = create_rag_prompt()

    messages = prompt.format_messages(
        context="Products can be returned within 30 days.",
        input="How many days do I have?",
    )

    text = "\n".join(
        str(message.content)
        for message in messages
    )

    assert (
        "Products can be returned within 30 days."
        in text
    )


# ============================================================
# DOCUMENT FORMATTER TESTS
# ============================================================


def test_format_documents(
    sample_documents: list[Document],
) -> None:
    """Test document formatting."""

    context = format_documents(
        sample_documents
    )

    assert isinstance(context, str)
    assert "return_policy.txt" in context
    assert "refund_policy.txt" in context
    assert "30 days" in context
    assert "5 to 7 business days" in context


def test_format_documents_empty() -> None:
    """Test empty documents."""

    context = format_documents([])

    assert context == ""


def test_extract_sources(
    sample_documents: list[Document],
) -> None:
    """Test source extraction."""

    sources = extract_sources(
        sample_documents
    )

    assert isinstance(sources, list)
    assert "return_policy.txt" in sources
    assert "refund_policy.txt" in sources


def test_extract_sources_unique() -> None:
    """Test duplicate sources are removed."""

    documents = [
        Document(
            page_content="First.",
            metadata={
                "source": "policy.txt"
            },
        ),
        Document(
            page_content="Second.",
            metadata={
                "source": "policy.txt"
            },
        ),
    ]

    sources = extract_sources(
        documents
    )

    assert sources == ["policy.txt"]


# ============================================================
# RAG CHAIN TESTS
# ============================================================


def test_rag_chain_requires_retriever(
    fake_llm: Any,
) -> None:
    """Retriever is required."""

    with pytest.raises(ValueError):
        RAGChain(
            retriever=None,
            llm=fake_llm,
        )


def test_rag_chain_creation(
    rag_chain: RAGChain,
) -> None:
    """Test RAGChain creation."""

    assert rag_chain is not None
    assert rag_chain.retriever is not None
    assert rag_chain.llm is not None
    assert rag_chain.prompt is not None


def test_rag_chain_retrieve(
    rag_chain: RAGChain,
) -> None:
    """Test document retrieval."""

    documents = rag_chain.retrieve(
        "What is the return policy?"
    )

    assert isinstance(documents, list)
    assert len(documents) == 2
    assert "30 days" in documents[0].page_content


def test_rag_chain_generate_answer(
    rag_chain: RAGChain,
    sample_documents: list[Document],
) -> None:
    """Test answer generation."""

    context = format_documents(
        sample_documents
    )

    answer = rag_chain.generate_answer(
        question="What is the return policy?",
        context=context,
    )

    assert isinstance(answer, str)
    assert answer != ""
    assert "30 days" in answer


def test_rag_chain_empty_context(
    rag_chain: RAGChain,
) -> None:
    """Test empty context."""

    answer = rag_chain.generate_answer(
        question="What is the return policy?",
        context="",
    )

    assert isinstance(answer, str)
    assert "couldn't find" in answer.lower()


def test_rag_chain_empty_question(
    rag_chain: RAGChain,
) -> None:
    """Test empty question."""

    with pytest.raises(ValueError):
        rag_chain.generate_answer(
            question="",
            context="Some context.",
        )


def test_rag_chain_invoke(
    rag_chain: RAGChain,
) -> None:
    """Test complete RAG pipeline."""

    result = rag_chain.invoke(
        {
            "input": "What is the return policy?"
        }
    )

    assert isinstance(result, dict)

    assert "answer" in result
    assert "context" in result
    assert "sources" in result

    assert isinstance(
        result["answer"],
        str,
    )

    assert isinstance(
        result["context"],
        list,
    )

    assert isinstance(
        result["sources"],
        list,
    )

    assert "30 days" in result["answer"]


def test_rag_chain_missing_input(
    rag_chain: RAGChain,
) -> None:
    """Test missing input."""

    with pytest.raises(ValueError):
        rag_chain.invoke({})


def test_rag_chain_empty_input(
    rag_chain: RAGChain,
) -> None:
    """Test empty input."""

    with pytest.raises(ValueError):
        rag_chain.invoke(
            {
                "input": ""
            }
        )


# ============================================================
# RESPONSE HANDLER TESTS
# ============================================================


def test_validate_response() -> None:
    """Test response validation."""

    response = {
        "answer": "The return period is 30 days.",
        "sources": ["return_policy.txt"],
    }

    assert validate_response(
        response
    ) is True


def test_invalid_response() -> None:
    """Test invalid response."""

    assert validate_response(
        None
    ) is False

    assert validate_response(
        {}
    ) is False


def test_clean_answer() -> None:
    """Test answer cleaning."""

    answer = (
        "   The return period is 30 days.   "
    )

    cleaned = clean_answer(answer)

    assert cleaned == (
        "The return period is 30 days."
    )


def test_extract_answer() -> None:
    """Test answer extraction."""

    response = {
        "answer": (
            "Products can be returned "
            "within 30 days."
        ),
        "sources": [
            "return_policy.txt"
        ],
    }

    answer = extract_answer(
        response
    )

    assert answer == (
        "Products can be returned "
        "within 30 days."
    )


def test_extract_response_sources() -> None:
    """Test source extraction."""

    response = {
        "answer": "The answer.",
        "sources": [
            "return_policy.txt",
            "refund_policy.txt",
        ],
    }

    sources = extract_response_sources(
        response
    )

    assert isinstance(sources, list)
    assert "return_policy.txt" in sources
    assert "refund_policy.txt" in sources


def test_format_sources() -> None:
    """Test source formatting."""

    sources = [
        "return_policy.txt",
        "refund_policy.txt",
    ]

    formatted = format_sources(
        sources
    )

    assert isinstance(
        formatted,
        str,
    )

    assert "return_policy.txt" in formatted
    assert "refund_policy.txt" in formatted


def test_handle_response() -> None:
    """Test response handler."""

    response = {
        "answer": (
            "The return period is 30 days."
        ),
        "sources": [
            "return_policy.txt"
        ],
    }

    result = handle_response(
        response
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["answer"] == (
        "The return period is 30 days."
    )

    assert result["sources"] == [
        "return_policy.txt"
    ]


# ============================================================
# END-TO-END TEST
# ============================================================


def test_end_to_end_rag(
    rag_chain: RAGChain,
) -> None:
    """Test complete RAG workflow."""

    result = rag_chain.invoke(
        {
            "input": (
                "How long can I return a product?"
            )
        }
    )

    assert result["answer"]
    assert result["context"]
    assert result["sources"]

    assert "30 days" in result["answer"]

    assert (
        "return_policy.txt"
        in result["sources"]
    )