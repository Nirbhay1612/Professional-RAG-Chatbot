"""
RAG chain for the Professional RAG Chatbot.

This module connects:
    Retriever -> Context -> Prompt -> Groq LLM -> Response
"""

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from pydantic import SecretStr

from config.settings import GROQ_API_KEY, GROQ_MODEL
from src.chatbot.prompts import create_rag_prompt


# =========================================================
# CONFIGURATION
# =========================================================

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

NO_CONTEXT_MESSAGE = (
    "I couldn't find the answer in the provided document."
)


# =========================================================
# LLM
# =========================================================

@lru_cache(maxsize=1)
def create_llm() -> ChatGroq:
    """
    Create and cache the Groq chat model.

    Returns:
        ChatGroq:
            Configured Groq LLM.

    Raises:
        ValueError:
            If the Groq API key or model is missing.
        RuntimeError:
            If the LLM cannot be initialized.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please configure it in your .env file."
        )

    if not GROQ_MODEL.strip():
        raise ValueError(
            "GROQ_MODEL cannot be empty."
        )

    try:
        return ChatGroq(
            api_key=SecretStr(GROQ_API_KEY),
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Groq LLM: {exc}"
        ) from exc


# =========================================================
# CONTEXT FORMATTER
# =========================================================

def format_documents(
    documents: list[Document],
) -> str:
    """
    Convert retrieved documents into structured RAG context.

    Args:
        documents:
            Retrieved LangChain documents.

    Returns:
        Formatted context string.
    """
    if not documents:
        return ""

    formatted_parts: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        if not isinstance(document, Document):
            continue

        content = document.page_content.strip()

        if not content:
            continue

        metadata = document.metadata or {}

        source = str(
            metadata.get(
                "source",
                "Unknown source",
            )
        )

        page = metadata.get("page")

        if page is not None:
            location = (
                f"{source}, page {page}"
            )
        else:
            location = source

        formatted_parts.append(
            f"[Document {index}]\n"
            f"Source: {location}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(
        formatted_parts
    )


# =========================================================
# SOURCE EXTRACTION
# =========================================================

def extract_sources(
    documents: list[Document],
) -> list[str]:
    """
    Extract unique source names from retrieved documents.

    Args:
        documents:
            Retrieved documents.

    Returns:
        Unique source names.
    """
    sources: list[str] = []

    for document in documents:
        if not isinstance(document, Document):
            continue

        metadata = document.metadata or {}

        source = metadata.get("source")

        if not source:
            continue

        source_string = str(source)

        if source_string not in sources:
            sources.append(source_string)

    return sources


# =========================================================
# RAG CHAIN
# =========================================================

class RAGChain:
    """
    Complete retrieval-augmented generation pipeline.

    Pipeline:
        Question
            ↓
        Retriever
            ↓
        Relevant Documents
            ↓
        Context
            ↓
        Prompt
            ↓
        Groq LLM
            ↓
        Answer + Sources
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: ChatGroq | None = None,
    ) -> None:
        if retriever is None:
            raise ValueError(
                "Retriever cannot be None."
            )

        self.retriever = retriever
        self.llm = (
            llm
            if llm is not None
            else create_llm()
        )
        self.prompt = create_rag_prompt()

    # =====================================================
    # QUESTION VALIDATION
    # =====================================================

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        """
        Validate and normalize a user question.

        Args:
            question:
                User question.

        Returns:
            Cleaned question.
        """
        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "Question cannot be empty."
            )

        return cleaned_question

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        question: str,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a question.

        Args:
            question:
                User question.

        Returns:
            Retrieved documents.

        Raises:
            RuntimeError:
                If retrieval fails.
        """
        cleaned_question = self._validate_question(
            question
        )

        try:
            documents = self.retriever.invoke(
                cleaned_question
            )

        except Exception as exc:
            raise RuntimeError(
                f"Document retrieval failed: {exc}"
            ) from exc

        if not documents:
            return []

        return [
            document
            for document in documents
            if isinstance(document, Document)
        ]

    # =====================================================
    # ANSWER GENERATION
    # =====================================================

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate an answer from retrieved context.

        Args:
            question:
                User question.

            context:
                Formatted retrieved context.

        Returns:
            Generated answer.

        Raises:
            RuntimeError:
                If LLM generation fails.
        """
        cleaned_question = self._validate_question(
            question
        )

        if not context or not context.strip():
            return NO_CONTEXT_MESSAGE

        messages = self.prompt.invoke(
            {
                "context": context,
                "input": cleaned_question,
            }
        )

        try:
            response = self.llm.invoke(
                messages
            )

        except Exception as exc:
            raise RuntimeError(
                f"LLM response generation failed: {exc}"
            ) from exc

        content = getattr(
            response,
            "content",
            "",
        )

        if not isinstance(content, str):
            content = str(content)

        answer = content.strip()

        if not answer:
            return NO_CONTEXT_MESSAGE

        return answer

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def invoke(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the complete RAG pipeline.

        Args:
            inputs:
                Dictionary containing the user question
                under the 'input' key.

        Returns:
            Dictionary containing:
                - answer
                - context
                - sources

        Raises:
            TypeError:
                If inputs is not a dictionary.
            ValueError:
                If the input question is invalid.
        """
        if not isinstance(inputs, dict):
            raise TypeError(
                "Input must be a dictionary."
            )

        question = inputs.get("input")

        if not isinstance(question, str):
            raise ValueError(
                "Input must contain a string "
                "value under the 'input' key."
            )

        cleaned_question = self._validate_question(
            question
        )

        # Retrieve relevant documents
        documents = self.retrieve(
            cleaned_question
        )

        # Build context
        context = format_documents(
            documents
        )

        # Generate answer
        answer = self.generate_answer(
            question=cleaned_question,
            context=context,
        )

        # Extract sources
        sources = extract_sources(
            documents
        )

        return {
            "answer": answer,
            "context": documents,
            "sources": sources,
        }


# =========================================================
# FACTORY
# =========================================================

def create_rag_chain(
    retriever: BaseRetriever,
) -> RAGChain:
    """
    Create a configured RAG chain.

    Args:
        retriever:
            LangChain retriever.

    Returns:
        Configured RAGChain instance.
    """
    return RAGChain(
        retriever=retriever,
    )