# app.py

"""
Professional RAG Chatbot
------------------------
A Streamlit application for document-based question answering
using Retrieval-Augmented Generation (RAG).
"""

from pathlib import Path
import html
import tempfile
import time

import streamlit as st

from src.ingestion.document_loader import load_document
from src.ingestion.text_cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import create_retriever
from src.chatbot.rag_chain import create_rag_chain
from src.chatbot.response_handler import handle_response


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Professional RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Main container ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* ---------- Header ---------- */

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.8rem;
    }


    /* ---------- Document status ---------- */

    .document-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.06);
        margin-bottom: 1.5rem;
    }

    .document-name {
        font-weight: 600;
        font-size: 1rem;
    }

    .document-status {
        color: #777;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }


    /* ---------- Sidebar ---------- */

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
    }

    .sidebar-description {
        color: #777;
        font-size: 0.9rem;
    }


    /* ---------- Chat ---------- */

    .chat-info {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.07);
        margin-bottom: 1rem;
        color: #777;
    }


    /* ---------- Sources ---------- */

    .source-item {
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session_state() -> None:
    """Initialize Streamlit session state."""

    defaults = {
        "messages": [],
        "vector_store": None,
        "retriever": None,
        "rag_chain": None,
        "processed_file": None,
        "document_ready": False,
        "chunk_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =========================================================
# FILE UTILITIES
# =========================================================

SUPPORTED_FILE_TYPES = [
    "pdf",
    "txt",
    "docx",
]


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save an uploaded Streamlit file temporarily.

    Args:
        uploaded_file:
            Streamlit UploadedFile object.

    Returns:
        Path:
            Temporary file path.
    """
    if uploaded_file is None:
        raise ValueError(
            "No file was provided."
        )

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix.lstrip(".") not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            "Unsupported file type."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        return Path(
            temp_file.name
        )


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

def process_document(uploaded_file):
    """
    Run the complete document ingestion pipeline.

    Pipeline:

        Upload
          ↓
        Load
          ↓
        Clean
          ↓
        Chunk
          ↓
        Embeddings
          ↓
        FAISS Vector Store
          ↓
        Retriever
          ↓
        RAG Chain
    """

    temp_path = None

    try:
        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------

        temp_path = save_uploaded_file(
            uploaded_file
        )

        with st.status(
            "Processing document...",
            expanded=True,
        ) as status:

            # -------------------------------------------------
            # Load
            # -------------------------------------------------

            st.write(
                "📄 Loading document..."
            )

            documents = load_document(
                temp_path
            )

            if not documents:
                raise ValueError(
                    "No readable content found in the document."
                )

            st.write(
                f"✓ Loaded {len(documents)} document section(s)"
            )

            # -------------------------------------------------
            # Clean
            # -------------------------------------------------

            st.write(
                "🧹 Cleaning document..."
            )

            cleaned_documents = clean_documents(
                documents
            )

            if not cleaned_documents:
                raise ValueError(
                    "Document became empty after cleaning."
                )

            # -------------------------------------------------
            # Chunk
            # -------------------------------------------------

            st.write(
                "✂️ Creating text chunks..."
            )

            chunks = chunk_documents(
                cleaned_documents
            )

            if not chunks:
                raise ValueError(
                    "Unable to generate document chunks."
                )

            st.write(
                f"✓ Created {len(chunks)} chunks"
            )

            # -------------------------------------------------
            # Embeddings
            # -------------------------------------------------

            st.write(
                "🧠 Loading embedding model..."
            )

            embedding_model = (
                get_embedding_model()
            )

            # -------------------------------------------------
            # Vector Store
            # -------------------------------------------------

            st.write(
                "🗂️ Building FAISS vector store..."
            )

            vector_store = create_vector_store(
                documents=chunks,
                embedding_model=embedding_model,
            )

            # -------------------------------------------------
            # Retriever
            # -------------------------------------------------

            st.write(
                "🔎 Creating document retriever..."
            )

            retriever = create_retriever(
                vector_store
            )

            # -------------------------------------------------
            # RAG Chain
            # -------------------------------------------------

            st.write(
                "🤖 Initializing RAG assistant..."
            )

            rag_chain = create_rag_chain(
                retriever
            )

            # -------------------------------------------------
            # Store pipeline in session
            # -------------------------------------------------

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.retriever = (
                retriever
            )

            st.session_state.rag_chain = (
                rag_chain
            )

            st.session_state.processed_file = (
                uploaded_file.name
            )

            st.session_state.document_ready = True

            st.session_state.chunk_count = (
                len(chunks)
            )

            # New document = new conversation
            st.session_state.messages = []

            status.update(
                label="Document processed successfully!",
                state="complete",
                expanded=False,
            )

        return len(chunks)

    except Exception as error:

        st.session_state.document_ready = False
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.rag_chain = None

        st.error(
            f"Document processing failed: {error}"
        )

        return None

    finally:

        # Always remove temporary file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# =========================================================
# SESSION CONTROLS
# =========================================================

def clear_chat() -> None:
    """Clear conversation history."""
    st.session_state.messages = []


def reset_document() -> None:
    """Reset the active document and RAG pipeline."""

    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.processed_file = None
    st.session_state.document_ready = False
    st.session_state.chunk_count = 0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📚 RAG Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-description">
        Upload a document and ask questions using
        Retrieval-Augmented Generation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    st.subheader("Document")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=SUPPORTED_FILE_TYPES,
        help="Supported formats: PDF, TXT and DOCX.",
    )

    if uploaded_file is not None:

        st.info(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Process Document",
            type="primary",
            use_container_width=True,
        ):

            process_document(
                uploaded_file
            )

    st.divider()

    # -----------------------------------------------------
    # Document status
    # -----------------------------------------------------

    st.subheader("Status")

    if st.session_state.document_ready:

        st.success(
            "Document ready"
        )

        st.caption(
            f"📄 {st.session_state.processed_file}"
        )

        if st.session_state.chunk_count:
            st.caption(
                f"🧩 {st.session_state.chunk_count} chunks"
            )

    else:

        st.warning(
            "No document processed"
        )

    st.divider()

    # -----------------------------------------------------
    # Controls
    # -----------------------------------------------------

    st.subheader("Controls")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        clear_chat()
        st.rerun()

    if st.button(
        "🔄 Reset Document",
        use_container_width=True,
    ):

        reset_document()
        st.rerun()

    st.divider()

    st.caption(
        "Python • Streamlit • LangChain • FAISS • Groq"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Professional RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about your documents using
    Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# ACTIVE DOCUMENT
# =========================================================

if st.session_state.document_ready:

    safe_filename = html.escape(
        str(
            st.session_state.processed_file
        )
    )

    st.markdown(
        f"""
        <div class="document-card">
            <div class="document-name">
                📄 {safe_filename}
            </div>
            <div class="document-status">
                Document processed and ready for questions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="chat-info">
        👈 Upload and process a document from the sidebar
        to start asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(role):

        st.markdown(content)

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                "📚 Sources",
                expanded=False,
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    st.markdown(
                        f"**Source {index}:** {source}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.document_ready,
)


# =========================================================
# GENERATE RESPONSE
# =========================================================

if question:

    question = question.strip()

    if not question:
        st.warning(
            "Please enter a question."
        )
        st.stop()

    # -----------------------------------------------------
    # Add user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------------------------------
    # Generate assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:

            start_time = time.perf_counter()

            with st.spinner(
                "Searching your document..."
            ):

                raw_response = (
                    st.session_state.rag_chain.invoke(
                        {
                            "input": question,
                        }
                    )
                )

            # Normalize response
            response = handle_response(
                raw_response
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            answer = response.get(
                "answer",
                "Unable to generate an answer.",
            )

            sources = response.get(
                "sources",
                [],
            )

            # -------------------------------------------------
            # Display answer
            # -------------------------------------------------

            response_placeholder.markdown(
                answer
            )

            # -------------------------------------------------
            # Display sources
            # -------------------------------------------------

            if sources:

                with st.expander(
                    "📚 Sources",
                    expanded=False,
                ):

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"**Source {index}:** {source}"
                        )

            # -------------------------------------------------
            # Response metadata
            # -------------------------------------------------

            st.caption(
                f"Response generated in "
                f"{elapsed_time:.2f} seconds"
            )

            # -------------------------------------------------
            # Save assistant message
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:

            error_message = (
                "I couldn't generate an answer. "
                "Please try again or check the document."
            )

            response_placeholder.error(
                error_message
            )

            # Keep technical details available
            # during development.
            with st.expander(
                "Technical details",
                expanded=False,
            ):

                st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Professional RAG Assistant •
        Document Retrieval • Semantic Search • LLM
    </div>
    """,
    unsafe_allow_html=True,
)