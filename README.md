# 🤖 Professional RAG Chatbot

A document-based **Retrieval-Augmented Generation (RAG) chatbot** built with Python, Streamlit, LangChain, FAISS, Sentence Transformers, and Groq.

Upload a PDF, TXT, or DOCX document and ask questions about its contents. The application retrieves relevant document chunks using semantic similarity search and uses a large language model (LLM) to generate context-grounded answers.

## ✨ Features

- 📄 **Document Upload:** Supports PDF, TXT, and DOCX files.
- 🧹 **Text Cleaning:** Normalizes and prepares extracted document text.
- ✂️ **Intelligent Chunking:** Splits documents into manageable text chunks.
- 🧠 **Semantic Embeddings:** Uses Sentence Transformers to represent text as vectors.
- 🔎 **Similarity Search:** Retrieves relevant document chunks for user questions.
- 🗂️ **FAISS Vector Store:** Enables efficient vector similarity search.
- 🤖 **LLM Integration:** Generates responses using Groq.
- 💬 **Interactive Chat UI:** Provides a Streamlit-based interface.
- 📚 **Source Information:** Displays retrieved source information alongside answers.
- 🔐 **Environment-Based Configuration:** Keeps API credentials outside application code.
- 🧪 **Automated Tests:** Includes tests for ingestion, retrieval, and RAG components.
- 🧩 **Modular Architecture:** Separates document processing, embeddings, retrieval, and response generation.

## 🖥️ Demo Link

- **[ASK Document PDF Chatbot](https://professional-rag-chatbot-zqtjbwxdba97tjtjcotzvt.streamlit.app/)**




### Request Flow

1. The user uploads a supported document.
2. The application loads and cleans its contents.
3. The text is split into smaller chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. FAISS stores and searches the document vectors.
6. The retriever finds relevant chunks for the user's question.
7. The RAG chain passes the retrieved context and question to the LLM.
8. The application displays the generated answer and available source information.

The chatbot is designed to ground answers in retrieved document content. Its accuracy still depends on document quality, retrieval relevance, and model behavior.

## 🏗️ Project Structure

```text
professional-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── text_cleaner.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── rag_chain.py
│   │   └── response_handler.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── validators.py
│
├── data/
│   ├── documents/
│   └── vectorstore/
│
└── test/
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_rag.py
```

> The tree above reflects the current project layout. The `.env` file and generated vector-store files should remain local and should not be committed to GitHub.

## 🛠️ Tech Stack


[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)

[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)

[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://sbert.net/)

[![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

[![PyPDF](https://img.shields.io/badge/PyPDF-3776AB?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypdf.readthedocs.io/)

[![python--docx](https://img.shields.io/badge/python--docx-2B579A?style=for-the-badge&logo=microsoftword&logoColor=white)](https://python-docx.readthedocs.io/)

[![python-dotenv](https://img.shields.io/badge/python--dotenv-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/theskumar/python-dotenv)

[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)


## 🧩 Core Components

### 1. Document Ingestion

**Files:** `document_loader.py`, `text_cleaner.py`, `chunker.py`

Loads supported documents, normalizes extracted text, and divides the content into smaller chunks for retrieval.

### 2. Embedding Generation

**File:** `embedding_model.py`

Uses the configured Sentence Transformers model to convert text into numerical vector representations.

### 3. Vector Store

**File:** `vector_store.py`

Creates and manages the FAISS vector store used for semantic similarity search.

### 4. Retrieval

**File:** `retriever.py`

Finds relevant document chunks for a user's question and provides the context used by the RAG pipeline.

### 5. RAG Chain

**Files:** `rag_chain.py`, `prompts.py`

Combines the question, retrieved context, and prompt before invoking the configured Groq LLM.

### 6. Response Handling

**File:** `response_handler.py`

Validates and normalizes generated responses and extracts available source information.

### 7. Application Interface

**File:** `app.py`

Provides document upload, processing status, interactive chat, response display, and session controls through Streamlit.

## 🧪 Testing

The project includes automated tests for document ingestion, retrieval, and RAG behavior.

Run the full test suite from the project root:

```bash
python -m pytest -v
```

Run individual test modules:

```bash
python -m pytest test/test_ingestion.py -v
python -m pytest test/test_retrieval.py -v
python -m pytest test/test_rag.py -v
```

**Latest local test result:** 63 tests passed, with 1 warning.

This result reflects the local test run. Results may vary across environments depending on dependency versions and configuration.

## 🔒 Security and Limitations

- Store API credentials in environment variables.
- Keep `.env` out of Git commits.
- Validate uploaded documents before processing.
- Do not load untrusted serialized vector-store files.
- Review retrieved context and generated answers for accuracy.
- Add authentication, authorization, and appropriate upload limits before supporting multiple users in production.
- Avoid uploading confidential documents to external services unless permitted by the relevant data policies.

## 🚀 Future Improvements

Potential enhancements include:

- Conversational memory
- Multi-document knowledge bases
- Hybrid search and reranking
- Retrieval-quality evaluation
- More detailed source citations
- Persistent document indexing
- Authentication and multi-user support
- Automated RAG evaluation
- Performance monitoring
- Production deployment and observability

## 🎯 Potential Use Cases

This architecture can be adapted for:

- Document question answering
- Internal knowledge assistants
- Research and educational assistants
- Customer support knowledge bases
- Company policies and technical documentation
- Product documentation assistants

These are potential applications of the architecture, not claims that separate production systems have already been implemented.

## 📌 Project Status

**Status: Core implementation completed; further improvements planned.**

The application runs locally, and the current test suite has passed. Additional enhancements can be introduced as the project evolves.

## 👨‍💻 Author

**Nirbhay**

Built as a practical project to explore Python, LLM integration, embeddings, vector search, and Retrieval-Augmented Generation.

## ⭐ Support

If you find this project useful, consider giving the repository a star on GitHub.
