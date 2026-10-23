# AI Document Intelligence API

A document intelligence backend built with **FastAPI, LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG)**. The application enables users to upload PDF documents and ask natural-language questions about their contents, generating context-aware answers supported by retrieved document passages.

## Overview

Large Language Models (LLMs) can generate answers that are not grounded in a user's documents. This project addresses that problem by implementing a RAG pipeline that retrieves relevant document content before generating an answer.

The application processes uploaded PDFs, splits their content into manageable chunks, generates vector embeddings, and stores them in ChromaDB. When a user submits a question, the system retrieves relevant passages and passes them to an LLM to generate an answer with source references.

The backend supports **Groq and Ollama** as configurable LLM providers and can run locally or inside Docker.

## Key Features

- **PDF ingestion:** Upload and extract text from PDF documents.
- **Document chunking:** Split extracted text into overlapping chunks for retrieval.
- **Semantic search:** Generate sentence-transformer embeddings and retrieve relevant content using ChromaDB.
- **Document-specific question answering:** Query uploaded documents using their unique document IDs.
- **Grounded responses:** Return generated answers with source passages and page references.
- **Multiple LLM providers:** Switch between Groq and Ollama through environment configuration.
- **Persistent vector storage:** Preserve document embeddings using ChromaDB and Docker volumes.
- **RAG evaluation:** Evaluate answer quality using DeepEval metrics.
- **Automated testing:** Validate application behavior using pytest.
- **Containerized deployment:** Run the backend with Docker and Docker Compose.

## Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.14 | Core application development |
| API Framework | FastAPI | REST API development, request validation, and automatic API documentation |
| LLM Orchestration | LangChain | Prompt management, model integration, and RAG workflows |
| Cloud LLM | Groq | Cloud-hosted language model inference |
| Local LLM | Ollama | Local language model inference |
| Embeddings | Hugging Face Sentence Transformers | Generate semantic vector representations of document chunks |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Convert text into 384-dimensional embeddings |
| Vector Database | ChromaDB | Store document embeddings and perform semantic retrieval |
| PDF Processing | pypdf | Extract text and metadata from PDF files |
| RAG Evaluation | DeepEval | Evaluate answer relevancy, faithfulness, and retrieval quality |
| Testing | pytest | Automated unit and integration testing |
| Package Management | uv | Dependency management and reproducible environments |
| Containerization | Docker | Package the application and its dependencies |
| Container Orchestration | Docker Compose | Configure application services, environment variables, and persistent volumes |

### LLM Configuration

The application supports two configurable LLM providers:

- **Groq:** Cloud-based inference using the configured Groq model.
- **Ollama:** Local inference using the configured Ollama model.

The active provider is selected using the `LLM_PROVIDER` environment variable, allowing the application to switch between providers without modifying the RAG pipeline.

## System Architecture

The application follows a modular Retrieval-Augmented Generation (RAG) architecture with two primary workflows.

### 1. Document Ingestion Pipeline

1. A user uploads a PDF through the FastAPI endpoint.
2. The application extracts text and page metadata using `pypdf`.
3. The extracted content is divided into overlapping text chunks.
4. The embedding model converts each chunk into a numerical vector.
5. ChromaDB stores the vectors, document identifiers, and associated metadata for later retrieval.

### 2. Question Answering Pipeline

1. A user submits a question associated with a previously uploaded document.
2. The application searches ChromaDB for semantically relevant chunks belonging to that document.
3. The retrieved chunks are combined into a context for the LLM.
4. LangChain constructs the prompt and invokes the configured Groq or Ollama model.
5. The API returns the generated answer along with retrieved source passages, page numbers, and retrieval scores.

### Persistence and Deployment

The FastAPI application runs inside a Docker container managed by Docker Compose. Named Docker volumes preserve the ChromaDB database and Hugging Face model cache across container recreation.

Application configuration is managed through environment variables, allowing different LLM providers and storage locations to be selected without changing the application code.
## API Endpoints

The application provides REST APIs for document ingestion, question answering, and application monitoring.

Interactive API documentation is available through FastAPI Swagger UI at:

`http://localhost:8000/docs`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents` | Upload a PDF document for text extraction, chunking, embedding generation, and vector storage |
| `POST` | `/documents/{document_id}/ask` | Ask a question about a previously uploaded document |
| `GET` | `/health` | Check basic application health |
| `GET` | `/ready` | Check application readiness |
| `GET` | `/app-info` | Retrieve application information |

### Upload a Document

**Endpoint:** `POST /documents`

Upload a PDF using a multipart form-data request.

Example successful response:

```json
{
  "document_id": "c4fac98e-ed5e-4b05-af0c-27fafea369e8",
  "filename": "spring_boot.pdf",
  "message": "Document uploaded and processed successfully."
}
```

The returned `document_id` is used to query the uploaded document.

### Ask a Question

**Endpoint:** `POST /documents/{document_id}/ask`

Example request body:

```json
{
  "question": "What are the main benefits of Spring Boot?"
}
```

Example response (abbreviated):

```json
{
  "document_id": "c4fac98e-ed5e-4b05-af0c-27fafea369e8",
  "question": "What are the main benefits of Spring Boot?",
  "answer": "Spring Boot accelerates application setup, reduces boilerplate configuration, and provides production-ready monitoring.",
  "sources": [
    {
      "page": "1",
      "text": "Spring Boot accelerates application setup...",
      "score": 0.6579689383506775
    }
  ],
  "retrieved_chunks": 3
}
```

The response includes the generated answer and supporting document passages with page references and retrieval scores.

### Application Monitoring

- `GET /health` — Basic health check.
- `GET /ready` — Application readiness check.
- `GET /app-info` — Application information.

Refer to Swagger UI for the complete request and response schemas.

## Installation and Setup

### Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Docker and Docker Compose (recommended)
- A Groq API key for cloud-based inference, or Ollama for local inference

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-document-intelligence.git
cd ai-document-intelligence
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username.

### 2. Configure Environment Variables

Create a `.env` file in the project root using the following configuration:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_COLLECTION=multi_document_test
CHROMA_PERSIST_DIRECTORY=./chroma_multi_db
```

Never commit `.env` files or real API keys to version control.

For local Ollama inference, configure:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3
OLLAMA_BASE_URL=http://localhost:11434
```

Make sure Ollama is installed, running, and has the configured model available.

### 3. Run Locally

Install project dependencies:

```bash
uv sync
```

Start the FastAPI server:

```bash
uv run uvicorn ai_document_intelligence.main:app --host 0.0.0.0 --port 8000
```

Access the interactive API documentation:

`http://localhost:8000/docs`

### 4. Run with Docker Compose

Docker Compose is the recommended deployment method for this project.

Ensure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
```

Check the running containers:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f api
```

Open Swagger UI:

`http://localhost:8000/docs`

The Docker Compose configuration uses named volumes for:

- ChromaDB persistent vector storage
- Hugging Face model cache

These volumes preserve data across container recreation.

### 5. Stop the Application

```bash
docker compose down
```

This stops and removes the application containers while preserving named volumes.

**Warning:** Running `docker compose down -v` also removes the project's named volumes, including persisted ChromaDB data.

## Testing and Evaluation

The project includes automated testing and RAG evaluation to assess application correctness, retrieval quality, and generated answers.

### Automated Testing

The application uses **pytest** to test backend functionality, including document processing, application services, LLM provider configuration, and dependency injection.

Run the test suite:

```bash
uv run pytest
```

**Latest verified result:** 26 tests passing.

### RAG Evaluation

The project uses **DeepEval** to evaluate the quality of Retrieval-Augmented Generation (RAG) responses.

The evaluation framework includes:

| Metric | Purpose |
|---|---|
| Answer Relevancy | Measures how relevant the generated answer is to the user's question |
| Faithfulness | Evaluates whether the answer is supported by the retrieved context |
| Contextual Precision | Evaluates whether relevant retrieved context is ranked appropriately |
| Contextual Recall | Evaluates whether the retrieved context contains information needed to answer the question |

### Evaluation Workflow

1. Prepare a dataset of questions and expected answers.
2. Retrieve relevant document chunks using ChromaDB.
3. Generate answers using the configured LLM.
4. Construct DeepEval test cases containing the question, generated answer, retrieved context, and expected answer where required.
5. Run evaluation metrics and inspect scores and explanations.

The evaluation setup supports experiments using Groq and Ollama models.

Evaluation scores help identify retrieval and generation weaknesses, but do not guarantee that every generated answer is correct or hallucination-free.