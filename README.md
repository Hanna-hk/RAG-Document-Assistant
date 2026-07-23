# RAG Document Assistant

A Streamlit web application that lets you upload PDF documents and ask natural-language questions about their content. Answers are generated using **Retrieval-Augmented Generation (RAG)**: relevant chunks are retrieved via semantic search over vector embeddings, then passed as grounded context to an LLM, which answers strictly based on the retrieved content and cites its sources.

## Demo

<video src="https://github.com/user-attachments/assets/7254453c-7483-4cdc-b295-4175e724bb8c" controls width="700"></video>

## Features

- **PDF ingestion** — upload any PDF through the web UI; text is extracted, cleaned, and split into overlapping chunks to preserve context across boundaries.
- **Semantic search** — each chunk is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in a Supabase/pgvector database, enabling similarity-based retrieval instead of simple keyword matching.
- **Grounded answer generation** — retrieved chunks are passed to an LLM (Llama 3.3 70B via the Groq API), which is instructed to answer only from the given context and to flag when no answer is available.
- **Source attribution** — every answer is displayed alongside the source documents and similarity scores used to generate it, so answers can be verified against the original text.
- **Re-ingestion safe** — re-uploading a document with the same name replaces its previously stored chunks instead of duplicating them.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector storage & search | Supabase (pgvector) |
| LLM | Llama 3.3 70B via Groq API |
| Testing | pytest, pytest-mock |

## Project Structure

```
.
├── app.py                  # Streamlit UI: file upload + question answering
├── ingest.py               # PDF parsing, chunking, embedding, and storage
├── rag.py                  # Retrieval + LLM-based answer generation
├── utils.py                # Text chunking utility
├── tests/
│   ├── __init__.py
│   ├── test_ingest.py      # Unit tests for ingestion pipeline
│   ├── test_rag.py         # Unit tests for retrieval and answer generation
│   └── test_utils.py       # Unit tests for text chunking
├── requirements.txt
├── .env                    # Environment variables (not committed)
└── .gitignore
```

## How It Works

1. **Ingestion** (`ingest.py`): a PDF is parsed page by page, its text is cleaned and split into overlapping word chunks (`utils.py`), each chunk is embedded, and the resulting `(content, metadata, embedding)` rows are stored in Supabase.
2. **Retrieval** (`rag.py`): a user's query is embedded with the same model, and the most similar stored chunks are fetched via a Supabase RPC function (`match_documents`, backed by pgvector similarity search).
3. **Generation** (`rag.py`): the retrieved chunks are assembled into a context block and sent to the LLM along with a system prompt instructing it to answer strictly from that context and cite its sources.

## Setup

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com/) project with the `pgvector` extension enabled, a `documents` table (`content`, `metadata`, `embedding` columns), and a `match_documents` RPC function for similarity search.
- A [Groq API](https://groq.com/) key.

### Installation

```bash
git clone https://github.com/Hanna-hk/RAG-Document-Assistant.git
cd RAG-Document-Assistant
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-service-or-anon-key
GROQ_API_KEY=your-groq-api-key
```

### Running the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a PDF, and start asking questions.

## Testing

The core pipeline (chunking, ingestion, retrieval, and answer generation) is covered with unit tests, with all external services (Supabase, the embedding model, and the Groq API) mocked so the tests run offline and in isolation:

```bash
pytest
```

## Possible Improvements

- Batch embedding/insertion for very large documents to avoid request size limits.
- Support for additional file formats (DOCX, TXT).
- Configurable retrieval parameters (`match_count`, `threshold`) exposed through the UI.
- Handling of scanned/image-only PDFs via OCR.

## Author
Hanna Krechyk
