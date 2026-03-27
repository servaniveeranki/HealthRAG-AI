# 🏥 MedRAG — Medical Knowledge RAG System

A production-ready AI-powered medical question answering system built with **LangChain**, **LangGraph**, **FastAPI**, and **Streamlit**. MedRAG retrieves evidence from trusted medical databases in real time, generates structured answers, scores their accuracy, detects hallucinations, filters for medical safety, and helps users find nearby hospitals — all for free with no paid API keys required.

---

## 📋 Table of Contents

- [What it does](#what-it-does)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Web Sources](#web-sources)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Document Ingestion](#document-ingestion)
- [Environment Variables](#environment-variables)

---

## What it does

MedRAG answers any health question — symptoms, diagnoses, treatments, medications, lab values — by:

1. Searching a local **ChromaDB vector database** of medical documents you provide
2. In parallel, fetching live evidence from **PubMed**, **MedlinePlus**, **WHO**, **Europe PMC**, and **OpenFDA**
3. Generating a structured answer using **Groq's llama-3.3-70b-versatile** LLM
4. Scoring the answer accuracy (0–100%), detecting hallucinations, and filtering for medical safety
5. Returning clickable citations sorted by relevance score
6. Suggesting nearby hospitals based on the user's location using OpenStreetMap

---

## Features

### Core AI Features
- **Retrieval-Augmented Generation (RAG)** — answers grounded in real retrieved documents, never fabricated
- **Temporal ranking** — recent guidelines ranked higher using `score = 0.8 × similarity + 0.2 × recency`
- **Parallel web retrieval** — PubMed, MedlinePlus, WHO, Europe PMC, OpenFDA all fetched simultaneously
- **Hallucination detection** — LLM-as-judge verifies every claim against source documents
- **Accuracy scoring** — every answer scored 0–100% with support level (High / Medium / Low), enforced by hard rules not just LLM opinion
- **Medical safety filtering** — checks for dangerous advice with severity levels: none / low / medium / high
- **60% accuracy threshold** — answers below 60% accuracy are automatically flagged as unverified
- **General knowledge fallback** — if no documents found, LLM answers from medical knowledge with a clear disclaimer

### UI Features (Streamlit)
- **Live search progress** — step-by-step status panel showing each pipeline stage in real time
- **Dark professional theme** — full dark UI with gradient header, styled citation cards, accuracy bar
- **Citation cards sorted by accuracy** — highest relevance first, top result highlighted in gold, each with a working clickable link
- **Proof panel** — green box for verified claims, yellow box for claims needing verification
- **Persistent Q&A history** — every answer auto-saved to `qa_history.json`, searchable and reviewable
- **Nearby hospital finder** — detects medical questions, geocodes user location, searches OpenStreetMap for hospitals within 15km, shows distance, address, and Google Maps link
- **Sidebar toggle button** — always-visible ☰ button using iframe postMessage
- **Document upload** — ingest PDFs or text files directly from the UI

### Infrastructure
- **FastAPI backend** with 8 REST endpoints
- **ChromaDB vector store** with persistent local storage
- **Conversation memory** — last 20 messages per session stored in memory
- **Multi-format ingestion** — PDF, TXT, MD, Word documents, scanned images (OCR)
- **CLI ingest tool** — ingest files, folders, raw text, or watch a folder for new files

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI 0.111, Uvicorn, Pydantic v2 |
| **AI Orchestration** | LangChain 0.2.5, LangGraph 0.1.5 |
| **LLM** | Groq `llama-3.3-70b-versatile` (also supports OpenAI, Ollama) |
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` (local, no API key) |
| **Vector Database** | ChromaDB 0.5.3 |
| **Web Retrieval** | httpx + concurrent.futures (parallel) |
| **Document Processing** | pypdf, pdfplumber, python-docx, pytesseract (OCR) |
| **Frontend** | Streamlit + custom CSS dark theme |
| **Geocoding / Maps** | OpenStreetMap Nominatim + Overpass API (free) |
| **Logging** | structlog |
| **Config** | pydantic-settings + python-dotenv |

---

## Project Structure

```
MEDICALRAG/
│
├── main.py                        # FastAPI entry point, app startup
├── streamlit_app.py               # Streamlit frontend (full UI)
├── ingest.py                      # CLI document ingestion tool
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not committed)
├── .env.example                   # Example environment file
├── qa_history.json                # Auto-generated Q&A history (gitignore this)
│
├── app/
│   ├── config.py                  # Settings via pydantic-settings
│   │
│   ├── api/
│   │   └── routes.py              # All 8 FastAPI endpoints
│   │
│   ├── core/
│   │   ├── embeddings.py          # SentenceTransformers singleton + cosine similarity
│   │   ├── vector_store.py        # ChromaDB operations + temporal rerank
│   │   ├── document_processor.py  # PDF / OCR / text chunking
│   │   ├── llm.py                 # LLM abstraction (Groq / OpenAI / Ollama) + prompts
│   │   ├── memory.py              # Conversation session memory (last 20 msgs)
│   │   └── web_retrieval.py       # Parallel web fetch: PubMed, WHO, MedlinePlus, EuropePMC, FDA
│   │
│   ├── graph/
│   │   ├── nodes.py               # All 7 pipeline node functions
│   │   └── pipeline.py            # Pipeline executor + LangGraph stub
│   │
│   └── models/
│       └── schemas.py             # Pydantic request/response models
│
├── utils/
│   └── sample_data.py             # Seeds 5 sample medical documents
│
├── tests/
│   └── test_pipeline.py           # Pipeline unit tests
│
└── data/
    └── chroma_db/                 # ChromaDB persistent storage (auto-created)
```

---

## Pipeline

The full query pipeline runs 7 nodes in sequence. Each node receives the complete state, does its job, and merges results back.

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Node 1 — Query Embedding                               │
│  all-MiniLM-L6-v2 converts question → 384-dim vector   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 2 — Document Retrieval                            │
│  ChromaDB cosine similarity search → top-k docs        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 3 — Temporal Ranking                              │
│  score = 0.8 × similarity + 0.2 × recency              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 4 — Context Filtering + Web Enrichment            │
│  • Drop docs below 0.35 threshold                      │
│  • Parallel fetch: PubMed + MedlinePlus + WHO +        │
│    EuropePMC + OpenFDA (all simultaneously)            │
│  • If local confidence > 0.5 → add 2 web docs         │
│  • If weak/empty → add up to 6 web docs               │
│  • Each source has independent 18s timeout             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 5 — Answer Generation                             │
│  Groq llama-3.3-70b generates structured answer from   │
│  merged context. Falls back to general LLM knowledge   │
│  if no documents found (clearly labelled).             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 6 — Accuracy Check (Hallucination Detection)      │
│  LLM-as-judge scores answer vs source documents 0–1   │
│  ≥ 0.75 = High  |  ≥ 0.60 = Medium  |  < 0.60 = Low  │
│  Final confidence = 40% cosine + 60% LLM accuracy     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Node 7 — Safety Filter                                 │
│  Checks for dangerous advice (dosages, stop-meds, etc) │
│  Severity: none / low / medium / high                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
        Final response with answer + citations
        + accuracy score + safety badge + source links
```

---

## Web Sources

All sources are free with no API key required. Fetched in parallel.

| Source | API | Content |
|---|---|---|
| **PubMed** | NCBI E-utilities | Peer-reviewed abstracts from 35M+ papers |
| **MedlinePlus** | NIH Connect API | Consumer health topics (NIH) |
| **WHO** | WHO Hub Search API | Global health guidelines and factsheets |
| **Europe PMC** | EBI REST API | 40M+ open-access biomedical publications |
| **OpenFDA** | FDA Open API | Drug labels, indications, warnings (drug questions only) |
| **OpenStreetMap** | Nominatim + Overpass | Hospital locations (hospital finder feature) |

---

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a medical question — runs full 7-node pipeline |
| `POST` | `/ingest/file` | Upload a PDF, TXT, or image file to the knowledge base |
| `POST` | `/ingest/text` | Ingest raw text directly |
| `POST` | `/conversation/new` | Create a new conversation session |
| `GET` | `/conversation/{id}` | Get conversation history |
| `DELETE` | `/conversation/{id}` | Delete a conversation session |
| `GET` | `/health` | System health check + KB stats |
| `GET` | `/stats` | Detailed metrics (chunk count, sessions, config) |

Interactive API docs available at `http://localhost:8000/docs`

### Example query request

```json
POST /api/v1/query
{
  "query": "What are the symptoms of diabetes?",
  "conversation_id": "abc123",
  "top_k": 5,
  "include_temporal_ranking": true
}
```

### Example query response

```json
{
  "answer": "Diabetes symptoms include...",
  "citations": [
    {
      "source": "WHO Diabetes Guidelines 2024",
      "organization": "World Health Organization (WHO)",
      "relevance_score": 0.89,
      "excerpt": "Symptoms of diabetes mellitus include...",
      "source_url": "https://www.who.int/...",
      "document_date": "2024-01-15"
    }
  ],
  "confidence_score": 0.82,
  "accuracy_score": 0.90,
  "support_level": "high",
  "is_safe": true,
  "safety_severity": "none",
  "hallucination_detected": false,
  "web_fallback_used": false
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/medrag.git
cd medrag
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-groq streamlit pydantic-settings
```

### Step 4 — Set up your environment file

```bash
cp .env.example .env
```

Edit `.env` and add your LLM provider credentials (see [Environment Variables](#environment-variables)).

### Step 5 — Seed sample medical data (optional but recommended)

```bash
python utils/sample_data.py
```

---

## Running the Project

### Terminal 1 — Start the FastAPI backend

```bash
python main.py
```

Server starts at `http://localhost:8000`. Check `http://localhost:8000/docs` for the interactive API explorer.

### Terminal 2 — Start the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

UI opens at `http://localhost:8501`.

---

## Document Ingestion

Use `ingest.py` to add medical documents to the knowledge base.

```bash
# Ingest a single PDF
python ingest.py --file path/to/guideline.pdf

# Ingest all PDFs and text files in a folder
python ingest.py --folder path/to/docs/ --source-type "WHO Guidelines"

# Ingest raw text directly
python ingest.py --text "Metformin 500mg is first-line therapy..." \
                 --source "BNF" --title "Metformin Monograph"

# Watch a folder and auto-ingest new files as they appear
python ingest.py --watch path/to/docs/ --poll 15
```

Supported file types: `.pdf`, `.txt`, `.md`

Documents are automatically split into 500-character chunks with 80-character overlap, embedded, and stored in ChromaDB.

---

## Configuration

All settings are in `app/config.py` and loaded from `.env`.

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend: `groq`, `openai`, or `ollama` |
| `LLM_MODEL` | `mistral` | Model name (e.g. `llama-3.3-70b-versatile` for Groq) |
| `GROQ_API_KEY` | — | Your Groq API key (free at console.groq.com) |
| `OPENAI_API_KEY` | — | OpenAI API key (if using OpenAI) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `MIN_CONFIDENCE_THRESHOLD` | `0.35` | Minimum similarity to include a local doc |
| `RETRIEVAL_TOP_K` | `5` | Number of docs to retrieve from ChromaDB |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `ENABLE_SAFETY_FILTER` | `true` | Enable/disable medical safety checking |

---

## Environment Variables

### Option A — Groq (free, recommended)

Get a free API key at [console.groq.com](https://console.groq.com).

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

### Option B — Ollama (completely free, local)

Install from [ollama.com](https://ollama.com), then run `ollama pull mistral`.

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
```

### Option C — OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_key_here
LLM_MODEL=gpt-4o-mini
```

---

## Notes

- The `qa_history.json` file is created automatically and stores all Q&A history. Add it to `.gitignore`.
- The `data/chroma_db/` folder is the vector database. Add it to `.gitignore` or commit it to preserve your knowledge base.
- ChromaDB telemetry errors (`capture() takes 1 positional argument`) are harmless and can be ignored.
- HuggingFace symlink warnings on Windows are harmless.
- The WHO API occasionally returns empty responses. The system automatically falls back to other sources.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration framework
- [ChromaDB](https://www.trychroma.com) — open-source vector database
- [Groq](https://groq.com) — fast LLM inference (free tier)
- [PubMed / NCBI](https://pubmed.ncbi.nlm.nih.gov) — free medical research API
- [MedlinePlus](https://medlineplus.gov) — NIH consumer health information
- [OpenStreetMap](https://www.openstreetmap.org) — free map and geocoding data
- [SentenceTransformers](https://www.sbert.net) — local embedding models
