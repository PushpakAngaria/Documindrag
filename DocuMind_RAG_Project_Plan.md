# 📚 DocuMind RAG — Full Project Plan, Features & Improvements

> **Project Title:** DocuMind RAG: Context-Aware Document Assistant
> **Course:** Generative AI | B.Tech CSE (AI) — 4th Semester
> **Team:** Pushpak, Gaurish, Arpit, SriSaran

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Being Solved](#2-problem-being-solved)
3. [Tech Stack](#3-tech-stack)
4. [System Architecture](#4-system-architecture)
5. [Team Responsibilities](#5-team-responsibilities)
6. [Core Features (MVP)](#6-core-features-mvp)
7. [Implementation Phases](#7-implementation-phases)
8. [Detailed Improvement Suggestions](#8-detailed-improvement-suggestions)
9. [Folder Structure](#9-folder-structure)
10. [Dependencies and Setup](#10-dependencies-and-setup)

---

## 1. Project Overview

**DocuMind RAG** is a **privacy-first, Retrieval-Augmented Generation (RAG)** application that turns personal academic documents (PDFs, handwritten notes, lecture slides) into an **interactive, queryable AI knowledge base**.

Instead of relying on a general-purpose LLM's pre-trained knowledge (which can hallucinate), the system:
1. Ingests the user's own documents
2. Chunks and embeds them into a local vector database
3. Retrieves only the most relevant chunks when a question is asked
4. Forces the LLM to answer **strictly from those chunks** — eliminating hallucination

> Think of it as a **personalized, 24/7 digital tutor** built from your own notes.

---

## 2. Problem Being Solved

| Problem | Current Tools | DocuMind Solution |
|--------|--------------|-------------------|
| Keyword-only search | Standard PDF readers (Ctrl+F) | Semantic similarity search via embeddings |
| AI hallucination | General LLMs like ChatGPT | RAG forces answers from uploaded docs only |
| Paywalls and rate limits | ChatPDF, Adobe AI | 100% free, open-source, runs locally |
| Handwritten notes ignored | Most tools skip images | Tesseract OCR handles handwritten content |
| Privacy concerns | Cloud-based tools upload your data | Fully local embeddings, no data leaves your machine |

---

## 3. Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **UI / Frontend** | Streamlit | Stateful web interface, file upload, chat UI |
| **PDF Parsing** | PyPDF2 | Extract text from machine-readable PDFs |
| **OCR** | Tesseract OCR | Extract text from images and handwritten notes |
| **Pipeline** | LangChain | RAG chain construction, conversational memory |
| **Embeddings** | HuggingFace Sentence-Transformers | Local, private vector generation |
| **Vector DB** | ChromaDB / FAISS | Semantic storage and low-latency retrieval |
| **LLM Backend** | Groq API (Llama-3) | Fast, high-quality language model inference |
| **Language** | Python 3.10+ | Core language for all components |

---

## 4. System Architecture

```
User Uploads Document (PDF / Image / Handwritten)
        |
        v
 [Data Ingestion]  <-- PyPDF2 (typed) + Tesseract OCR (handwritten)
        |
        | Raw Text
        v
 [Text Chunking]  <-- LangChain RecursiveCharacterTextSplitter
        |
        | Chunks
        v
 [Embedding Model]  <-- HuggingFace sentence-transformers (runs locally)
        |
        | Vectors
        v
 [Vector Database]  <-- ChromaDB / FAISS (in-memory or persistent)
        ^
        | Top-K Relevant Chunks
 [User Query]  <-- Streamlit Chat Input
        |
        v
 [Semantic Retrieval]  <-- Similarity search in ChromaDB
        |
        | Context Chunks + Query
        v
 [LLM - Groq API Llama-3]  <-- Strict system prompt: Answer ONLY from context
        |
        | Answer
        v
    Streamlit Chat UI --> User Sees Final Answer
```

---

## 5. Team Responsibilities

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Pushpak Angaria** | Frontend and Integration | Streamlit UI, session state, file upload, UX |
| **Arpit Sharma** | Data Extraction Pipeline | PyPDF2, Tesseract OCR, text normalization |
| **Gaurish Sharma** | Embeddings and Vector Store | LangChain chunking, HuggingFace embeddings, ChromaDB/FAISS |
| **SriSaran** | LLM Inference and Prompting | Groq API integration, system prompt engineering, hallucination prevention |

---

## 6. Core Features (MVP)

### 6.1 Document Ingestion
- [ ] Upload single or multiple PDF files via Streamlit file uploader
- [ ] Extract text from typed/machine-readable PDFs using **PyPDF2**
- [ ] Extract text from image-based or handwritten pages using **Tesseract OCR**
- [ ] Normalize and clean extracted text (remove noise, fix encoding)

### 6.2 Text Processing
- [ ] Split documents into overlapping chunks (e.g., 500 tokens, 50 overlap) using **LangChain**
- [ ] Attach metadata to each chunk (source file name, page number)
- [ ] Generate dense vector embeddings using **HuggingFace sentence-transformers**

### 6.3 Vector Storage
- [ ] Store embeddings in **ChromaDB** (lightweight, in-memory or persistent)
- [ ] Support fast **FAISS** as an alternative for large document sets
- [ ] Persist vector store across sessions (reload without re-embedding)

### 6.4 Question Answering
- [ ] Accept natural language questions via Streamlit chat input
- [ ] Perform semantic similarity search to retrieve Top-K relevant chunks
- [ ] Inject retrieved chunks as context into the LLM prompt
- [ ] Strict system prompt: LLM answers **only from retrieved context**
- [ ] Display the final answer in a conversational chat format

### 6.5 Chat Memory
- [ ] Maintain conversation history using **LangChain ConversationBufferMemory**
- [ ] Support follow-up questions with context from prior turns

### 6.6 Source Citation
- [ ] Show which page/chunk the answer was derived from
- [ ] Allow user to see the raw source snippet used by the LLM

---

## 7. Implementation Phases

### Phase 1 — Environment Setup
- [ ] Create Python virtual environment (`venv`)
- [ ] Install all dependencies (`requirements.txt`)
- [ ] Set up Groq API key (`.env` file)
- [ ] Verify Tesseract installation and PATH configuration

### Phase 2 — Data Pipeline
- [ ] Build `document_loader.py` — handles PDF upload and PyPDF2 parsing
- [ ] Build `ocr_processor.py` — Tesseract OCR for image/handwritten pages
- [ ] Build `text_chunker.py` — LangChain text splitting with metadata

### Phase 3 — Embeddings and Vector Store
- [ ] Build `embedder.py` — HuggingFace model loading and embedding generation
- [ ] Build `vector_store.py` — ChromaDB CRUD operations and persistence
- [ ] Test retrieval accuracy with sample queries

### Phase 4 — LLM Integration
- [ ] Build `llm_chain.py` — Groq API setup with Llama-3
- [ ] Engineer strict system prompt template
- [ ] Build `rag_pipeline.py` — ties retrieval and LLM together
- [ ] Test hallucination prevention with out-of-context questions

### Phase 5 — Frontend (Streamlit)
- [ ] Build `app.py` — main Streamlit application
- [ ] File uploader with progress bar
- [ ] Chat interface with message history
- [ ] Source snippet expander panel
- [ ] Session state management

### Phase 6 — Testing and Polish
- [ ] Test with diverse document types (typed PDF, scanned PDF, handwritten notes)
- [ ] Measure response latency and chunk retrieval accuracy
- [ ] Add error handling and user-friendly messages
- [ ] Write `README.md` with setup instructions

---

## 8. Detailed Improvement Suggestions

### 8.1 UI/UX Improvements

| # | Improvement | Details |
|---|------------|---------|
| 1 | Replace Streamlit with Next.js + FastAPI | Full control over design, animations, routing, and mobile responsiveness |
| 2 | Dark mode / Light mode toggle | Improves accessibility and user comfort for long study sessions |
| 3 | Drag-and-drop file upload | More intuitive than the default file picker |
| 4 | Progress bar during document processing | Show ingestion, chunking, and embedding steps visually |
| 5 | Document preview panel | Show the uploaded PDF as a side panel while chatting |
| 6 | Streaming token output | Typing animation for AI responses makes the chat feel alive |
| 7 | Mobile-responsive layout | Ensure the app works well on phones and tablets |
| 8 | Chat history export | Allow users to download Q&A sessions as PDF or text |
| 9 | Syntax highlighting for code answers | If the document contains code, format code blocks properly |
| 10 | Multi-language support | Allow querying documents in different languages |

---

### 8.2 Backend and Pipeline Improvements

| # | Improvement | Details |
|---|------------|---------|
| 1 | Hybrid Chunking Strategy | Use semantic chunking (split by paragraph/section headers) instead of fixed-size chunks |
| 2 | Chunk Overlap Tuning | Experiment with chunk sizes 256, 512, 1024 and overlaps for retrieval accuracy |
| 3 | Multi-file Cross-document Search | Upload multiple documents and query across all of them at once |
| 4 | Persistent Vector Store | Save ChromaDB to disk so re-upload is not needed on app restart |
| 5 | Document Versioning | Track and manage multiple versions of the same document |
| 6 | Auto-detect PDF type | Automatically check if PDF is text-based or image-based and route accordingly |
| 7 | Async Processing | Use asyncio or background tasks to process large files without blocking the UI |
| 8 | Embedding Cache | Cache embeddings for already-processed documents to avoid redundant computation |
| 9 | Support more file formats | Add support for DOCX, PPTX, TXT, and Markdown files |
| 10 | Image extraction | Extract and describe images/diagrams from PDFs using a vision model |

---

### 8.3 AI and RAG Quality Improvements

| # | Improvement | Details |
|---|------------|---------|
| 1 | Re-ranking Retrieved Chunks | Use a cross-encoder model like ms-marco-MiniLM to re-rank Top-K results |
| 2 | HyDE (Hypothetical Document Embeddings) | Generate a hypothetical answer first, embed it, then search for better retrieval |
| 3 | Query Decomposition | Break complex multi-part questions into simpler sub-questions |
| 4 | MMR (Maximal Marginal Relevance) | Retrieve diverse chunks instead of redundant ones for richer context |
| 5 | Confidence Scoring | Show a confidence score based on similarity scores of retrieved chunks |
| 6 | Fallback Response | If no relevant chunks found, reply "This is not in your documents" |
| 7 | Multi-LLM Support | Let users choose between Groq Llama-3, OpenAI GPT-4o, or Google Gemini |
| 8 | Summarization Mode | Add a "Summarize this document" button in addition to Q&A chat |
| 9 | Quiz Generation | Auto-generate MCQs from the uploaded document to help students test themselves |
| 10 | Answer with Page Numbers | Cite the exact page number from the document in every answer |
| 11 | Keyword Highlighting | Highlight the relevant keywords in the source chunk shown to the user |
| 12 | Contextual Compression | Compress retrieved chunks to remove irrelevant sentences before sending to LLM |

---

### 8.4 Privacy and Security Improvements

| # | Improvement | Details |
|---|------------|---------|
| 1 | 100% Offline Mode | Use Ollama with Mistral/Llama so no data leaves the device |
| 2 | API Key Security | Store keys in .env file, never hardcode in code |
| 3 | Per-user Session Isolation | Each user session gets its own vector store namespace |
| 4 | File Size Validation | Reject files over a configurable limit to prevent abuse |
| 5 | Input Sanitization | Sanitize user queries to prevent prompt injection attacks |
| 6 | Session Cleanup | Delete uploaded files and vectors when session ends |

---

### 8.5 Analytics and Monitoring

| # | Improvement | Details |
|---|------------|---------|
| 1 | Query Logging | Log all questions asked locally for analytics and debugging |
| 2 | Response Quality Feedback | Add thumbs up / thumbs down so users can rate answers |
| 3 | Retrieval Metrics Display | Show number of chunks retrieved, similarity scores, and response time |
| 4 | Usage Dashboard | Show stats like documents uploaded, questions asked, average response time |
| 5 | Error Tracking | Log errors with context for easier debugging |

---

### 8.6 Deployment and DevOps

| # | Improvement | Details |
|---|------------|---------|
| 1 | Docker Container | Package the entire app in a Docker image for one-command deployment |
| 2 | Deploy to Hugging Face Spaces | Free hosting with GPU support, perfect for project demos |
| 3 | Deploy to Streamlit Cloud | Easy free hosting for the Streamlit version |
| 4 | CI/CD Pipeline | GitHub Actions to auto-test and deploy on every push |
| 5 | Environment Config | Use .env.example to document all required environment variables |
| 6 | Health Check Endpoint | Add a /health endpoint to monitor app status in production |

---

## 9. Folder Structure

```
documind-rag/
|
|-- app.py                    # Main Streamlit application
|-- requirements.txt          # All Python dependencies
|-- .env                      # API keys (never commit this to git)
|-- .env.example              # Template for environment variables
|-- README.md                 # Setup and usage guide
|
|-- src/
|   |-- document_loader.py    # PDF loading with PyPDF2
|   |-- ocr_processor.py      # Tesseract OCR for images/handwriting
|   |-- text_chunker.py       # LangChain text splitting
|   |-- embedder.py           # HuggingFace embedding model
|   |-- vector_store.py       # ChromaDB / FAISS operations
|   |-- llm_chain.py          # Groq API + LangChain chain
|   `-- rag_pipeline.py       # Full RAG pipeline orchestration
|
|-- data/
|   |-- uploads/              # Temporarily stored uploaded files
|   `-- vector_db/            # Persistent ChromaDB storage
|
`-- tests/
    |-- test_loader.py
    |-- test_embedder.py
    `-- test_rag_pipeline.py
```

---

## 10. Dependencies and Setup

### requirements.txt
```
streamlit>=1.32.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-groq>=0.1.0
PyPDF2>=3.0.0
pytesseract>=0.3.10
Pillow>=10.0.0
sentence-transformers>=2.2.2
chromadb>=0.4.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
groq>=0.4.0
```

### .env Template
```
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

### Setup Commands
```bash
# 1. Clone the repo
git clone https://github.com/yourteam/documind-rag
cd documind-rag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# 5. Set up environment variables
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Run the app
streamlit run app.py
```

---

## Final Checklist

- [ ] Set up project structure and virtual environment
- [ ] Implement PDF text extraction (PyPDF2)
- [ ] Implement OCR for handwritten notes (Tesseract)
- [ ] Build text chunking pipeline (LangChain)
- [ ] Generate and store embeddings (HuggingFace + ChromaDB)
- [ ] Integrate Groq API with strict RAG prompt
- [ ] Build Streamlit chat interface
- [ ] Add source citation display
- [ ] Add persistent vector store
- [ ] Test with real academic documents
- [ ] Write README and submit

---

*Generated for DocuMind RAG — B.Tech CSE AI, 4th Semester, Generative AI Course*
