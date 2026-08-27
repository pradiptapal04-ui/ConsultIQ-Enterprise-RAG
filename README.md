# 📊 ConsultIQ: Enterprise Document Intelligence & Financial Analysis Platform

ConsultIQ is an end-to-end, privacy-first Retrieval-Augmented Generation (RAG) and Business Intelligence assistant. Built using local open-source LLMs and dense vector search, ConsultIQ automates document due diligence, ad-hoc inquiry answering with page citations, and chronological financial KPI analysis without cloud data leakage.

--

## 🏗️ Architecture & Pipeline Flow

--
               ┌────────────────────────────────────────────────────────┐
               │              Enterprise PDF / TXT Document             │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                  [ Document Loader ]
                                           │
                           [ Recursive Semantic Chunker ]
                                (800 chars / 150 overlap)
                                           │
                         [ HuggingFace Embeddings (MiniLM) ]
                                           │
                              [ FAISS Vector Index ]
                                           │
                      ┌────────────────────┴────────────────────┐
                      │                                         │
                      ▼                                         ▼
            [ Conversational RAG ]                   [ Dynamic BI Extractor ]
           • Similarity Search (k=3)                • Targeted Financial Retrieval
           • Zero-shot Llama 3.2 via Ollama         • Schema-Constrained JSON Extraction
           • Deterministic Page Citations           • Dynamic Period Detection (YoY/QoQ)
                      │                                         │
                      ▼                                         ▼
            [ Streamlit Q&A UI ]                     [ Pandas Analytical Engine ]
                                                    • Margin & Growth Calculations
                                                    • Performance Trend Visualization
                                                    • Automated Strategic Interpretation



✨ Key Capabilities
1) 100% Local & Privacy-Preserving: Powered by local quantized models (llama3.2 via Ollama) and local embeddings (all-MiniLM-L6-v2), ensuring strict zero data leakage for confidential enterprise records.

2) Conversational Due-Diligence (RAG): Answers natural-language queries grounded strictly in retrieved context, complete with page numbers and source chunk citations to eliminate hallucinations.

3) Dynamic Financial Parsing: Uses structured JSON prompts to dynamically detect arbitrary reporting periods (e.g., FY22, FY23, FY24) without rigid hardcoding.

4) Deterministic Financial KPI Engine: Offloads growth rate and profit margin arithmetic to Pandas, avoiding the calculation errors common in standard LLMs.

5) Interactive Business Intelligence Dashboard: Provides interactive trend charts (Revenue, Net Profit) and automated strategic commentary (operating leverage, margin compression, cost restructuring).

6) Export Ready: 1-click export of structured financial datasets to CSV for downstream reporting.


🛠️ Tech Stack
Frontend: Streamlit

Orchestration & RAG: LangChain, LangChain Community, LangChain Ollama

LLM Runtime: Ollama (llama3.2)

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Vector Store: FAISS (Facebook AI Similarity Search)

Data Processing & Analysis: Pandas, PyPDF          


CONSULTIQ1/
│
├── document_loader.py     # Ingests and parses PDF/TXT documents
├── text_chunker.py        # Splits text into semantically preserved overlapping chunks
├── vector_store.py        # Manages HuggingFace embeddings and local FAISS vector indexing
├── rag_engine.py          # LCEL RAG chains & structured financial extraction pipelines
├── bianalysis.py          # Pandas engine for period-over-period growth & margin calculations
├── app.py                 # Streamlit UI & interactive analytics dashboard
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules for virtual environments & indexes
└── README.md              # Project documentation




                                                    
