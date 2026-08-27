# 📊 ConsultIQ: Enterprise RAG & Financial BI Assistant

A privacy-first, 100% local Retrieval-Augmented Generation (RAG) platform that combines conversational document intelligence with deterministic financial KPI analytics.

---

## ⚡ Key Features
* **Zero Data Leakage:** Runs entirely on-premise using local LLMs (**Llama 3.2** via Ollama) and embeddings (**all-MiniLM-L6-v2**).
* **Auditable Document Q&A:** Grounded semantic search with exact page numbers and text chunk citations.
* **Automated Financial Extraction:** Uses JSON-constrained prompting to dynamically detect arbitrary reporting periods (YoY/QoQ).
* **Deterministic BI Engine:** Offloads growth rates and margin calculations to **Pandas** to eliminate LLM math hallucinations.
* **Visual Dashboard & Export:** Interactive Streamlit trend charts and 1-click CSV export.

---

## 🏗️ Architecture

$$\text{PDF/TXT} \longrightarrow \text{PyPDF} \longrightarrow \text{Recursive Chunker} \longrightarrow \text{MiniLM Embeddings} \longrightarrow \text{FAISS Index}$$

$$\text{FAISS Index} \longrightarrow \begin{cases} \text{Conversational QA} \longrightarrow \text{Llama 3.2} \longrightarrow \text{Streamlit Chat (with Citations)} \\ \text{Targeted Retrieval} \longrightarrow \text{JSON Extraction} \longrightarrow \text{Pandas KPIs/Charts} \longrightarrow \text{CSV Export} \end{cases}$$

---

## 📂 Project Structure
* `app.py` – Streamlit UI, chat interface, and dashboard.
* `rag_engine.py` – LCEL Q&A chain and structured JSON financial extraction.
* `bianalysis.py` – Pandas-powered KPI, margin, and trend logic.
* `vector_store.py` – HuggingFace embeddings and FAISS index management.
* `text_chunker.py` – Recursive character text splitter.
* `document_loader.py` – PDF and TXT ingestion.

---

## 🛠️ Tech Stack
* **LLM & Embeddings:** Ollama (`llama3.2`), HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Framework & Storage:** LangChain, FAISS
* **Data & Frontend:** Pandas, Streamlit, PyPDF

---

## 🚀 Quickstart

1. **Pull Model:**
   ```bash
   ollama pull llama3.2
                                                    
