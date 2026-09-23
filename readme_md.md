# 🤖 Enterprise Knowledge Assistant (RAG Chatbot)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-assistant.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, secure, cloud-native **Retrieval-Augmented Generation (RAG)** platform designed to process, index, and query custom PDF documents in real time. Powered by **Qdrant Vector DB**, **LangChain**, **HuggingFace Embeddings**, and **Google Gemini 2.5 Flash**.

🌐 **Live Demo:** [https://rag-assistant.streamlit.app](https://rag-assistant.streamlit.app)

---

## 📐 System Architecture

```
                                 [ PDF Document Upload ]
                                            │
                                            ▼
                                   [ PyPDF Processing ]
                                            │
                                            ▼
                             [ Recursive Text Splitter ]
                             (Chunk Size: 500, Overlap: 50)
                                            │
                                            ▼
                               [ MiniLM-L6 Embeddings ]
                               (384-dimensional Vectors)
                                            │
                                            ▼
                            [ Qdrant Vector DB (:memory:) ]
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
             [ Text Input Query ]                          [ Voice Query Input ]
                     │                                             │
                     │                                             ▼
                     │                                  [ Gemini Voice-to-Text ]
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                                 [ Similarity Search ]
                                (Retrieve Top k=5 Chunks)
                                            │
                                            ▼
                                  [ Gemini 2.5 Flash ]
                                 (Temperature = 0.0)
                                            │
                                            ▼
                                 [ Grounded Response ]
                             + Source Chunk Attribution
```

---

## ✨ Enterprise Features

- 📄 **Dynamic PDF Ingestion & Indexing:** Instant parsing, chunking, and vector embedding of user-uploaded PDF documents directly into memory.
- 🎙️ **Multimodal Query Input:** Supports standard text input as well as native voice recording query processing powered by Google Gemini speech transcription.
- ⚡ **Sub-Millisecond Vector Retrieval:** In-memory vector similarity searches executed via Qdrant Vector DB using cosine distance.
- 🔍 **Source Attribution & Auditability:** Collapsible UI dropdowns showing exact source text chunks and original PDF page numbers for every answer generated.
- 📥 **Transcript Export:** Export complete chat histories with timestamps into formatted `.txt` transcripts with one click.
- 📊 **Document Analytics Dashboard:** Dynamic document metadata extraction displaying total page count and total vector chunks created upon upload.
- 💡 **Executive Quick Queries:** One-click pre-built quick prompt buttons (Executive Summary, Key Takeaways, Core Recommendations).

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend & UI** | `Streamlit` | Interactive web dashboard and user interface |
| **LLM Engine** | `Google Gemini 2.5 Flash` | Grounded answer synthesis & audio transcription |
| **RAG Orchestration** | `LangChain` (LCEL) | Chain execution, prompt management, and retrievers |
| **Vector Database** | `Qdrant Client` | In-memory vector store for cosine similarity indexing |
| **Embeddings** | `HuggingFace` (`all-MiniLM-L6-v2`) | 384-dimensional dense vector embeddings |
| **PDF Processing** | `PyPDFLoader` / `pypdf` | Server-side document extraction |
| **Secret Management** | `python-dotenv` / `st.secrets` | Secure credential handling |

---

## 🔒 Security & Governance

- **Zero Data Retention (ZDR):** The application runs Qdrant in-memory (`:memory:`). When a session finishes or refreshes, all vector collections and documents are automatically purged from memory.
- **Credential Decoupling:** API keys are never hardcoded or committed to version control. The application natively reads from `.env` locally or encrypted platform secrets (`st.secrets`) in production.
- **Strict Grounding:** Gemini 2.5 Flash operates at temperature `0.0` with explicit prompt constraints to prevent hallucinations and strictly answer within document context.

---

## 📁 Project Structure

```text
rag-assistant/
│
├── .env                  # Local secret configuration (Git ignored)
├── .gitignore            # Git exclusion rules
├── app.py                # Main Streamlit application and RAG pipeline
├── README.md             # Project architecture and setup documentation
└── requirements.txt      # Python package dependencies
```

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Google Gemini API Key ([Get API Key](https://aistudio.google.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/Harsinisenthilkumar/rag-assistant.git
cd rag-assistant
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root folder:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Cloud Deployment Settings

When deploying to **Streamlit Community Cloud**:
1. Connect your GitHub repository (`your-username/rag-assistant`).
2. Set Main File Path to: `app.py`.
3. Under **Advanced Settings -> Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_api_key_here"
   ```
4. Click **Deploy!**

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.