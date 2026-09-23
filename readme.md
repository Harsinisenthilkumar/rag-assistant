# 🤖 Enterprise Knowledge Assistant

A secure, cloud-native Retrieval-Augmented Generation (RAG) platform designed to answer complex domain queries over custom PDF documents using Qdrant Vector DB and Google Gemini Flash.

🔗 **Live Application:** [https://harsinisenthilkumar-rag-assistant-app-6fkmgw.streamlit.app/](https://harsinisenthilkumar-rag-assistant-app-6fkmgw.streamlit.app/)
---

## 📄 Technical & Architecture Documentation

**From:** Harsini S.  
**Project:** Enterprise Knowledge Assistant (RAG Chatbot)  
**Status:** Deployed & Live  

---

### Executive Overview
The **Enterprise Knowledge Assistant** is a secure, cloud-native retrieval-augmented generation (RAG) platform designed to answer complex domain queries over custom PDF documents. The application ingests documents dynamically, builds vector representations in memory, and synthesizes answers using Google's Gemini Flash model with exact source attribution.

---

### Key Technical Architecture

[ PDF Document ] ──► [ PyPDF Processing ] ──► [ Recursive Text Splitter ]
│
[ Qdrant Vector DB ] ◄── [ MiniLM-L6 Embeddings ] ◄──────┘
│
├──► [ Context Retrieval (k=5) ] ──► [ Gemini 2.5 Flash ] ──► [ UI Output ]

1. **Document Processing & Chunking**
   * **Ingestion:** `PyPDFLoader` handles document parsing directly in memory.
   * **Chunking Strategy:** `RecursiveCharacterTextSplitter` with a chunk size of `500` characters and `50` character overlap to preserve localized semantic boundaries across page breaks.

2. **Vector Indexing & Embeddings**
   * **Embedding Model:** `all-MiniLM-L6-v2` via HuggingFace transformers, producing 384-dimensional dense vector embeddings.
   * **Vector Storage:** Qdrant running in **in-memory mode (`:memory:`)**. Delivers sub-millisecond similarity searches without requiring external database hosting or persistent storage overhead.
   * **Distance Metric:** Cosine similarity.

3. **LLM Orchestration & RAG Pipeline**
   * **Language Model:** Google Gemini Flash (`gemini-2.5-flash`) set to zero temperature (`0`) to enforce strictly deterministic, non-hallucinatory grounded answers.
   * **Retrieval Engine:** Fetches top $k=5$ most relevant text chunks per query.
   * **LangChain Chains:** Implemented using LangChain Expression Language (LCEL) for low-latency streaming and prompt parameter binding.

4. **User Capabilities & Features**
   * **Multimodal Input:** Native support for both text and voice queries using Gemini-powered speech-to-text transcription.
   * **Transparency & Auditability:** Collapsible UI expandable views for retrieved context chunks with exact page numbering.
   * **Export Tools:** Instant browser download of formatted `.txt` conversation transcripts (`[USER]` / `[ASSISTANT]`).
   * **Document Analytics:** Real-time extraction and display of dynamic metadata (total pages, total chunks processed).

---

### Security & Compliance Governance

* **Credential Decoupling:** API keys are never stored in source code or pushed to Git repositories. Local execution uses `.env` configuration files enforced via `.gitignore`.
* **Cloud Security:** Deployed on Streamlit Community Cloud using encrypted platform secrets management (`st.secrets`).
* **Zero Data Retention:** In-memory vector indices auto-purge completely when session contexts terminate, ensuring client documents never persist on server storage.

---

### Local Setup & Dependencies

```bash
# Clone Repository
git clone [https://github.com/Harsinisenthilkumar/rag-assistant.git](https://github.com/Harsinisenthilkumar/rag-assistant.git)
cd rag-assistant

# Install Requirements
pip install -r requirements.txt

# Run Application Locally
streamlit run app.py
