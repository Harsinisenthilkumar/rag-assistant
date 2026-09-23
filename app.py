import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from local .env file
load_dotenv()

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 Enterprise Knowledge Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by RAG Architecture, Qdrant Vector DB & Google Gemini Flash</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar Configuration & API Key Management
# ---------------------------------------------------------
# Fetch key from environment (.env) or Streamlit Secrets (Cloud Deployment)
google_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Optional fallback UI input if environment variable is missing
    if not google_api_key:
        google_api_key = st.text_input("Enter Gemini API Key", type="password")
        if google_api_key:
            st.success("API Key provided!")
    else:
        st.success("🔒 Gemini API Key loaded securely")

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    st.divider()
    
    st.header("📊 Document Insights")
    stats_placeholder = st.empty()
    stats_placeholder.info("Upload a PDF to view metadata insights.")
    
    st.divider()
    
    # Chat History Controls
    st.header("💬 Conversation Controls")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.messages:
        # Clear Chat History Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Download Chat History Transcript
        chat_transcript = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="📥 Export Chat Transcript",
            data=chat_transcript,
            file_name="rag_chat_transcript.txt",
            mime="text/plain",
            use_container_width=True
        )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to convert recorded voice bytes to text via Gemini Flash
def transcribe_audio(audio_file, api_key):
    client = genai.Client(api_key=api_key)
    audio_bytes = audio_file.read()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            "Transcribe the following user audio recording accurately into text. Return ONLY the transcribed text and nothing else.",
            genai.types.Part.from_bytes(
                data=audio_bytes,
                mime_type=audio_file.type or "audio/wav"
            )
        ]
    )
    return response.text.strip()

# ---------------------------------------------------------
# 3. RAG Ingestion Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_and_index_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    client = QdrantClient(":memory:")
    collection_name = "pdf_rag_collection"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    vector_store.add_documents(chunks)
    
    return vector_store, len(docs), len(chunks)

# ---------------------------------------------------------
# 4. Main Application Flow & Chat Interface
# ---------------------------------------------------------
if uploaded_file and google_api_key:
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Indexing PDF chunks into Qdrant Vector DB..."):
        vector_store, total_pages, total_chunks = process_and_index_pdf(temp_pdf_path)
    
    with stats_placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric("Total Pages", total_pages)
        col2.metric("Total Chunks", total_chunks)
        st.caption("✅ Vector Index Status: Active in Memory")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        google_api_key=google_api_key
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
    You are an expert enterprise business assistant. Provide a detailed, comprehensive, and well-structured answer to the user's question based strictly on the context provided below.

    Guidelines:
    - Explain the answer clearly using bullet points or structured paragraphs.
    - Be thorough and detailed. Do not provide vague responses unless specifically asked.
    - If context does not contain the answer, state "I couldn't find this information in the document."

    Context:
    {context}

    Question:
    {question}
    """)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Quick Action Buttons
    st.markdown("##### 💡 Suggested Questions")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    quick_query = None

    if btn_col1.button("📌 Executive Summary", use_container_width=True):
        quick_query = "Provide a high-level executive summary of this document."
    if btn_col2.button("🔑 Key Takeaways", use_container_width=True):
        quick_query = "List the top 5 key takeaways from this document."
    if btn_col3.button("❓ Core Recommendations", use_container_width=True):
        quick_query = "What are the primary recommendations or conclusions in this PDF?"

    st.divider()

    # RENDER ALL HISTORICAL CHAT MESSAGES
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # CHAT INPUT BAR WITH INTEGRATED MICROPHONE (RECORD BUTTON NEAR SEND)
    chat_response = st.chat_input("Ask a question or tap mic to speak...", accept_audio=True)
    
    user_query = None

    # Parse user input (Text or Mic Recording)
    if chat_response:
        # Scenario A: User typed plain text
        if isinstance(chat_response, str):
            user_query = chat_response
        # Scenario B: Dict return object from chat_input with audio
        elif hasattr(chat_response, "audio") and chat_response.audio:
            with st.spinner("Transcribing recorded voice via Gemini..."):
                user_query = transcribe_audio(chat_response.audio, google_api_key)
        elif hasattr(chat_response, "text") and chat_response.text:
            user_query = chat_response.text

    # Fallback to Quick Query buttons if clicked
    if not user_query and quick_query:
        user_query = quick_query

    # PROCESS NEW USER MESSAGE
    if user_query:
        # Append and display user message in chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display assistant message in chat
        with st.chat_message("assistant"):
            with st.spinner("Searching Qdrant DB & generating response..."):
                start_time = time.time()
                
                answer = rag_chain.invoke(user_query)
                retrieved_docs = retriever.invoke(user_query)
                
                elapsed_time = round(time.time() - start_time, 2)
                
                st.markdown(answer)
                st.caption(f"⚡ Response generated in **{elapsed_time}s** using Qdrant Vector DB & Gemini Flash")
                
                with st.expander("🔍 View Retrieved Chunks from Qdrant Vector DB"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 0) + 1}):**")
                        st.caption(doc.page_content)
                        st.divider()
                        
        # Save assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif not uploaded_file:
    st.info("👈 Please upload a PDF document in the sidebar to initialize the knowledge assistant.")
elif not google_api_key:
    st.warning("👈 Please enter or configure your Gemini API key.")