import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# UPDATED: Importing Google Gemini Libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="AI Learning Companion (Gemini)", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .report-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "General"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR: TEACHER/ADMIN PORTAL ---
with st.sidebar:
    st.header("Teacher/Admin Portal")
    st.write("Powered by **Google Gemini**")
    
    # 1. User enters Google API Key here
    api_key = st.text_input("Enter Google API Key", type="password")
    
    pdf_docs = st.file_uploader("Upload Course Material (PDF)", accept_multiple_files=True)
    
    if st.button("Process Material"):
        if not api_key:
            st.error("Please add your Google API Key first.")
        elif not pdf_docs:
            st.error("Please upload a PDF.")
        else:
            with st.spinner("Processing content..."):
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += text
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(raw_text)
                
                # 2. Uses Google Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                st.session_state.vector_store = vectorstore
                st.success("Material Loaded! Gemini is ready.")

# --- HELPER: SYSTEM PROMPTS ---
def get_system_prompt(mode):
    prompts = {
        "Explain": "You are an expert tutor. Explain the concept simply. End with a check for understanding question.",
        "QuizMe": "You are a diagnostic tool. Ask a question based on the text. If wrong, identify the misconception.",
        "FixMyWork": "You are a writing coach. Critique the student's input. Do NOT rewrite it. Point out strengths and areas for growth.",
        "SocraticDialogue": "You are Socrates. Never give the answer. Ask probing questions to guide the student.",
        "Vocabulary Builder": "You are a linguist. Use the word in a sentence related to the context. Ask the student to use it in a new sentence."
    }
    return prompts.get(mode, "You are a helpful AI assistant.")

# --- MAIN INTERFACE ---
st.title("🎓 Student Learning Hub (Gemini Edition)")

if st.session_state.vector_store is None:
    st.info("👋 Hello! Please ask your teacher to upload the lesson material in the sidebar to begin.")
else:
    # Button Row
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("📖 Explain"): st.session_state.mode = "Explain"
    if col2.button("❓ QuizMe"): st.session_state.mode = "QuizMe"
    if col3.button("🔧 FixMyWork"): st.session_state.mode = "FixMyWork"
    if col4.button("🤔 Socratic"): st.session_state.mode = "SocraticDialogue"
    if col5.button("🗣️ Vocab"): st.session_state.mode = "Vocabulary Builder"

    st.markdown(f"**Current Mode:
