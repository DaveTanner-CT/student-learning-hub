import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Learning Companion", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-weight: bold;
        border-radius: 10px;
    }
    .report-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "General"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- API KEY MANAGEMENT ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    has_key = True
else:
    api_key = None
    has_key = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("Teacher/Admin Portal")
    st.write("Powered by **Google Gemini**")
    
    if not has_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    else:
        st.success("Authentication: ✅ Connected securely")
    
    pdf_docs = st.file_uploader("Upload Course Material (PDF)", accept_multiple_files=True)
    
    if st.button("Process Material"):
        if not api_key:
            st.error("Missing API Key.")
        elif not pdf_docs:
            st.error("Please upload a PDF.")
        else:
            with st.spinner("Processing content..."):
                try:
                    raw_text = ""
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                raw_text += text
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    text_chunks = text_splitter.split_text(raw_text)
                    
                    # Embedding Model
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    st.session_state.vector_store = vectorstore
                    st.success("Material Loaded! Gemini is ready.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- SYSTEM PROMPTS ---
def get_system_prompt(mode):
    prompts = {
        "Explain": "You are an expert tutor. Explain the concept simply. End with a check for understanding question.",
        "QuizMe": "You are a diagnostic tool. Ask a question based on the text. If wrong, identify the misconception.",
        "FixMyWork": "You are a writing coach. Critique the student's input. Do NOT rewrite it. Point out strengths and areas for growth.",
        "SocraticDialogue": "You are Socrates. Never give the answer. Ask probing questions to guide the student.",
        "Vocabulary Builder": "You are a linguist. Use the word in a sentence related to the context. Ask the student to use it in a new sentence."
    }
    return prompts.get(mode, "You are a helpful AI assistant.")

# --- MAIN APP ---
st.title("🎓 Student Learning Hub")

if st.session_state.vector_store is None:
    st.info("👋 Hello! Please ask your teacher to upload the lesson material in the sidebar to begin.")
else:
    # Button Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("📖 Explain"): st.session_state.mode = "Explain"
    if col2.button("❓ QuizMe"): st.session_state.mode = "QuizMe"
    if col3.button("🔧 FixMyWork"): st.session_state.mode = "FixMyWork"
    if col4.button("🤔 Socratic"): st.session_state.mode = "SocraticDialogue"
    if col5.button("🗣️ Vocab"): st.session_state.mode = "Vocabulary Builder"

    # Display Current Mode
    mode_display = f"**Current Mode: `{st.session_state.mode}`**"
    st.markdown(mode_display)

    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if user_input := st.chat_input("Type your response here..."):
        if not api_key:
            st.error("Authentication Error: No API Key found.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Gemini is thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                    context_text = "\n".join([doc.page_content for doc in docs])
                    persona = get_system_prompt(st.session_state.mode)
                    
                    full_prompt = f"System: {persona}\nContext: {context_text}\nUser: {user_input}"
                    
                    # UPDATED: Added "models/" prefix
                    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
                    response = llm.invoke(full_prompt)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Reporting Section
    st.markdown("---")
    if st.button("📊 Generate Insight Report"):
        if st.session_state.chat_history:
            with st.spinner("Analyzing..."):
                # UPDATED: Added "models/" prefix
                llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
                report_prompt = f"Analyze this chat history for student growth/misconceptions: {st.session_state.chat_history}"
                report = llm.invoke(report_prompt)
                
                st.markdown("<div class='report-box'>", unsafe_allow_html=True)
                st.write(report.content)
                st.markdown("</div>", unsafe_allow_html=True)
