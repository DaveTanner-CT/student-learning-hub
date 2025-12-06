import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Learning Hub", layout="wide")

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
    st.header("Teacher Portal")
    st.write("Powered by **Google Gemini**")
    
    if not has_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    else:
        st.success("Authentication: Connected")
    
    pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)
    
    if st.button("Process Material"):
        if not api_key:
            st.error("Missing API Key.")
        elif not pdf_docs:
            st.error("Please upload a PDF.")
        else:
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += text
                
                # Split text safely
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                text_chunks = text_splitter.split_text(raw_text)
                
                # Create embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004", 
                    google_api_key=api_key
                )
                vectorstore = FAISS.from_texts(
                    texts=text_chunks, 
                    embedding=embeddings
                )
                
                st.session_state.vector_store = vectorstore
                st.success("Gemini is ready!")

# --- SYSTEM PROMPTS ---
def get_system_prompt(mode):
    # Short prompts to prevent copy-paste errors
    if mode == "Explain":
        return "You are an expert tutor. Explain simply. Check for understanding."
    elif mode == "QuizMe":
        return "You are a quiz tool. Ask a question based on text. Identify misconceptions if wrong."
    elif mode == "FixMyWork":
        return "You are a coach. Critique input. Do NOT rewrite. Point out strengths/weaknesses."
    elif mode == "SocraticDialogue":
        return "You are Socrates. Never give answers. Ask probing questions."
    elif mode == "Vocabulary Builder":
        return "You are a linguist. Use word in context. Ask student to use it."
    else:
        return "You are a helpful AI assistant."

# --- MAIN APP ---
st.title("🎓 Student Learning Hub")

if st.session_state.vector_store is None:
    st.info("👋 Please ask your teacher to upload the lesson PDF.")
else:
    # Button Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("📖 Explain"): st.session_state.mode = "Explain"
    if col2.button("❓ QuizMe"): st.session_state.mode = "QuizMe"
    if col3.button("🔧 Fix"): st.session_state.mode = "FixMyWork"
    if col4.button("🤔 Socratic"): st.session_state.mode = "SocraticDialogue"
    if col5.button("🗣️ Vocab"): st.session_state.mode = "Vocabulary Builder"

    # Display Current Mode
    st.markdown(f"**Current Mode: `{st.session_state.mode}`**")

    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if user_input := st.chat_input("Type response..."):
        if not api_key:
            st.error("No API Key found.")
        else:
            # 1. Show user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # 2. Generate AI response
            with st.spinner("Thinking..."):
                docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                context_text = "\n".join([doc.page_content for doc in docs])
                persona = get_system_prompt(st.session_state.mode)
                
                # Construct prompt safely
                full_prompt = (
                    f"System: {persona}\n"
                    f"Context: {context_text}\n"
                    f"User: {user_input}"
                )
                
                # Using the CLEAN name for the Flash model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key
                )
                response = llm.invoke(full_prompt)
                
                # 3. Show AI message
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                with st.chat_message("assistant"):
                    st.write(response.content)

    # Reporting Section
    st.markdown("---")
    if st.button("📊 Generate Insight Report"):
        if st.session_state.chat_history:
            with st.spinner("Analyzing..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=api_key
                )
                report_prompt = (
                    f"Analyze this chat history for student growth: "
                    f"{st.session_state.chat_history}"
                )
                report = llm.invoke(report_prompt)
                
                st.markdown("<div class='report-box'>", unsafe_allow_html=True)
                st.write(report.content)
                st.markdown("</div>", unsafe_allow_html=True)
