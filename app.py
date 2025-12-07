import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Learning Hub", layout="wide")

# --- CSS STYLING (Safe Injection) ---
custom_css = """
<style>
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px !important;
        font-weight: bold;
        border-radius: 12px;
        background-color: #f0f2f6;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        border-color: #4CAF50;
        color: #4CAF50;
    }
    .tool-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        height: 200px;
    }
    .tool-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    .tool-desc {
        font-size: 0.95em;
        color: #666;
        margin-bottom: 15px;
        line-height: 1.4;
    }
    .report-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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

# --- HELPER: FIND WORKING MODEL ---
def get_working_model_name(key):
    try:
        genai.configure(api_key=key)
        # Try Flash first
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name: return m.name
        # Fallback to Pro
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-pro' in m.name: return m.name
    except:
        pass
    return "gemini-pro"

# --- SYSTEM PROMPTS ---
def get_system_prompt(mode):
    if mode == "Analogy Connector":
        return (
            "You are a schema-building expert.\n"
            "RULES:\n"
            "1. Take a key concept from the context and create a clear analogy.\n"
            "2. Present ONLY the analogy first.\n"
            "3. Ask a follow-up question to check comprehension.\n"
            "4. Only move on after confirmation."
        )
    elif mode == "Vocabulary Coach":
        return (
            "You are a dedicated vocabulary coach.\n"
            "RULES:\n"
            "1. Ask the user for words to review (or offer to pick).\n"
            "2. Define the word clearly, then ask the user to use it in a NEW sentence.\n"
            "3. If they struggle, provide an example context."
        )
    elif mode == "Reassessment Practice":
        return (
            "You are an adaptive tutor focused on remediation.\n"
            "RULES:\n"
            "1. Ask if the user has specific weak areas.\n"
            "2. Create targeted quiz questions (One at a time).\n"
            "3. TRACK PERFORMANCE SILENTLY.\n"
            "4. Do NOT give a score until requested.\n"
            "5. When grading, ONLY grade the answer to the specific question you just asked."
        )
    else:
        return "You are a helpful AI assistant."

# --- HELPER: START INTERACTION ---
def start_automated_interaction(mode_name, initial_instruction):
    st.session_state.mode = mode_name
    
    if st.session_state.vector_store is None:
        st.warning("Please wait for material to load.")
        return

    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "user", "content": f"**[Starting Tool: {mode_name}]**"})

    with st.spinner(f"Initializing {mode_name}..."):
        try:
            docs = st.session_state.vector_store.similarity_search("Summary", k=3)
            context_text = "\n".join([doc.page_content for doc in docs])
            
            full_prompt = (
                f"System: {get_system_prompt(mode_name)}\n"
                f"Context: {context_text}\n"
                f"Instruction: {initial_instruction}"
            )
            
            valid_model = get_working_model_name(api_key)
            llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
            response = llm.invoke(full_prompt)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            st.rerun()
        except Exception as e:
            st.error(f"Error starting interaction: {e}")

# --- HELPER: PDF PROCESSING ---
def process_pdf(files):
    raw_text = ""
    for pdf in files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.session_state.vector_store = vectorstore

# --- SIDEBAR ---
with st.sidebar:
    st.header("Teacher Dashboard")
    
    if not has_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    preloaded_path = "lesson.pdf"
    is_preloaded = os.path.exists(preloaded_path)

    if is_preloaded:
        st.success(f"📚 Material Loaded: '{preloaded_path}'")
        if st.session_state.vector_store is None and api_key:
            with st.spinner("Analyzing text..."):
                try:
                    with open(preloaded_path, "rb") as f:
                        process_pdf([f])
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    else:
        st.write("No preloaded lesson found.")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)
        if st.button("Process Material"):
            if not api_key:
                st.error("Missing API Key.")
            elif not pdf_docs:
                st.error("Please upload a PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        process_pdf(pdf_docs)
                        st.success("Ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- MAIN APP ---
st.title("🎓 Student Learning Hub")

if st.session_state.vector_store is None:
    st.info("👋 Welcome! Please wait for the lesson material to load.")
else:
    # 3-COLUMN TOOL LAYOUT
    col1, col2, col3 = st.columns(3)

    # Tool A: Analogy
    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">🔗 Real-World Connect</div>
            <div class="tool-desc">
                Connect new ideas to things you already know using analogies.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Connector"):
            start_automated_interaction("Analogy Connector", "Find a complex concept in this text and give me an analogy for it.")

    # Tool B: Vocabulary
    with col2:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">🗣️ Vocabulary Coach</div>
            <div class="tool-desc">
                Identify hard words. Define them, use them in sentences, and get feedback.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Vocab"):
            start_automated_interaction("Vocabulary Coach", "Ask me what words I want to review, or offer to pick some for me.")

    # Tool C: Reassessment
    with col3:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">📈 Practice & Drill</div>
            <div class="tool-desc">
                Drill weak spots and get a Readiness Score (1-10) based on performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Drill"):
            start_automated_interaction("Reassessment Practice", "Ask me if I have specific weak areas I want to practice.")

    st.markdown("---")

    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if user_input := st.chat_input("Type your response here..."):
        if not api_key:
            st.error("No API Key found.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                    context_text = "\n".join([doc.page_content for doc in docs])
                    
                    persona = get_system_prompt(st.session_state.mode)
                    
                    full_prompt = (
                        f"System: {persona}\n"
                        f"Context: {context_text}\n"
                        f"User Input: {user_input}"
                    )
                    
                    valid_model = get_working_model_name(api_key)
                    llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
                    response = llm.invoke(full_prompt)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Reporting (Only visible if history exists)
    if len(st.session_state.chat_history) > 2:
        st.markdown("---")
        if st.button("📊 Generate Insight Report"):
            with st.spinner("Analyzing your session..."):
                try:
                    valid_model = get_working_model_name(api_key)
                    llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
                    
                    if st.session_state.mode == "Reassessment Practice":
                         report_prompt = (
                            "Calculate the student's Readiness Score (1-10).\n"
                            "Summarize correct vs incorrect answers.\n"
                            "List Strengths and Weaknesses.\n"
                            f"Chat History: {st.session_state.chat_history}"
                        )
                    else:
                        report_prompt = (
                            "Analyze learning progress.\n"
                            "1. What did they understand well?\n"
                            "2. What concepts need review?\n"
                            "3. Suggested next steps.\n"
                            f"Chat History: {st.session_state.chat_history}"
                        )

                    report = llm.invoke(report_prompt)
                    
                    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
                    st.subheader("Your Progress Report")
                    st.write(report.content)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
