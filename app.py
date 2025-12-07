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

# --- CSS STYLING (Larger Buttons & Cards) ---
st.markdown("""
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
        height: 200px; /* Fixed height for alignment */
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

# --- HELPER: FIND WORKING MODEL ---
def get_working_model_name(key):
    try:
        genai.configure(api_key=key)
        # Try Flash first (Fastest)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name: return m.name
        # Fallback to Pro (Most Stable)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-pro' in m.name: return m.name
    except:
        pass
    return "gemini-pro"

# --- SYSTEM PROMPTS (Your 3 New Modes) ---
def get_system_prompt(mode):
    if mode == "Analogy Connector":
        return (
            "You are a schema-building expert. Your goal is to connect new concepts to the user's existing knowledge.\n"
            "RULES:\n"
            "1. Take a key concept from the provided context and create a clear analogy or real-world connection for it.\n"
            "2. Present ONLY the analogy/connection first.\n"
            "3. Immediately ask a follow-up, open-ended question to check the user's comprehension of that specific analogy.\n"
            "4. Only move to a new concept/analogy after the user confirms understanding."
        )
    
    elif mode == "Vocabulary Coach":
        return (
            "You are a dedicated vocabulary coach. Your goal is to ensure mastery of challenging terms.\n"
            "RULES:\n"
            "1. Start by asking the user to input the specific vocabulary words they want to review (or offer to find difficult words in the text).\n"
            "2. For each word, define it clearly based on the context, then ask the user to use it in a NEW sentence.\n"
            "3. If the user struggles or asks for a hint, provide an analogy or an example of the word used in a new, distinct context.\n"
            "4. Grade their sentence gently and move to the next word."
        )
    
    elif mode == "Reassessment Practice":
        return (
            "You are an adaptive tutor focused on remediation. Your goal is to drill the user on specific weak points.\n"
            "RULES:\n"
            "1. Start by asking if the user wants to provide their known weak areas (concepts or questions they missed).\n"
            "2. Create targeted quiz questions (One at a time) ONLY on those weak areas. If no weak areas are given, quiz on general main topics.\n"
            "3. Mix in other concepts from the provided materials to check broad understanding.\n"
            "4. TRACK PERFORMANCE SILENTLY: Keep a running count of correct vs. incorrect answers in your head.\n"
            "5. Do NOT give a score until the user explicitly requests the final report.\n"
            "6. When the user asks for the report, provide:\n"
            "   - A summary of strengths and weaknesses.\n"
            "   - A Readiness Score (1-10 scale, where 10 is Expert).\n"
            "CRITICAL: When grading, ensure you are grading the answer to the QUESTION YOU JUST ASKED. Do not hallucinate other questions."
        )
    
    else:
        return "You are a helpful AI assistant."

# --- HELPER: START AUTOMATED INTERACTION ---
def start_automated_interaction(mode_name, initial_instruction):
    st.session_state.mode = mode_name
    
    if st.session_state.vector_store is None:
        st.warning("Please wait for the class material to load first.")
        return

    # Clear previous chat history to start fresh with new mode
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

# --- SIDEBAR & PRELOAD LOGIC ---
with st.sidebar:
    st.header("Teacher Dashboard")
    
    if not has_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    preloaded_file_path = "lesson.pdf"
    is_preloaded = os.path.exists(preloaded_file_path)

    if is_preloaded:
        st.success(f"📚 Material Loaded: '{preloaded_file_path}'")
        if st.session_state.vector_store is None and api_key:
            with st.spinner("Analyzing text..."):
                try:
                    with open(preloaded_file_path, "rb") as f:
                        process_pdf([f])
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading lesson file: {e}")
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
    # --- 3-COLUMN TOOL LAYOUT ---
    col1, col2, col3 = st.columns(3)

    # Tool A: Analogy
    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">🔗 Real-World Connect</div>
            <div class="tool-desc">
                Don't just memorize it—understand it. Connect new ideas to things you already know using analogies.
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
                Identify hard words and master them. You define them, use them in sentences, and get instant feedback.
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
                Focus on your weak spots. I'll drill you on what you missed and give you a Readiness Score (1-10).
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
                    # Retrieve context based on the CURRENT question/input
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
                    
                    # Custom prompt depending on the mode
                    if st.session_state.mode == "Reassessment Practice":
                         report_prompt = (
                            "Based on the chat history, calculate the student's Readiness Score (1-10).\n"
                            "Provide a summary of correct vs incorrect answers.\n"
                            "List Strengths and Weaknesses.\n"
                            f"Chat History: {st.session_state.chat_history}"
                        )
                    else:
                        report_prompt = (
                            "Analyze the student's learning progress.\n"
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

