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

# --- CSS STYLING ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-weight: bold;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    .report-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .instruction-text {
        font-size: 1.1em;
        color: #555;
        margin-bottom: 20px;
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
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name: return m.name
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-pro' in m.name: return m.name
    except:
        pass
    return "gemini-pro"

# --- SYSTEM PROMPTS ---
def get_system_prompt(mode):
    if mode == "Explain":
        return (
            "You are an expert tutor. Your goal is to ensure deep understanding.\n"
            "RULES:\n"
            "1. Explain the concept simply using the context.\n"
            "2. After explaining, do NOT just stop. Ask a specific 'Check for Understanding' question to verify they get it.\n"
            "3. If they answer correctly, move on to the next related concept."
        )
    
    elif mode == "QuizMe":
        return (
            "You are an adaptive Quiz Master.\n"
            "RULES:\n"
            "1. Ask a multiple-choice or open-ended question based on the text.\n"
            "2. If CORRECT: Confirm it, explain why briefly, and IMMEDIATELY ask the NEXT question.\n"
            "3. If INCORRECT: Identify the misconception, give a hint, and ask them to try again."
        )
    
    elif mode == "FixMyWork":
        return (
            "You are a writing coach.\n"
            "RULES:\n"
            "1. Ask the student to paste their paragraph.\n"
            "2. Identify 1 specific strength and 1 specific weakness.\n"
            "3. Ask them to REWRITE the weak sentence based on your feedback.\n"
            "4. If they rewrite it well, praise the improvement."
        )
    
    elif mode == "SocraticDialogue":
        return (
            "You are Socrates.\n"
            "RULES:\n"
            "1. Ask deep, open-ended questions about the text's themes.\n"
            "2. When the student answers, acknowledge their point, but then Challenge it with a follow-up question (e.g., 'But what if...').\n"
            "3. Never give the answer. Keep the chain of questioning going."
        )
    
    elif mode == "Vocabulary Builder":
        return (
            "You are a linguist.\n"
            "RULES:\n"
            "1. Select a complex word from the text and define it.\n"
            "2. Ask the student to write a NEW sentence using that word.\n"
            "3. If they use it correctly, say 'Perfect!' and give them a NEW word.\n"
            "4. If incorrect, correct their usage gently and ask for a retry."
        )
    
    else:
        return "You are a helpful AI assistant."

# --- HELPER: START AUTOMATED INTERACTION ---
def start_automated_interaction(mode_name, initial_instruction):
    st.session_state.mode = mode_name
    
    if st.session_state.vector_store is None:
        st.warning("Please wait for the class material to load first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": f"**[Mode Selected: {mode_name}]**"})

    with st.spinner(f"{mode_name} is starting..."):
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
    st.header("Classroom Portal")
    
    if not has_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    preloaded_file_path = "lesson.pdf"
    is_preloaded = os.path.exists(preloaded_file_path)

    if is_preloaded:
        st.success(f"📚 Class Material: '{preloaded_file_path}' Loaded")
        if st.session_state.vector_store is None and api_key:
            with st.spinner("Initializing Class Material..."):
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
    st.info("👋 Welcome! Waiting for class material to load...")
else:
    st.markdown("<div class='instruction-text'>Select a learning mode below to start automatically:</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if col1.button("📖 Explain"):
        start_automated_interaction("Explain", "Start by explaining the most important concept in this text, then check if I understand.")
    
    if col2.button("❓ QuizMe"):
        start_automated_interaction("QuizMe", "Ask me the first multiple-choice question about this text.")
        
    if col3.button("🔧 Fix"):
        start_automated_interaction("FixMyWork", "Introduce yourself as a coach and ask me to paste a paragraph for review.")
        
    if col4.button("🤔 Socratic"):
        start_automated_interaction("SocraticDialogue", "Ask me a thought-provoking question to start our discussion.")
        
    if col5.button("🗣️ Vocab"):
        start_automated_interaction("Vocabulary Builder", "Teach me one hard word from this text.")

    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input
    if user_input := st.chat_input("Type your answer here..."):
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
                        f"User Answer/Input: {user_input}"
                    )
                    
                    valid_model = get_working_model_name(api_key)
                    llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
                    response = llm.invoke(full_prompt)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.write(response.content)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Reporting Section
    st.markdown("---")
    if st.button("📊 Generate Insight Report"):
        if st.session_state.chat_history:
            with st.spinner("Generating your Personal Growth Report..."):
                try:
                    valid_model = get_working_model_name(api_key)
                    llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
                    
                    # UPDATED: Student-Facing Prompt
                    report_prompt = (
                        "You are a supportive, encouraging learning coach. "
                        "Analyze the chat history below and write a feedback report DIRECTLY TO THE STUDENT.\n"
                        "Do not use complex jargon. Be friendly and clear.\n\n"
                        "Structure your response with these three exact headers:\n"
                        "1. 🌟 What You Did Well\n"
                        "2. 💡 Concepts to Review\n"
                        "3. 🚀 Your Next Steps\n\n"
                        "Make the next steps specific based on the text they struggled with.\n"
                        f"Chat History: {st.session_state.chat_history}"
                    )
                    
                    report = llm.invoke(report_prompt)
                    
                    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
                    st.subheader("Your Learning Journey")
                    st.write(report.content)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
