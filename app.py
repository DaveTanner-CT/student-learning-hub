import streamlit as st
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

# --- HELPER: FIND WORKING MODEL ---
def get_working_model_name(key):
    """Asks Google which models are actually available to avoid 404s"""
    try:
        genai.configure(api_key=key)
        # Look for the new Flash model first
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name:
                    return m.name
        # Fallback to Pro
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-pro' in m.name:
                    return m.name
    except:
        pass
    # Absolute fallback if search fails
    return "gemini-1.5-flash"

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
                    
                    # Embeddings usually prefer the 'models/' prefix
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    st.session_state.vector_store = vectorstore
                    st.success("Gemini is ready!")
                except Exception as e:
                    st.error(f"Error processing: {e}")

# --- SYSTEM PROMPTS ---
def get_system_prompt(mode):
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
                        f"User: {user_input}"
                    )
                    
                    # AUTO-DETECT VALID MODEL
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
            with st.spinner("Analyzing..."):
                try:
                    valid_model = get_working_model_name(api_key)
                    llm = ChatGoogleGenerativeAI(model=valid_model, google_api_key=api_key)
                    report_prompt = f"Analyze this chat history for student growth: {st.session_state.chat_history}"
                    report = llm.invoke(report_prompt)
                    
                    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
                    st.write(report.content)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
