import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile
import os
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta
import shutil

load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stChatMessage p {
        font-size: 18px !important;
    }
    .stChatMessage {
        font-size: 18px !important;
    }
    .stTextInput > div > div > input {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Small talk check
def is_small_talk(text):
    return bool(re.search(r"\b(hello|hi|hey|how are you|what's up|good morning|good evening)\b", text, re.I))

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Time-based DB expiry
DB_PATH = "chroma_db"
DB_EXPIRY_MINUTES = 20

if "db_created_time" in st.session_state:
    if datetime.now() - st.session_state.db_created_time > timedelta(minutes=DB_EXPIRY_MINUTES):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        st.session_state.pop("db_created_time", None)
        st.session_state.pop("chain", None)
        st.warning("⚠️ The previous PDF session has expired after 15 minutes. Please re-upload your PDF.")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📂 Upload PDF")
    st.caption("ℹ️ Uploaded PDFs are stored temporarily and will be deleted after 20 minutes.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        st.info("🔍 Creating embeddings... (just once per upload)")
        db = Chroma.from_documents(
            docs,
            GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
            persist_directory=DB_PATH
        )

        retriever = db.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question using only the context provided.
        Be clear, concise, and use markdown formatting. Avoid making things up.

        <context>
            {context}
        </context>
        Question: {input}
        """)

        llm = GoogleGenerativeAI(model="gemini-2.0-flash-001")
        doc_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.chain = create_retrieval_chain(retriever, doc_chain)
        st.session_state.db_created_time = datetime.now()

# Chat area
st.title("📄 Chat with Your PDF")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if uploaded_file and "chain" in st.session_state:
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if is_small_talk(user_input):
                    answer = "👋 Hello! I'm here to help you understand your PDF. Feel free to ask any questions about it."
                else:
                    response = st.session_state.chain.invoke({"input": user_input})
                    answer = response['answer']

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("👈 Please upload a PDF from the sidebar to begin.")
