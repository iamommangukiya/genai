import os
from typing import Dict
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

DB_DIR = "new_chroma_db"

def create_vectorstore_from_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        split_docs,
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        persist_directory=DB_DIR,
    )
    vectorstore.persist()

def get_retrieval_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the context below. Be concise and factual.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    llm = GoogleGenerativeAI(model="gemini-2.0-flash-001")
    doc_chain = create_stuff_documents_chain(llm, prompt)

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    retriever = vectorstore.as_retriever()

    return create_retrieval_chain(retriever, doc_chain)
