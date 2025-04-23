from fastapi import FastAPI, UploadFile, File
from langserve import add_routes
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import shutil
import uvicorn

# ==== Config ====
DB_DIR = "new_chroma_db"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==== FastAPI App ====
app = FastAPI(
    title="LangChain PDF Chat Server",
    version="1.0",
    description="Chat with your uploaded PDFs using LangChain and Google Generative AI"
)

# ==== Upload Endpoint ====
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        split_docs,
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        persist_directory=DB_DIR
    )
    vectorstore.persist()

    os.remove(file_path)
    return {"message": "PDF uploaded and vector store created."}

# ==== LangServe PDF Chat Route ====
def get_pdf_chat_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question using the context below.
    Be concise and base your response strictly on the context.
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

# Add LangServe route
add_routes(
    app,
    get_pdf_chat_chain(),
    path="/chat"
)

# ==== Start Server ====
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
