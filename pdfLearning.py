# pdfLearning.py

import os
import tempfile
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.3, model=llm_model)

user_pdf_chains = {}

def get_user_pdf_chain(user_id: str):
    if user_id not in user_pdf_chains:
        raise ValueError("No uploaded PDF for this user.")
    return user_pdf_chains[user_id]

def clear_user_pdf_chain(user_id: str):
    if user_id in user_pdf_chains:
        del user_pdf_chains[user_id]

def handle_pdf_upload(contents: bytes, user_id: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    user_pdf_chains[user_id] = chain

    # Cleanup temp file
    os.remove(tmp_path)

def handle_pdf_question(question: str, user_id: str):
    chain = get_user_pdf_chain(user_id)
    result = chain.invoke({"question": question})
    return result["answer"]

__all__ = [
    "handle_pdf_upload",
    "handle_pdf_question",
    "get_user_pdf_chain",
    "clear_user_pdf_chain",
]
