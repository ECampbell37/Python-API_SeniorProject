'''
*************************************************************
* Name:    Elijah Campbell‑Ihim
* Project: AI Tutor Python API
* Class:   CMPS-450 Senior Project
* Date:    May 2025
* File:    pdfLearning.py
*************************************************************
'''

import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI

# Load the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use GPT-4o-mini as llm model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.3, model=llm_model)

# In-memory dictionary to store each user's personalized PDF chain
user_pdf_chains = {}


# Retrieve an existing ConversationalRetrievalChain for a user
def get_user_pdf_chain(user_id: str):
    if user_id not in user_pdf_chains:
        raise ValueError("No uploaded PDF for this user.")
    return user_pdf_chains[user_id]


# Clear the stored PDF chain for a user (e.g., on logout or file reset)
def clear_user_pdf_chain(user_id: str):
    if user_id in user_pdf_chains:
        del user_pdf_chains[user_id]


# Handle the PDF file upload, process it into searchable chunks, and store the chain
def handle_pdf_upload(contents: bytes, user_id: str):
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Load and parse the PDF into documents
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    # Split the documents into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Convert chunks into vector embeddings and store them in FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Set up conversation memory to track chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create a conversational chain using the LLM and vectorstore retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    # Store the chain for this specific user
    user_pdf_chains[user_id] = chain


    # Delete the temporary PDF file
    os.remove(tmp_path)


# Handle a user's question about their uploaded PDF by invoking the chain
def handle_pdf_question(question: str, user_id: str):
    chain = get_user_pdf_chain(user_id)
    result = chain.invoke({"question": question})
    return result["answer"]


# Exported functions from this module
__all__ = [
    "handle_pdf_upload",
    "handle_pdf_question",
    "get_user_pdf_chain",
    "clear_user_pdf_chain",
]
