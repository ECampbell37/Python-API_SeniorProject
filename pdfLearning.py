'''
*************************************************************
* Name:    Elijah Campbell‑Ihim
* Project: AI Tutor Python API
* Class:   CMPS-450 Senior Project
* Date:    May 2025
* File:    pdfLearning.py
*************************************************************
'''



################################################################################################
# pdfLearning.py – Manages PDF-based learning mode in AI Tutor.
#
# This module allows users to upload a PDF document, processes its content into a vector store,
# and creates a ConversationalRetrievalChain that lets the AI answer questions based on the file.
#
# Features:
# - Parses uploaded PDF files using PyMuPDF
# - Splits text into chunks for embedding
# - Uses OpenAI embeddings and FAISS vector store
# - Tracks conversation history per user with memory
#
# Exports:
# - handle_pdf_upload        -> Process and store PDF content for retrieval
# - handle_pdf_question      -> Ask questions against the uploaded PDF
# - get_user_pdf_chain       -> Retrieve user's active PDF chain
# - clear_user_pdf_chain     -> Clear/reset a user's uploaded PDF chain
################################################################################################



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

# Dictionary to store each user's conversational retrieval chain
user_pdf_chains = {}



#####################################################################
# Retrieves the user's active ConversationalRetrievalChain instance.
# Raises an error if no PDF has been uploaded yet.
#####################################################################
def get_user_pdf_chain(user_id: str):
    if user_id not in user_pdf_chains:
        raise ValueError("No uploaded PDF for this user.")
    return user_pdf_chains[user_id]



#####################################################################
# Clears the stored PDF chain for a user, useful on logout/reset.
#####################################################################
def clear_user_pdf_chain(user_id: str):
    if user_id in user_pdf_chains:
        del user_pdf_chains[user_id]



#####################################################################
# Handles a new PDF file upload:
# - Saves the file temporarily
# - Loads and splits the document into chunks
# - Embeds the content using OpenAI embeddings
# - Stores a retriever-backed conversation chain for the user
#####################################################################
def handle_pdf_upload(contents: bytes, user_id: str):
    
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name


    # Load and parse the PDF
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()


    # Split the text into manageable chunks
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



#####################################################################
# Handles a user's question by invoking their active PDF chain.
# Returns the AI's answer from the PDF-based retriever.
#####################################################################
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
