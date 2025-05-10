'''
*************************************************************
* Name:    Elijah Campbell‑Ihim
* Project: AI Tutor Python API
* Class:   CMPS-450 Senior Project
* Date:    May 2025
* File:    freeChat.py
*************************************************************
'''



################################################################################################
# freeChat.py – Defines the open-ended conversation logic for AI Tutor's Free Chat Mode.
#
# This module sets up a flexible, user-friendly GPT-4o-mini LLM for simple, unstructured
# dialogue with memory support. Users can chat about anything and receive meaningful,
# engaging responses designed to educate and entertain.
#
# Exports:
# - A conversation prompt template for unstructured chat
# - Utility functions to manage per-user memory using LangChain's `ConversationSummaryMemory`
# - The language model instance used in Free Chat
################################################################################################



import os
import warnings

# Langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

warnings.filterwarnings('ignore')

# Load API key from env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the GPT Model
llm_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.7, model=llm_model)


# In-memory dictionary for storing user-specific memory
user_memories = {}



#####################################################################
# Retrieves or creates conversation memory tied to a specific user.
# Returns a ConversationSummaryMemory object used for dialogue recall.
#####################################################################
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", input_key="userResponse"
        )
    return user_memories[user_id]



#####################################################################
# Clears the existing memory for the specified user.
#####################################################################
def clear_user_memory(user_id: str):
    if user_id in user_memories:
        user_memories[user_id].clear()



# --------------------- PROMPT TEMPLATE ----------------------


# Prompt for open-ended conversation mode
chat_prompt = PromptTemplate(
    input_variables=["userResponse", "chat_history"],
    template="""You are a master of conversation with a wide range of knowledge \
and interests. Your goal is to facilitate the user's interests and have a great conversation! \
Make the user feel like they are having a real worthwhile interaction, and whenever you can, \
help them learn something new and interesting!

Previous Conversation:
{chat_history}

Please respond to the user:

{userResponse}"""
)



# --------------------- EXPORTS ----------------------


__all__ = [
    "llm",
    "chat_prompt",
    "get_user_memory",
    "clear_user_memory"
]
