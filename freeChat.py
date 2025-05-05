'''
*************************************************************
* Name:    Elijah Campbell‑Ihim
* Project: AI Tutor Python API
* Class:   CMPS-450 Senior Project
* Date:    May 2025
* File:    freeChat.py
*************************************************************
'''

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

# Per-user memory dictionary
user_memories = {}

#Gets specific user's memory
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", input_key="userResponse"
        )
    return user_memories[user_id]


# Clears specific user's memory
def clear_user_memory(user_id: str):
    if user_id in user_memories:
        user_memories[user_id].clear()


# Prompt for free chat
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


# Export components for FastAPI
__all__ = [
    "llm",
    "chat_prompt",
    "get_user_memory",
    "clear_user_memory"
]
