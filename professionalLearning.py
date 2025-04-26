# professionalLearning.py

import os
import warnings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

warnings.filterwarnings("ignore")

# Load API key and set model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o"

llm = ChatOpenAI(temperature=0.5, model=llm_model, streaming=True)

# Per-user memory dictionary
user_memories = {}

def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            input_key="userResponse"
        )
    return user_memories[user_id]

def clear_user_memory(user_id: str):
    if user_id in user_memories:
        user_memories[user_id].clear()

# --------------------- PROMPT ----------------------

pro_prompt = PromptTemplate(
    input_variables=["userResponse", "chat_history"],
    template="""
You are a professional AI tutor specializing in technical, academic, and advanced subject areas.
Communicate clearly and respectfully, and adapt to the user's level of knowledge.

Format your responses properly:
- Use **Markdown** for headings, lists, and emphasis.
- Use **LaTeX** for math expressions inside `$$...$$`.
- Use triple backticks for **code blocks**, labeled with the language (e.g., ```python).

Never say you're just an AI model. Stay confident and helpful.
If the user provides ambiguous or incomplete information, help them clarify.

Conversation so far:
{chat_history}

User: {userResponse}
AI:
"""
)

# Chain using the pro-level prompt
response_chain = LLMChain(llm=llm, prompt=pro_prompt)

# --------------------- EXPORTS ----------------------

__all__ = [
    "llm",
    "response_chain",
    "get_user_memory",
    "clear_user_memory",
]
