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
llm_model = "gpt-4"
llm = ChatOpenAI(temperature=0.7, model=llm_model)

# Per-user memory dictionary
user_memories = {}

def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", input_key="userResponse"
        )
    return user_memories[user_id]

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

# Interactive terminal-only mode (optional for local dev)
def chatBot():
    print("Welcome to the open chat with the AI Tutor! Type 'exit' to quit at any time.")
    print("Start the conversation however you'd like!")

    # Just use a placeholder test user for demo mode
    memory = get_user_memory("test_user")
    chat_chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

    while True:
        user_response = input("\nYou: ")

        if user_response.lower() == 'exit':
            print("Thank you for learning with AI Tutor! Goodbye!")
            break

        chat_text = chat_chain.run({"userResponse": user_response})
        print("\nAI: " + chat_text)

# Expose components for FastAPI
__all__ = [
    "llm",
    "chat_prompt",
    "get_user_memory",
    "clear_user_memory"
]
