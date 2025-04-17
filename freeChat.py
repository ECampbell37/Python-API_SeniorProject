# Import Libraries
import os

# Langchain
from langchain.chains import LLMChain, SimpleSequentialChain, RouterChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

# Load API key from env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Ignore Depreciation Warnings
import warnings
warnings.filterwarnings('ignore')

#Define the GPT Model
llm_model = "gpt-3.5-turbo"

#Create the model instance
llm = ChatOpenAI(temperature=0.7, model=llm_model)

# Initialize model summary memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", input_key="userResponse")


# Define a chat prompt
chat_prompt = PromptTemplate(
    input_variables = ["userResponse", "chat_history"], 
    template="""You are a master of conversation with a wide range of knowledge \
    and interests. Your goal is to facilitate the user's interests and have a great coversation! \
    Make the user feel like they are having a real worthwhile interaction, and whenever you can, \
    help them learn something new and interesting!
    
    
    Previous Conversation:
    {chat_history}
    
    
    Please respond to the user:
    
    {userResponse}"""
    )


#Create the chain
chat_chain = LLMChain(llm=llm, prompt=chat_prompt, memory = memory)


# Function for interaction control
def chatBot():
    print("Welcome to the open chat with the AI Tutor! Type 'exit' to quit at any time. ")

    print("Start the conversation however you'd like!")
    while True:
        #Take in user chat response
        user_response = input("\nYou: ")
        
        
        #Exit if the user types 'exit'
        if user_response.lower() == 'exit':
            print("Thank you for learning with AI Tutor! Goodbye!")
            break

        # AI Responds
        chat_text = chat_chain.run({"userResponse": user_response})  # AI continues the conversation
        print("\nAI: " + chat_text)
        
        

# Expose components for use in FastAPI
__all__ = [
    "memory",
    "chat_chain"
]