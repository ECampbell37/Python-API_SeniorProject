# Import Libraries
import os

# Langchain
from langchain.chains import LLMChain, SimpleSequentialChain, RouterChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

import warnings
warnings.filterwarnings("ignore")

# Load API key from env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#Define the GPT Model
llm_model = "gpt-3.5-turbo"

#Create the model instance
llm = ChatOpenAI(temperature=0.7, model=llm_model, streaming=True)

# Initialize model summary memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", input_key="userResponse")

# Define subject (to be determined in js app menu)
subject = None





# Define an introduction prompt
intro_prompt = PromptTemplate(
    input_variables = ["subject"], 
    template = """You are an excellent, helpful educator, specializing in {subject}. \
    It is your job to engage the user's interest with \
    an attention grabbing introduction of {subject}. \
    The introduction should welcome the user, give a brief overview of the topic, \
    and inspire the user to want to learn more. End the output with a clear,\
    engaging question or prompt for the user to faciliate the lesson. 
    
    
    Remember that you are the educator, and you should not ask the user to specify what they \
    want to learn, as they may not yet know. Instead, guide them in a particular direction, \
    or give them some options to choose from. 
    """
    )



# Define a tutoring prompt
response_prompt = PromptTemplate(
    input_variables = ["subject", "userResponse", "chat_history"], 
    template="""You are an excellent, helpful educator, specializing in {subject}. \
    The user is engaging with you on subject matter, and wants to learn more and explore \
    {subject}. It is your job to keep them engaged, encourage dialogue, and keep the conversation \
    moving in a positive direction. Education is awesome!  
    
    Remember that you are the educator, and you should not ask the user to specify what they \
    want to learn, as they may not yet know. Instead, guide them in a particular direction, \
    or engage with their ideas. Also, try not to be redundant!
    
    Whenever you don't know the answer to a question, you should admit \
    that you don't know.
    
    
    Previous Conversation:
    {chat_history}
    
    
    Please respond to the user:
    
    {userResponse}"""
    )


# Chains for into bot
intro_chain = LLMChain(llm=llm, prompt=intro_prompt)
response_chain = LLMChain(llm=llm, prompt=response_prompt, memory = memory)


# Function for intro interaction 
def introductoryBot():
    print("Welcome to Casual Learning with the AI Tutor! Type 'exit' to quit at any time. ")
    #topic = input("Enter the topic you'd like to learn about: ")

    print("\n--- Introduction ---\n")
    intro_text = intro_chain.run({"subject": subject})  # AI gives an introduction
    print("AI: " + intro_text)

    
    memory.save_context({"userResponse": ""}, {"chat_history": intro_text})
    
    while True:
        #Take in user chat response
        user_response = input("\nYou: ")
        
        
        #Exit if the user types 'exit'
        if user_response.lower() == 'exit':
            print("Thank you for learning with E.L.I.! Goodbye!")
            break

        # AI Responds
        response_text = response_chain.run({"subject": subject, "userResponse": user_response})  # AI continues the conversation
        print("\nAI: " + response_text)
        
        
        
# Define the prompt to generate a quiz
quizGen_prompt = PromptTemplate(
    input_variables = ["subject", "previousChat"], 
    template = """You are responsible for generating a quiz as part of a user's \
    learning experience. Generate 5 multiple choice questions to test the user's \
    knowledge in {subject}. Draw from specific information covered in the past \
    conversation. The goal is to test if the user is grasping the information well \
    and furthering their knowledge in {subject}. 
    
    Here is the previous conversation:
    {previousChat}
    """
    )




# Define the prompt to give feedback on the quiz results
quizFeedback_prompt = PromptTemplate(
    input_variables = ["subject", "previousChat", "generatedQuiz", "userAnswers"], 
    template="""Your job is to provide feedback to the user's {subject} quiz results. \
    Based on the user's answers, give some constructive feedback to their quiz results \
    and guide them on the path of learning. Make sure to output the question, the user's answer \
    (as a full answer choice if they only put the letter), the correct answer, and a helpful feedback explanation. In the feedback section, say something like \
    'Great Job!' if the user gets it right and 'Sorry that is incorrect' if they get it wrong. 
    
    Use these to help
    
    Generated Quiz: 
    {generatedQuiz}
    
    
    User's Response: 
    {userAnswers}
    
    
    Previous Coversation: 
    {previousChat}
    """
    )




# Define the prompt to grade the quiz
quizGrade_prompt = PromptTemplate(
    input_variables = ["subject", "quizFeedback"], 
    template="""Your job is to grade the user's answers to a generated {subject} quiz. \
    You should output which questions the user got correct, which they got wrong, \
    and their total score out of 5. Make sure the grade is consistent with the feedback results. \
    Whenever there is text like 'Good Job!' in the feedback section, the question is correct. \
    If the question is wrong, it will indicate that as well. 
    
    Here is an example: (\nCorrect: 1 3 5 \nIncorrect: 2 4 \nScore: 3/5 \nGrade: 60%) \
    
    
    Here is the quiz feedback:
    {quizFeedback}
    
    
    """
    )


# Chains for quiz 
quizGen_chain = LLMChain(llm=llm, prompt=quizGen_prompt) # Generation
quizFeedback_chain = LLMChain(llm=llm, prompt=quizFeedback_prompt) # Feedback
quizGrade_chain = LLMChain(llm=llm, prompt=quizGrade_prompt) # Grade



# Function for quiz flow
def quizBot():
    print("\n*********Quiz Time!**********\n\n")

    try:
        # Generate Quiz
        quizGen_text = quizGen_chain.run({"subject": subject, "previousChat": memory.chat_memory})
        print(quizGen_text)

        # Take in user answers
        userAnswers = []
        print("\n\nEnter you answers: ")

        for i in range(5):
            answer = input(f"\nQuestion {i+1}: ")
            userAnswers.append(answer)



        # Provide Feedback
        print("\n\n=== Quiz Feedback ===")
        quizFeedback_text = quizFeedback_chain.run({"subject": subject, 
                                                    "previousChat": memory.chat_memory, 
                                                    "generatedQuiz": quizGen_text, 
                                                    "userAnswers": userAnswers})
        print("\n" + quizFeedback_text)



        # Grade user Answers
        print("\n\n=== Quiz Grade ===")
        quizGrade_text = quizGrade_chain.run({"subject": subject, 
                                              "quizFeedback": quizFeedback_text})
        print("\n" + quizGrade_text)


        print("\n****************************\n\n")


        #Return Quiz result as dictionary to be accessed by Continuation bot
        return ({"quizFeedback_text": quizFeedback_text, 
                 "quizGrade_text": quizGrade_text, 
                 "previousMemory": memory.chat_memory})
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 
    
    
# Define the continuation introduction prompt for adjustment after quiz
continueIntro_prompt = PromptTemplate(
    input_variables = ["subject", "quizFeedback", "quizGrade", "chat_history"], 
    template = """You are an excellent, helpful educator, specializing in {subject}. \
    The user has just completed a quiz and the results will be provided below. \
    Your job is to adjust the lesson for the user to accomodate for their quiz performance. \
    If they have performed well (above 75%), you should congratulate them and advance to a new {subject} topic. \
    If they did not perform well (below 75%), you should slow down the lesson and simplify your language to \
    make it easier for them to understand the material. 
    
    
    Remember that you are the educator, and you should not ask the user to specify what they \
    want to learn, as they may not yet know. Instead, guide them in a particular direction, \
    or give them some options to choose from. 
    
    
    Here is the quiz grade:
    {quizGrade}
    
    
    
    Here is the quiz feedback:
    {quizFeedback}
    
    
    
    Here is the previous conversation history:
    {chat_history}
    """
    )


# Chain for continuation
continueIntro_chain = LLMChain(llm=llm, prompt=continueIntro_prompt)



# Function for new interaction
def continuationBot(results):
    print("Welcome back!\n")
    #topic = input("Enter the topic you'd like to learn about: ")

    print("\n--- Introduction ---\n")
    continueIntro_text = continueIntro_chain.run({"subject": subject,
                                                 "quizFeedback": results.get("quizFeedback_text"),
                                                 "quizGrade": results.get("quizGrade_text"),
                                                 "chat_history": memory.chat_memory})  # AI gives an introduction
    print("AI: " + continueIntro_text)

    
    memory.save_context({"userResponse": ""}, {"chat_history": continueIntro_text})
    
    while True:
        #Take in user chat response
        user_response = input("\nYou: ")
        
        
        #Exit if the user types 'exit'
        if user_response.lower() == 'exit':
            print("Thank you for learning with E.L.I.! Goodbye!")
            break

        # AI Responds
        response_text = response_chain.run({"subject": subject, "userResponse": user_response})  # AI continues the conversation
        print("\nAI: " + response_text)
        
        
        
# Expose components for use in FastAPI
__all__ = [
    "memory",
    "intro_chain",
    "response_chain",
    "quizGen_chain",
    "quizFeedback_chain",
    "quizGrade_chain",
    "continueIntro_chain"
]
