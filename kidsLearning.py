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

# Define subject (to be determined in js app menu)
subject = None

# Define an introduction prompt
kids_intro_prompt = PromptTemplate(
    input_variables = ["subject"], 
    template = """You are an excellent, helpful elementary school educator, specializing in {subject}. \
    It is your job to engage the child's interest with \
    an attention grabbing introduction of {subject}. \
    The introduction should welcome the child, give a brief overview of the topic, \
    and inspire the child to want to learn more. End the output with a clear,\
    engaging question or prompt for the child to faciliate the lesson. 
    
    Note, you are speaking to a child, so make sure to use very simple language (no big words), stick to \
    simple concepts, and keep everything friendly towards a young audience. Be sure to be \
    enthusiastic and guide the child through learning. Make learning fun!
    
    
    Remember that you are the educator, and you should not ask the child to specify what they \
    want to learn, as they may not yet know. Instead, guide them in a particular direction, \
    or give them some options to choose from. 
    """
    )



# Define a tutoring prompt
kids_response_prompt = PromptTemplate(
    input_variables = ["subject", "userResponse", "chat_history"], 
    template="""You are an excellent, helpful elementary school educator, specializing in {subject}. \
    The child is engaging with you on subject matter, and wants to learn more and explore \
    {subject}. It is your job to keep them engaged, encourage dialogue, and keep the conversation \
    moving in a positive direction. Education is awesome!  
    
    Note, you are speaking to a child, so make sure to use very simple language, stick to \
    simple concepts, and keep everything friendly towards a young audience. Be sure to be \
    enthusiastic and guide the child through learning. Make learning fun!
    
    Remember that you are the educator, and you should not ask the child to specify what they \
    want to learn, as they may not yet know. Instead, guide them in a particular direction, \
    or engage with their ideas. Also, try not to be redundant!
    
    Whenever you don't know the answer to a question, you should admit \
    that you don't know.
    
    
    Previous Conversation:
    {chat_history}
    
    
    Please respond to the user:
    
    {userResponse}"""
    )


# Chains using shared memory
kids_intro_chain = LLMChain(llm=llm, prompt=kids_intro_prompt)
kids_response_chain = LLMChain(llm=llm, prompt=kids_response_prompt, memory = memory)


# Function for interaction control
def kids_introductoryBot():
    print("Welcome to Kid's Learning with the AI Tutor! Type 'exit' to quit at any time. ")

    print("\n--- Introduction ---\n")
    kids_intro_text = kids_intro_chain.run({"subject": subject})  # AI gives an introduction
    print("AI: " + kids_intro_text)

    
    memory.save_context({"userResponse": ""}, {"chat_history": kids_intro_text})
    
    while True:
        #Take in user chat response
        user_response = input("\nYou: ")
        
        
        #Exit if the user types 'exit'
        if user_response.lower() == 'exit':
            print("Thank you for learning with AI Tutor! Goodbye!")
            break

        # AI Responds
        kids_response_text = kids_response_chain.run({"subject": subject, "userResponse": user_response})  # AI continues the conversation
        print("\nAI: " + kids_response_text)
        
        
        
# Define the prompt to generate a quiz
kids_quizGen_prompt = PromptTemplate(
    input_variables = ["subject", "previousChat"], 
    template = """You are responsible for generating a quiz as part of a user's \
    learning experience. Generate 5 multiple choice questions to test the user's \
    knowledge in {subject}. Draw from specific information covered in the past \
    conversation. The goal is to test if the user is grasping the information well \
    and furthering their knowledge in {subject}. 
    
    Note, this should be at an elementary school level, and you are creating this quiz for a child, \
    so make sure to use very simple language (no big words), stick to simple concepts, and keep \
    everything friendly towards a young audience.
    
    Here is the previous conversation:
    {previousChat}
    """
    )




# Define the prompt to give feedback on the quiz results
kids_quizFeedback_prompt = PromptTemplate(
    input_variables = ["subject", "previousChat", "generatedQuiz", "userAnswers"], 
    template="""Your job is to provide feedback to the user's {subject} quiz results. \
    Based on the user's answers, give some constructive feedback to their quiz results \
    and guide them on the path of learning. Make sure to output the question, the user's answer \
    (as a full answer choice if they only put the letter), the correct answer, and a helpful feedback explanation. In the feedback section, say something like \
    'Great Job!' if the user gets it right and 'Sorry that is incorrect' if they get it wrong. Double check on every grade \
    and make sure everything is completely factually correct, as we do not want to confuse the kids. 
    
    Note, this should be at an elementary school level, and you are creating this feedback for a child, \
    so make sure to use very simple language (no big words), stick to simple concepts, and keep \
    everything friendly towards a young audience.
    
    
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
kids_quizGrade_prompt = PromptTemplate(
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
kids_quizGen_chain = LLMChain(llm=llm, prompt=kids_quizGen_prompt) # Generation
kids_quizFeedback_chain = LLMChain(llm=llm, prompt=kids_quizFeedback_prompt) # Feedback
kids_quizGrade_chain = LLMChain(llm=llm, prompt=kids_quizGrade_prompt) # Grade



# Function for quiz flow
def kids_quizBot():
    print("\n*********Quiz Time!**********\n\n")

    try:
        # Generate Quiz
        kids_quizGen_text = kids_quizGen_chain.run({"subject": subject, "previousChat": memory.chat_memory})
        print(kids_quizGen_text)

        # Take in user answers
        userAnswers = []
        print("\n\nEnter you answers: ")

        for i in range(5):
            answer = input(f"\nQuestion {i+1}: ")
            userAnswers.append(answer)



        # Provide Feedback
        print("\n\n=== Quiz Feedback ===")
        kids_quizFeedback_text = kids_quizFeedback_chain.run({"subject": subject, 
                                                    "previousChat": memory.chat_memory, 
                                                    "generatedQuiz": kids_quizGen_text, 
                                                    "userAnswers": userAnswers})
        print("\n" + kids_quizFeedback_text)



        # Grade user Answers
        print("\n\n=== Quiz Grade ===")
        kids_quizGrade_text = kids_quizGrade_chain.run({"subject": subject, 
                                              "quizFeedback": kids_quizFeedback_text})
        print("\n" + kids_quizGrade_text)


        print("\n****************************\n\n")


        #Return Quiz result as dictionary to be accessed by Continuation bot
        return ({"quizFeedback_text": kids_quizFeedback_text, 
                 "quizGrade_text": kids_quizGrade_text, 
                 "previousMemory": memory.chat_memory})
                 
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 
    
    
# Define an introduction prompt
kids_continueIntro_prompt = PromptTemplate(
    input_variables = ["subject", "quizFeedback", "quizGrade", "chat_history"], 
    template = """You are an excellent, helpful elementary school educator, specializing in {subject}. \
    The child has just completed a quiz and the results will be provided below. \
    Your job is to adjust the lesson for the child to accomodate for their quiz performance. \
    If they have performed well (above 75%), you should congratulate them and advance to a new {subject} topic. \
    If they did not perform well (below 75%), you should slow down the lesson and simplify your language to \
    reinforce the lesson and make it easier for them to understand the material. 
    
    Note, you are speaking to a child, so make sure to use very simple language (no big words), stick to \
    simple concepts, and keep everything friendly towards a young audience. Be sure to be \
    enthusiastic and guide the child through learning. Make learning fun!
    
    
    Remember that you are the educator, and you should not ask the child to specify what they \
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


# Chains
kids_continueIntro_chain = LLMChain(llm=llm, prompt=kids_continueIntro_prompt)


# Function for interaction control
def kids_continuationBot(results):
    print("Welcome back!\n")

    print("\n--- Introduction ---\n")
    kids_continueIntro_text = kids_continueIntro_chain.run({"subject": subject,
                                                 "quizFeedback": results.get("quizFeedback_text"),
                                                 "quizGrade": results.get("quizGrade_text"),
                                                 "chat_history": memory.chat_memory})  # AI gives an introduction
    print("AI: " + kids_continueIntro_text)

    
    memory.save_context({"userResponse": ""}, {"chat_history": kids_continueIntro_text})
    
    while True:
        #Take in user chat response
        user_response = input("\nYou: ")
        
        
        #Exit if the user types 'exit'
        if user_response.lower() == 'exit':
            print("Thank you for learning with AI Tutor! Goodbye!")
            break

        # AI Responds
        kids_response_text = kids_response_chain.run({"subject": subject, "userResponse": user_response})  # AI continues the conversation
        print("\nAI: " + kids_response_text)
        
        
        
# Expose components for use in FastAPI
__all__ = [
    "memory",
    "kids_intro_chain",
    "kids_response_chain",
    "kids_quizGen_chain",
    "kids_quizFeedback_chain",
    "kids_quizGrade_chain",
    "kids_continueIntro_chain"
]

