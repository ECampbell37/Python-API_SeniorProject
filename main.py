'''
*************************************************************
* Name:    Elijah Campbell‑Ihim
* Project: AI Tutor Python API
* Class:   CMPS-450 Senior Project
* Date:    May 2025
* File:    main.py
*************************************************************
'''



###############################################################################################
# main.py – Entry point for the FastAPI backend that powers the AI Tutor web application.
#
# This file registers all routes for each learning mode, including:
# - Casual Learning
# - Kids Mode
# - Professional Mode
# - Free Chat
# - PDF Mode
#
# It also handles:
# - CORS middleware configuration
# - In-memory tracking of per-user quiz state
# - Delegation to specialized modules for memory, prompts, and LLM logic
#
# Exports:
# - All REST endpoints for AI Tutor frontend
# - Per-user memory and quiz data handling
###############################################################################################


from fastapi import FastAPI, Request, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Import modules for each learning mode
import casualLearning
import freeChat
import kidsLearning
import professionalLearning
import pdfLearning


# Initialize FastAPI app
app = FastAPI()


#############################################
# In-memory quiz tracking (non-persistent)
#############################################

user_quizzes = {}
kids_user_quizzes = {}

def get_user_quiz(user_id: str):
    if user_id not in user_quizzes:
        user_quizzes[user_id] = {"quiz": "", "feedback": "", "grade": ""}
    return user_quizzes[user_id]

def get_kids_user_quiz(user_id: str):
    if user_id not in kids_user_quizzes:
        kids_user_quizzes[user_id] = {"quiz": "", "feedback": "", "grade": ""}
    return kids_user_quizzes[user_id]



#############################################
# CORS configuration for frontend compatibility
#############################################

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-tutor-senior-project.vercel.app"
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-User-Id"],
)



#############################################
# Health/Status check endpoint
#############################################

@app.get("/health")
async def health_check():
    return {"status": "ok"}




#############################################
# Casual Learning Endpoints
#############################################


@app.get("/intro")
async def get_intro(subject: str = "Astronomy", x_user_id: str = Header(...)):
    """
    Initialize casual-learning memory and generate an introductory message.

    Args:
        subject (str): Topic to introduce. Defaults to "Astronomy".
        x_user_id (str): Header-based user id.

    Returns:
        dict: {"message": intro_text} or {"error": str(e)}.
    """
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        intro_text = casualLearning.intro_chain.run({"subject": subject})
        memory.save_context({"userResponse": ""}, {"chat_history": intro_text})
        return {"message": intro_text}
    except Exception as e:
        return {"error": str(e)}




@app.post("/chat")
async def post_chat(request: Request, subject: str = "Astronomy", x_user_id: str = Header(...)):
    """
    Continue a casual-learning conversation.

    Expects JSON:
        {"message": "<user input>"}

    Returns:
        dict: {"message": response_text} or {"error": str(e)}.
    """
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        response_chain = casualLearning.LLMChain(
            llm=casualLearning.llm,
            prompt=casualLearning.response_prompt,
            memory=memory
        )
        response_text = response_chain.run({
            "subject": subject,
            "userResponse": user_message
        })
        return {"message": response_text}
    except Exception as e:
        return {"error": str(e)}



@app.post("/memory/clear")
async def clear_memory(x_user_id: str = Header(...)):
    """
    Clear all casual-learning memory for the given user.

    Returns:
        dict: {"status": "Memory cleared"} or {"error": str(e)}.
    """
    try:
        casualLearning.clear_user_memory(x_user_id)
        return {"status": "Memory cleared"}
    except Exception as e:
        return {"error": str(e)}



@app.get("/quiz/start")
async def start_quiz(subject: str = "Astronomy", x_user_id: str = Header(...)):
    """
    Generate a 5-question quiz based on current memory.

    Returns:
        dict: {"quiz": "<quiz text>"} or {"error": str(e)}.
    """
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        quiz_data = get_user_quiz(x_user_id)
        quiz_data["quiz"] = casualLearning.quizGen_chain.run({
            "subject": subject,
            "previousChat": memory.chat_memory
        })
        return {"quiz": quiz_data["quiz"]}
    except Exception as e:
        return {"error": str(e)}



@app.post("/quiz/submit")
async def submit_quiz(request: Request, subject: str = "Astronomy", x_user_id: str = Header(...)):
    """
    Grade a submitted 5-question quiz and provide feedback.

    Expects JSON:
        {"answers": ["A", "B", "C", "D", "E"]}

    Returns:
        dict: {"feedback": "<text>", "grade": "<text>"} or {"error": str(e)}.
    """
    data = await request.json()
    answers = data.get("answers", [])
    if not isinstance(answers, list) or len(answers) != 5:
        return {"error": "Expected 'answers' as a list of 5 answers"}
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        quiz_data = get_user_quiz(x_user_id)
        quiz_data["feedback"] = casualLearning.quizFeedback_chain.run({
            "subject": subject,
            "previousChat": memory.chat_memory,
            "generatedQuiz": quiz_data["quiz"],
            "userAnswers": answers
        })
        quiz_data["grade"] = casualLearning.quizGrade_chain.run({
            "subject": subject,
            "quizFeedback": quiz_data["feedback"]
        })
        return {
            "feedback": quiz_data["feedback"],
            "grade": quiz_data["grade"]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/continue")
async def continue_lesson(subject: str = "Astronomy", x_user_id: str = Header(...)):
    """
    Continue the lesson after quiz completion.

    Returns:
        dict: {"message": "<continuation>"} or {"error": str(e)}.
    """
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        quiz_data = get_user_quiz(x_user_id)
        continuation = casualLearning.continueIntro_chain.run({
            "subject": subject,
            "quizFeedback": quiz_data["feedback"],
            "quizGrade": quiz_data["grade"],
            "chat_history": memory.chat_memory
        })
        memory.save_context({"userResponse": ""}, {"chat_history": continuation})
        return {"message": continuation}
    except Exception as e:
        return {"error": str(e)}




#############################################
# Free Chat Endpoints
#############################################


@app.post("/free_chat")
async def post_free_chat(request: Request, x_user_id: str = Header(...)):
    """
    Engage in an open-ended free-form chat.

    Expects JSON:
        {"message": "<user input>"}

    Returns:
        dict: {"message": "<AI reply>"} or {"error": str(e)}.
    """
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        memory = freeChat.get_user_memory(x_user_id)
        chat_chain = freeChat.LLMChain(
            llm=freeChat.llm,
            prompt=freeChat.chat_prompt,
            memory=memory
        )
        chat_text = chat_chain.run({"userResponse": user_message})
        return {"message": chat_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/free_chat/memory/clear")
async def clear_free_chat_memory(x_user_id: str = Header(...)):
    """
    Clear free-chat memory for the given user.

    Returns:
        dict: {"status": "Free chat memory cleared"} or {"error": str(e)}.
    """
    try:
        freeChat.clear_user_memory(x_user_id)
        return {"status": "Free chat memory cleared"}
    except Exception as e:
        return {"error": str(e)}




#############################################
# Kids Mode Endpoints
#############################################


@app.get("/kids_intro")
async def kids_get_intro(subject: str = "Nature", x_user_id: str = Header(...)):
    """
    Initialize memory and generate kids-mode introduction.

    Args:
        subject (str): Topic to introduce. Defaults to "Nature".
        x_user_id (str): Header-based user id.

    Returns:
        dict: {"message": "<intro>"} or {"error": str(e)}.
    """
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        kids_intro_text = kidsLearning.kids_intro_chain.run({"subject": subject})
        memory.save_context({"userResponse": ""}, {"chat_history": kids_intro_text})
        return {"message": kids_intro_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/kids_chat")
async def kids_post_chat(request: Request, subject: str = "Nature", x_user_id: str = Header(...)):
    """
    Continue a kids-mode conversation.

    Expects JSON:
        {"message": "<user input>"}

    Returns:
        dict: {"message": "<AI reply>"} or {"error": str(e)}.
    """
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        response_chain = kidsLearning.LLMChain(
            llm=kidsLearning.llm,
            prompt=kidsLearning.kids_response_prompt,
            memory=memory
        )
        kids_response_text = response_chain.run({
            "subject": subject,
            "userResponse": user_message
        })
        return {"message": kids_response_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/kids_memory/clear")
async def clear_kids_memory(x_user_id: str = Header(...)):
    """
    Clear kids-mode memory for the given user.

    Returns:
        dict: {"status": "Kids memory cleared"} or {"error": str(e)}.
    """
    try:
        kidsLearning.clear_user_memory(x_user_id)
        return {"status": "Kids memory cleared"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/kids_quiz/start")
async def kids_start_quiz(subject: str = "Nature", x_user_id: str = Header(...)):
    """
    Generate a 5-question quiz in kids mode.

    Returns:
        dict: {"quiz": "<quiz text>"} or {"error": str(e)}.
    """
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        quiz_data = get_kids_user_quiz(x_user_id)
        quiz_data["quiz"] = kidsLearning.kids_quizGen_chain.run({
            "subject": subject,
            "previousChat": memory.chat_memory
        })
        return {"quiz": quiz_data["quiz"]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/kids_quiz/submit")
async def kids_submit_quiz(request: Request, subject: str = "Nature", x_user_id: str = Header(...)):
    """
    Grade a submitted 5-question kids-mode quiz and return feedback.

    Expects JSON:
        {"answers": ["A", "B", "C", "D", "E"]}

    Returns:
        dict: {"feedback": "<text>", "grade": "<text>"} or {"error": str(e)}.
    """
    data = await request.json()
    answers = data.get("answers", [])
    if not isinstance(answers, list) or len(answers) != 5:
        return {"error": "Expected 'answers' as a list of 5 answers"}
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        quiz_data = get_kids_user_quiz(x_user_id)
        quiz_data["feedback"] = kidsLearning.kids_quizFeedback_chain.run({
            "subject": subject,
            "previousChat": memory.chat_memory,
            "generatedQuiz": quiz_data["quiz"],
            "userAnswers": answers
        })
        quiz_data["grade"] = kidsLearning.kids_quizGrade_chain.run({
            "subject": subject,
            "quizFeedback": quiz_data["feedback"]
        })
        return {
            "feedback": quiz_data["feedback"],
            "grade": quiz_data["grade"]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/kids_continue")
async def kids_continue_lesson(subject: str = "Nature", x_user_id: str = Header(...)):
    """
    Continue the kids-mode lesson after quiz completion.

    Returns:
        dict: {"message": "<continuation>"} or {"error": str(e)}.
    """
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        quiz_data = get_kids_user_quiz(x_user_id)
        kids_continuation = kidsLearning.kids_continueIntro_chain.run({
            "subject": subject,
            "quizFeedback": quiz_data["feedback"],
            "quizGrade": quiz_data["grade"],
            "chat_history": memory.chat_memory
        })
        memory.save_context({"userResponse": ""}, {"chat_history": kids_continuation})
        return {"message": kids_continuation}
    except Exception as e:
        return {"error": str(e)}




#############################################
# Professional Mode Endpoints
#############################################


@app.post("/professional_chat")
async def post_professional_chat(request: Request, x_user_id: str = Header(...)):
    """
    Handle a professional-mode chat interaction.

    Expects JSON:
        {"message": "<user input>"}

    Returns:
        dict: {"message": "<AI reply>"} or {"error": str(e)}.
    """
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        memory = professionalLearning.get_user_memory(x_user_id)
        chat_chain = professionalLearning.LLMChain(
            llm=professionalLearning.response_chain.llm,
            prompt=professionalLearning.response_chain.prompt,
            memory=memory
        )
        response_text = chat_chain.run({
            "userResponse": user_message,
            "chat_history": memory.chat_memory
        })
        return {"message": response_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/professional_chat/memory/clear")
async def clear_pro_chat_memory(x_user_id: str = Header(...)):
    """
    Clear professional-mode memory for the given user.

    Returns:
        dict: {"status": "Pro chat memory cleared"} or {"error": str(e)}.
    """
    try:
        professionalLearning.clear_user_memory(x_user_id)
        return {"status": "Pro chat memory cleared"}
    except Exception as e:
        return {"error": str(e)}




#####################################
# PDF Mode Endpoints
#####################################


@app.post("/pdf/upload")
async def pdf_upload(file: UploadFile = File(...), x_user_id: str = Header(...)):
    """
    Upload and process a PDF for later question-answering.

    Args:
        file (UploadFile): PDF file.
        x_user_id (str): Header-based user id.

    Returns:
        dict: {"status": "PDF uploaded and processed successfully."}
              or {"error": str(e)}.
    """
    try:
        contents = await file.read()
        pdfLearning.handle_pdf_upload(contents, x_user_id)
        await file.close()
        return {"status": "PDF uploaded and processed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/pdf/ask")
async def pdf_ask_question(request: Request, x_user_id: str = Header(...)):
    """
    Ask a question about the uploaded PDF.

    Expects JSON:
        {"message": "<question>"}

    Returns:
        dict: {"message": "<answer>"} or {"error": str(e)}.
    """
    data = await request.json()
    question = data.get("message", "")
    if not question:
        return {"error": "Missing 'message'"}
    try:
        answer = pdfLearning.handle_pdf_question(question, x_user_id)
        return {"message": answer}
    except Exception as e:
        return {"error": str(e)}


@app.post("/pdf/memory/clear")
async def pdf_clear_memory(x_user_id: str = Header(...)):
    """
    Clear all PDF-related memory/chains for the given user.

    Returns:
        dict: {"status": "PDF memory cleared"} or {"error": str(e)}.
    """
    try:
        pdfLearning.clear_user_pdf_chain(x_user_id)
        return {"status": "PDF memory cleared"}
    except Exception as e:
        return {"error": str(e)}


