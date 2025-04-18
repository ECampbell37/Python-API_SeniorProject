# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import casualLearning
import freeChat
import kidsLearning

app = FastAPI()

# Global quiz state
last_quiz = ""
last_quiz_feedback = ""
last_quiz_grade = ""
kids_last_quiz = ""
kids_last_quiz_feedback = ""
kids_last_quiz_grade = ""

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ********* Casual Learning ***************

@app.get("/intro")
async def get_intro(subject: str = "Astronomy"):
    try:
        intro_text = casualLearning.intro_chain.run({"subject": subject})
        casualLearning.memory.save_context({"userResponse": ""}, {"chat_history": intro_text})
        return {"message": intro_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def post_chat(request: Request, subject: str = "Astronomy"):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        response_text = casualLearning.response_chain.run({
            "subject": subject,
            "userResponse": user_message
        })
        return {"message": response_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/quiz/start")
async def start_quiz(subject: str = "Astronomy"):
    global last_quiz
    try:
        last_quiz = casualLearning.quizGen_chain.run({
            "subject": subject,
            "previousChat": casualLearning.memory.chat_memory
        })
        return {"quiz": last_quiz}
    except Exception as e:
        return {"error": str(e)}

@app.post("/quiz/submit")
async def submit_quiz(request: Request, subject: str = "Astronomy"):
    global last_quiz_feedback, last_quiz_grade
    data = await request.json()
    answers = data.get("answers", [])
    if not isinstance(answers, list) or len(answers) != 5:
        return {"error": "Expected 'answers' as a list of 5 answers"}
    try:
        last_quiz_feedback = casualLearning.quizFeedback_chain.run({
            "subject": subject,
            "previousChat": casualLearning.memory.chat_memory,
            "generatedQuiz": last_quiz,
            "userAnswers": answers
        })
        last_quiz_grade = casualLearning.quizGrade_chain.run({
            "subject": subject,
            "quizFeedback": last_quiz_feedback
        })
        return {
            "feedback": last_quiz_feedback,
            "grade": last_quiz_grade
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/continue")
async def continue_lesson(subject: str = "Astronomy"):
    try:
        continuation = casualLearning.continueIntro_chain.run({
            "subject": subject,
            "quizFeedback": last_quiz_feedback,
            "quizGrade": last_quiz_grade,
            "chat_history": casualLearning.memory.chat_memory
        })
        casualLearning.memory.save_context({"userResponse": ""}, {"chat_history": continuation})
        return {"message": continuation}
    except Exception as e:
        return {"error": str(e)}


# ********* Free Chat ***************

@app.post("/free_chat")
async def post_free_chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        chat_text = freeChat.chat_chain.run({
            "userResponse": user_message
        })
        return {"message": chat_text}
    except Exception as e:
        return {"error": str(e)}




# ********** Kids Mode *************


@app.get("/kids_intro")
async def kids_get_intro(subject: str = "Nature"):
    try:
        kids_intro_text = kidsLearning.kids_intro_chain.run({"subject": subject})
        kidsLearning.memory.save_context({"userResponse": ""}, {"chat_history": kids_intro_text})
        return {"message": kids_intro_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/kids_chat")
async def kids_post_chat(request: Request, subject: str = "Nature"):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        kids_response_text = kidsLearning.kids_response_chain.run({
            "subject": subject,
            "userResponse": user_message
        })
        return {"message": kids_response_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/kids_quiz/start")
async def kids_start_quiz(subject: str = "Nature"):
    global kids_last_quiz
    try:
        kids_last_quiz = kidsLearning.kids_quizGen_chain.run({
            "subject": subject,
            "previousChat": kidsLearning.memory.chat_memory
        })
        return {"quiz": kids_last_quiz}
    except Exception as e:
        return {"error": str(e)}

@app.post("/kids_quiz/submit")
async def kids_submit_quiz(request: Request, subject: str = "Nature"):
    global kids_last_quiz_feedback, kids_last_quiz_grade
    data = await request.json()
    answers = data.get("answers", [])
    if not isinstance(answers, list) or len(answers) != 5:
        return {"error": "Expected 'answers' as a list of 5 answers"}
    try:
        kids_last_quiz_feedback = kidsLearning.kids_quizFeedback_chain.run({
            "subject": subject,
            "previousChat": kidsLearning.memory.chat_memory,
            "generatedQuiz": kids_last_quiz,
            "userAnswers": answers
        })
        kids_last_quiz_grade = kidsLearning.kids_quizGrade_chain.run({
            "subject": subject,
            "quizFeedback": kids_last_quiz_feedback
        })
        return {
            "feedback": kids_last_quiz_feedback,
            "grade": kids_last_quiz_grade
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/kids_continue")
async def kids_continue_lesson(subject: str = "Nature"):
    try:
        kids_continuation = kidsLearning.kids_continueIntro_chain.run({
            "subject": subject,
            "quizFeedback": kids_last_quiz_feedback,
            "quizGrade": kids_last_quiz_grade,
            "chat_history": kidsLearning.memory.chat_memory
        })
        kidsLearning.memory.save_context({"userResponse": ""}, {"chat_history": kids_continuation})
        return {"message": kids_continuation}
    except Exception as e:
        return {"error": str(e)}

