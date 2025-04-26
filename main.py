from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import casualLearning
import freeChat
import kidsLearning
import professionalLearning


app = FastAPI()

# Per-user quiz tracking
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

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple status check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ********* Casual Learning ***************

@app.get("/intro")
async def get_intro(subject: str = "Astronomy", x_user_id: str = Header(...)):
    try:
        memory = casualLearning.get_user_memory(x_user_id)
        intro_text = casualLearning.intro_chain.run({"subject": subject})
        memory.save_context({"userResponse": ""}, {"chat_history": intro_text})
        return {"message": intro_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def post_chat(request: Request, subject: str = "Astronomy", x_user_id: str = Header(...)):
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
    try:
        casualLearning.clear_user_memory(x_user_id)
        return {"status": "Memory cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/quiz/start")
async def start_quiz(subject: str = "Astronomy", x_user_id: str = Header(...)):
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

# ********* Free Chat ***************

@app.post("/free_chat")
async def post_free_chat(request: Request, x_user_id: str = Header(...)):
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
    try:
        freeChat.clear_user_memory(x_user_id)
        return {"status": "Free chat memory cleared"}
    except Exception as e:
        return {"error": str(e)}

# ********** Kids Mode *************

@app.get("/kids_intro")
async def kids_get_intro(subject: str = "Nature", x_user_id: str = Header(...)):
    try:
        memory = kidsLearning.get_user_memory(x_user_id)
        kids_intro_text = kidsLearning.kids_intro_chain.run({"subject": subject})
        memory.save_context({"userResponse": ""}, {"chat_history": kids_intro_text})
        return {"message": kids_intro_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/kids_chat")
async def kids_post_chat(request: Request, subject: str = "Nature", x_user_id: str = Header(...)):
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
    try:
        kidsLearning.clear_user_memory(x_user_id)
        return {"status": "Kids memory cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/kids_quiz/start")
async def kids_start_quiz(subject: str = "Nature", x_user_id: str = Header(...)):
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



# ********** Professional Mode *************

@app.post("/professional_chat")
async def post_professional_chat(request: Request, x_user_id: str = Header(...)):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "Missing 'message'"}
    try:
        memory = professionalLearning.get_user_memory(x_user_id)
        pro_response_text = professionalLearning.response_chain.run({
            "userResponse": user_message,
            "chat_history": memory.chat_memory
        })
        return {"message": pro_response_text}
    except Exception as e:
        return {"error": str(e)}


