# 🧠 AI Tutor Python API — FastAPI + LangChain Backend


## 📘 Overview

This repository contains the backend API for the [AI Tutor](https://ai-tutor-senior-project.vercel.app/) application. Built with **FastAPI** and **LangChain**, it powers the conversational tutoring experience by handling:

- Subject-based tutoring sessions
- Interactive quizzes
- PDF document Q&A
- Professional mode with Markdown and LaTeX support
- User progress tracking and badge awarding

The frontend communicates with this API to deliver a seamless and interactive learning experience.

---

## 🛠 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **AI Orchestration**: [LangChain](https://www.langchain.com/)
- **Language Model**: [OpenAI GPT-4o-mini](https://platform.openai.com/)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **Deployment**: [Render](https://render.com/)

---

## 📁 Project Structure

```
python-api/
├── casualLearning.py          # Logic for casual tutoring sessions
├── freeChat.py                # Handles free-form chat interactions
├── kidsLearning.py            # Tailored sessions for younger users
├── professionalLearning.py    # Advanced sessions with Markdown/LaTeX support
├── pdfLearning.py             # PDF upload and Q&A functionality
├── main.py                    # FastAPI app and route definitions
├── requirements.txt           # Python dependencies
├── render.yaml                # Deployment configuration for Render
└── README.md                  # Project documentation
```

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ECampbell37/Python-API_SeniorProject.git
cd Python-API_SeniorProject
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## 🌐 API Endpoints

| Method | Endpoint             | Description                                                  |
|--------|----------------------|--------------------------------------------------------------|
| GET    | `/health`            | Health check endpoint                                        |
| GET    | `/intro`             | Start a casual tutoring session with an intro message        |
| POST   | `/chat`              | Continue a casual tutoring conversation                      |
| GET    | `/quiz/start`        | Generate a quiz based on the tutoring session                |
| POST   | `/quiz/submit`       | Submit quiz answers and receive feedback + grade             |
| GET    | `/continue`          | Continue the session after the quiz                          |
| POST   | `/free_chat`         | Start a free-form, open-ended conversation                   |
| GET    | `/kids_intro`        | Start a kids mode session with a subject-based intro         |
| POST   | `/kids_chat`         | Continue the conversation in kids mode                       |
| GET    | `/kids_quiz/start`   | Generate a quiz for kids                                     |
| POST   | `/kids_quiz/submit`  | Submit kids quiz answers and receive feedback + grade        |
| GET    | `/kids_continue`     | Continue kids session after quiz                             |
| POST   | `/professional_chat` | Chat with formatting-aware AI (Markdown, LaTeX, code, etc.)  |
| POST   | `/pdf/upload`        | Upload a PDF for document-based tutoring                     |
| POST   | `/pdf/ask`           | Ask a question about the uploaded PDF                        |

Each endpoint requires a valid `x-user-id` header and a JSON or file payload.  
Refer to the code for full request/response details.


---

## 🚀 Deployment

This API is configured for deployment on [Render](https://render.com/). The `render.yaml` file contains the necessary specifications. To deploy:

1. Push your repository to a GitHub repository.
2. Create a new Web Service on Render.
3. Connect your GitHub repository.
4. Render will automatically detect the `render.yaml` and set up the service.

---

## 🤝 Author

🏆 Made by [Elijah Campbell-Ihim](https://github.com/ECampbell37)  
🎓 CMPS-450 Senior Project — Spring 2025
