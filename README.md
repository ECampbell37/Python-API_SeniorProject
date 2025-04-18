# 🧠 AI Tutor Python API (LangChain + FastAPI)

This is the backend API for the AI Tutor application, built using **FastAPI** and **LangChain**, powered by **OpenAI GPT-3.5**. It provides endpoints for subject-based tutoring, interactive quizzes, and adaptive learning feedback.

The frontend (built in Next.js) communicates with this API to drive conversational tutoring and lesson generation.

---

## 📂 Folder Structure

```
AI_Tutor_Python-API/
├── casualLearning.py      # Core LangChain logic and memory
├── main.py                # FastAPI server and route definitions
├── requirements.txt       # Python dependencies
├── render.yaml            # Optional Render deployment config
└── .env                   # Local environment variables (not pushed to GitHub)
```

---

## ⚙️ Requirements

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/account/api-keys)
- FastAPI
- Uvicorn
- LangChain

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI_Tutor_Python-API.git
cd AI_Tutor_Python-API
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your `.env` file

Create a `.env` file in the root folder:

```
OPENAI_API_KEY=your_openai_key_here
```

> (Alternatively, set the key directly in your terminal with `export OPENAI_API_KEY=...`)

### 5. Run the server

```bash
uvicorn main:app --reload
```

Visit:  
`http://localhost:8000/intro?subject=Astronomy`

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/intro`         | Start the lesson with an intro message   |
| POST   | `/chat`          | Continue the conversation                |
| GET    | `/quiz/start`    | Generate a 5-question quiz               |
| POST   | `/quiz/submit`   | Submit quiz answers and get feedback     |
| GET    | `/continue`      | Continue the lesson after the quiz       |

All routes accept a `subject` query parameter (e.g., `?subject=Astronomy`)

---

## ☁️ Deploying to Render

1. Push your project to a new GitHub repo
2. Go to [render.com](https://render.com)
3. Create a **New Web Service**
4. Fill in:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Add your environment variable:
   ```
   OPENAI_API_KEY=your_openai_key
   ```

Render will give you a public API URL like:  
`https://your-api-name.onrender.com`

---

## 🧠 About the Project

This backend powers an educational AI that:
- Introduces users to new subjects
- Conducts engaging back-and-forth tutoring
- Generates AI-driven quizzes with feedback and grades
- Adjusts lesson difficulty based on performance


---
