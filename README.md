
# Question Anticipator Agent – LangGraph Edition

A smart backend system that predicts academic exam questions based on syllabus, past papers, and exam patterns. Fully orchestrated using **LangGraph**, with memory, pattern analysis, and LLM-driven question generation.

---

## **Features**

* Generates multiple-choice (MCQs), short-answer, and long-answer questions.
* Incorporates **past exam patterns** and **syllabus weightage**.
* Uses **Vector Memory** (ChromaDB) for caching predictions.
* **RAG (Retrieval-Augmented Generation)** to provide context-aware questions.
* Supports **Gemini, OpenAI, Anthropic, and Grok** LLMs.
* FastAPI endpoint for easy integration.
* Optional inclusion of answers in output.
* Health endpoint to check memory, embeddings, and LLM availability.

---

## **Tech Stack**

* **Backend:** Python 3.11+, FastAPI
* **Frontend:** React
* **LLM Providers:** Gemini (default), OpenAI, Anthropic, Grok
* **Memory & Embeddings:** ChromaDB + SentenceTransformers
* **Orchestration:** LangGraph
* **Testing:** Pytest

---

## **Installation**

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/question-anticipator.git
cd question-anticipator
```

2. **Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

3. **Set environment variables:**

Create a `.env` file:

```env
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
GEMINI_API_KEY=your_gemini_api_key
LLM_PROVIDER=gemini
LLM_MODEL=models/gemma-3-4b-it
INCLUDE_ANSWERS=true
```

> Note: Only the API key for the LLM provider you want to use is required.

---

## **Usage**

### **Start the API**

```bash
python question_anticipator.py
```

The API will run at:

```
http://0.0.0.0:8000
```

---

### **Endpoints**

1. **Predict Questions**

* **URL:** `/api/predict-questions`
* **Method:** `POST`
* **Request Body:**

```json
{
  "syllabus": ["Topic 1", "Topic 2"],
  "past_papers": [
    {
      "year": "2023",
      "questions": ["What is Topic 1?", "Explain Topic 2."]
    }
  ],
  "exam_pattern": {
    "mcqs": 2,
    "short_questions": 2,
    "long_questions": 1
  },
  "weightage": {"Topic 1": 3, "Topic 2": 2},
  "difficulty_preference": "medium",
  "include_answers": true
}
```

* **Response:**

```json
{
  "predicted_questions": [
    {
      "topic": "Topic 1",
      "question_text": "What is Topic 1?",
      "difficulty_level": "medium",
      "question_type": "short_question",
      "probability_score": 0.9,
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_option": 0,
      "rag_used": true,
      "rag_score": 0.85
    }
  ],
  "from_memory": false,
  "processing_time": 1.23,
  "memory_hash": "a1b2c3d4"
}
```

---

2. **Health Check**

* **URL:** `/health`
* **Method:** `GET`
* **Response:**

```json
{
  "status": "healthy",
  "memory_entries": 10,
  "embeddings_available": true,
  "llm_available": true,
  "timestamp": "2025-11-30T21:50:00"
}
```

---

## **Project Structure**

```
.
├── question_anticipator.py        # Main backend
├── question_anticipator/     # Frontend
├── requirements.txt        # Python dependencies
├── tests/                  # Pytest test cases
├── chroma_db/              # Vector memory storage (ChromaDB)
├── .env                    # Environment variables
└── README.md
```

---

## **Testing**

Run the tests:

```bash
pytest -v
```

> Note: For faster tests, you may disable embeddings & LLM calls or mock them.

---

## **Notes**

* **Memory persistence:** ChromaDB stores previous predictions to speed up repeated requests.
* **RAG:** Questions use syllabus context for realistic content.
* **Fallbacks:** If LLM fails, template-based questions are generated.

---

## **License**

MIT License © 2025 Unzila Anjum


