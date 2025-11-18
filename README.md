# AmbedkarGPT-Intern-Task

This project is the official submission for the **KalpIT AmbedkarGPT Intern Assignment**.
It implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

- LangChain
- ChromaDB (local vectorstore)
- HuggingFaceEmbeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Ollama (local) with **Mistral 7B** as the LLM

The system loads a speech text file, splits it into chunks, creates embeddings,
stores them in Chroma, retrieves relevant chunks for a user question, and then
generates an answer using the Mistral model via Ollama.

---

## Project Structure

```
AmbedkarGPT-Intern-Task/
│── main.py
│── main_advanced.py
│── speech.txt
│── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- Git
- Virtual environment (recommended)
- **Ollama installed locally**
- **Mistral model pulled locally**

Install Ollama from: https://ollama.com  
Then run:

```
ollama pull mistral
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
```

### 2. Create Virtual Environment

**Windows:**
```
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

### First time (needs embedding creation):

```
python main.py --rebuild
```

After that:

```
python main.py
```

You will see:

```
AmbedkarGPT — ask a question about the speech (type 'exit' to quit)
```

Example:

```
Question: What is the real remedy for caste?
```

---

## Advanced Version

An enhanced version is available as `main_advanced.py`. It provides:
- Better prompt templating
- Simple logging
- Colored CLI prompts (if `colorama` is installed)
- Option to change retrieval `k` via CLI

Run:

```
python main_advanced.py --rebuild
python main_advanced.py
```

---

## Submission

Make sure your GitHub repo is **public** and named:

```
AmbedkarGPT-Intern-Task
```

Push all files and share the repo link.

Contact: kalpiksingh2005@gmail.com
