# 📚 StudyAgent: a RAG tutor for your own class notes

Upload a PDF of your notes and chat with an AI tutor that **answers from your material and quizzes you on it**.
Built as a hands-on project to learn and teach **Retrieval-Augmented Generation (RAG)** and tool-calling agents.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lparedes14/studyagent-rag/blob/main/notebooks/AI_Explore_Fall26_RAG_Agent.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<!-- TODO: add a demo GIF at docs/demo.gif, then replace this comment with: ![StudyAgent demo](docs/demo.gif) -->

## What it does
- Extracts and chunks a PDF, embeds it locally, and finds the most relevant passages for each question.
- A Gemini-powered agent decides *when* to search your notes (tool calling) and stays grounded in them.
- Explains concepts, quizzes you one question at a time, and evaluates your answers.

## How it works
```
PDF → text → chunks → embeddings → vector index
                                        ↑
user question → embed → top-k search ───┘ → passages → LLM agent → answer
```
**New to RAG? Read the step-by-step guide: [docs/how-rag-works.md](docs/how-rag-works.md)**
or open the annotated notebook: [notebooks/AI_Explore_Fall26_RAG_Agent.ipynb](notebooks/AI_Explore_Fall26_RAG_Agent.ipynb).

## Tech stack
PyMuPDF · sentence-transformers (`all-MiniLM-L6-v2`) · NumPy · Google Gemini (`google-genai`) · Gradio

## Quickstart
**Try it in the browser:** click the **Open in Colab** badge above, add your key under 🔑 Secrets as `GEMINI_API_KEY`, and run the notebook.

**Run it locally** (Python 3.10+):
```bash
git clone https://github.com/Lparedes14/studyagent-rag.git
cd studyagent-rag
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows PowerShell: copy .env.example .env  (then add your Gemini API key)
PYTHONPATH=src python -m studyagent.app   # PowerShell: $env:PYTHONPATH="src"; python -m studyagent.app
```
Then open the local URL, upload a PDF, click **Process Notes**, and say *"Quiz me!"*
Get a free API key at [Google AI Studio](https://aistudio.google.com/).

## Project structure
```
src/studyagent/
  ingest.py      load PDF, chunk text
  retrieval.py   embeddings + similarity search (NotesIndex)
  agent.py       system prompt, search tool, Gemini agent
  app.py         Gradio UI
notebooks/       annotated walkthrough (Colab-friendly)
docs/            step-by-step RAG explanation
tests/           unit tests (pytest; install dev deps with requirements-dev.txt)
```

## Design decisions
- **Local embeddings** keep notes on your machine; only the retrieved passages go to the LLM.
- **Per-session state** (`gr.State`), not globals, so multiple users don't share one index.
- **Agentic retrieval** instead of one-shot RAG, so the model can search repeatedly while quizzing.

## Limitations & roadmap
- [ ] Page-number citations in answers
- [ ] Sentence/heading-aware chunking
- [ ] Hybrid search (BM25 + embeddings) and reranking
- [ ] Multi-document support with a vector DB (FAISS/Chroma)
- [ ] Retrieval evaluation set
- [ ] Hosted demo on Hugging Face Spaces

## Product thinking
**Problem:** students re-read notes passively, which is a weak way to learn. **Solution:** active recall, quizzing generated from their *own* material. **Success metrics to track:** quiz completion rate, answer accuracy over repeated sessions, and share of answers grounded in retrieved passages.

## Author
Luis Paredes Franco. AI Product Owner. Built during the Fall 2026 AI Explore program.

## License
MIT
