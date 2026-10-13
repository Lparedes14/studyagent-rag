# How RAG works, step by step

This guide walks through the exact pipeline in this repo. Each step maps to a file you can open and read.

**The problem:** an LLM doesn't know *your* notes, and if you ask it about them it may make things up.
**The idea:** before the model answers, *retrieve* the relevant passages from your notes and hand them to it.
That's **R**etrieval-**A**ugmented **G**eneration.

```
PDF ─► text ─► chunks ─► embeddings ─► vector index
                                            │
 question ─► embed ─► similarity search ────┘
                           │ top-k passages
                           ▼
              LLM (with passages as context) ─► grounded answer
```

## Step 1. Load the document  (`ingest.load_pdf`)
Extract raw text from the PDF with PyMuPDF. Scanned PDFs have no text layer, so they return nothing. That needs OCR, which this project doesn't do.

## Step 2. Chunk the text  (`ingest.chunk_text`)
Models can't search a whole book at once, and one giant vector would blur every topic together. So we cut the text into **1000-character chunks with 200 characters of overlap**.
- *Smaller chunks:* more precise matches, but less context per match.
- *Larger chunks:* more context, but noisier matches.
- *Overlap:* a sentence cut at a boundary still appears whole in one chunk.

## Step 3. Embed the chunks  (`retrieval.NotesIndex.build`)
An **embedding model** (`all-MiniLM-L6-v2`) turns each chunk into a 384-number vector. Text with similar *meaning* ends up with vectors that point in similar directions, even with no shared keywords. We normalise vectors to length 1.

## Step 4. Search by meaning  (`retrieval.NotesIndex.search`)
1. Embed the user's question with the *same* model.
2. Compute similarity with every chunk. Because the vectors are normalised, a dot product is the cosine similarity.
3. Sort and keep the **top-k** (default 5).

These top-k chunks are the "retrieved" context. At this point you have a working semantic search engine, with no LLM yet. (The notebook shows it on its own.)

## Step 5. Augment and generate
Give the LLM the question plus the retrieved passages and tell it to answer *from them*. That's the plain "RAG" pattern.

## Step 6. Make it an agent  (`agent.StudyAgent`)
Instead of retrieving once up front, we expose retrieval as a **tool** (`search_notes`) and let the model decide **when** and **what** to search. That's *agentic RAG*:
- "Quiz me!" → the model searches for a topic, asks a question, then searches again to check your answer.
- "What is X?" → one targeted search, then an answer.
- "Hi" → no search at all.

The **system prompt** defines the tutor behaviour (one question at a time, don't reveal answers, remember the conversation).

## Step 7. Wrap it in a UI  (`app.py`)
Gradio gives upload, chat, and per-session state with no front-end code.

## Try it yourself: interacting with a RAG system
Upload a PDF, then try these in order and watch the behaviour change:

| Prompt | What it shows |
|---|---|
| "Summarize my notes." | broad retrieval |
| "What does <specific term from your notes> mean?" | precise retrieval |
| "What is the capital of Mars?" | grounding. It should say the notes don't cover it |
| "Quiz me!" → "anything" | multi-turn tool use and memory |
| Answer wrong on purpose | the evaluation behaviour |

## Known limitations (and how I'd improve them)
- **Fixed-size chunking** can split mid-sentence → try sentence- or heading-aware chunking.
- **Brute-force search** is fine for one document → use FAISS/Chroma for many.
- **No citations** → return page numbers with each chunk and have the model cite them.
- **Dense-only retrieval** misses exact terms → add hybrid search (BM25 + embeddings) and a reranker.
- **No evaluation** → build a small question/answer set and measure retrieval hit-rate.
