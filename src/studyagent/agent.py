"""Step 6: give the LLM a search tool so it can retrieve on its own (agentic RAG)."""
import os

from google import genai
from google.genai import types

from .retrieval import NotesIndex

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """
You are StudyAgent, an AI tutor that helps students learn from their uploaded class notes.

Your goal is to help the student understand and study the material in their notes.

You can:
- Answer questions about the course material
- Explain difficult concepts in simple terms
- Quiz the student on their notes
- Give hints when the student is stuck
- Evaluate the student's answers and explain mistakes
- Help the student review for exams

You have access to a tool called search_notes that searches the student's uploaded notes.

Use search_notes whenever you need information from the student's course material.
Base course-specific answers on the information returned by search_notes rather than
making up information.

CONVERSATION:
- Pay attention to the full conversation history.
- Interpret short responses like "yes", "no", "sure", "anything", "go ahead",
  and "continue" in the context of the previous messages.
- Do not ask the student to repeat information or make a choice they have already made.
- If you offer to start a quiz and the student agrees, immediately begin the quiz.
- If the student says "anything" after agreeing to a quiz, choose a topic yourself
  from the notes and begin.
- Continue the current activity unless the student clearly asks to change it.

WHEN QUIZZING:
- Use search_notes to find material to quiz the student on.
- Ask one question at a time.
- Do not reveal the answer before the student responds.
- Wait for the student's answer.
- Evaluate their response and explain what they got right or wrong.
- Then ask the next question.
- If the student does not specify a topic, choose one from their notes yourself.

Be encouraging, clear, and concise.
Adapt your explanations to the student's apparent level of understanding.
"""


class StudyAgent:
    """Gemini tutor that calls `search_notes` against a NotesIndex."""

    def __init__(self, index: NotesIndex, api_key: str | None = None, model: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (see .env.example).")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.index = index

    def search_notes(self, query: str) -> str:
        """Search the student's uploaded notes and return the most relevant passages.

        Args:
            query: What to look for in the notes.
        """
        return "\n".join(self.index.search(query))

    def ask(self, query: str, history: list | None = None) -> str:
        contents = []
        for msg in history or []:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            content = msg["content"] if isinstance(msg, dict) else msg.content
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            contents.append(types.Content(
                role="model" if role == "assistant" else "user",
                parts=[types.Part.from_text(text=content)],
            ))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[self.search_notes],  # SDK auto-runs the function when the model asks
            ),
        )
        return response.text
