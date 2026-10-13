"""Gradio UI. Run with:  python -m studyagent.app"""
import gradio as gr
from dotenv import load_dotenv

from .agent import StudyAgent
from .ingest import load_pdf
from .retrieval import NotesIndex

load_dotenv()

CSS = """
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; opacity: 0.72; margin-bottom: 18px; }
#sidebar { padding: 8px; }
"""

# Loading the embedding model is slow, so do it once and share it across sessions.
_shared_index = NotesIndex()
_model = _shared_index.model


def process_pdf(file_path, state):
    """Build a fresh per-session index from the uploaded PDF."""
    if not file_path:
        return "Upload a PDF first.", state
    text, pages = load_pdf(file_path)
    if not text.strip():
        return "I couldn't extract text from that PDF (is it a scan?).", state
    index = NotesIndex(model=_model)
    n_chunks = index.build(text)
    state = StudyAgent(index)
    return f"**Notes are ready!**  \n**Pages:** {pages}  \n**Chunks:** {n_chunks}", state


def respond(message, history, state):
    history = history or []
    if not message or not message.strip():
        return "", history, state
    if state is None:
        reply = "Please upload and process your PDF before starting a study session."
    else:
        try:
            reply = state.ask(message, history)
        except Exception as e:  # surface API errors in the chat instead of crashing
            reply = f"Something went wrong: {e}"
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", history, state


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        agent_state = gr.State(None)  # one StudyAgent (and index) per browser session
        gr.Markdown("# 📚 StudyAgent", elem_id="title")
        gr.Markdown("Upload your class notes and study with your own AI tutor.", elem_id="subtitle")
        with gr.Row():
            with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
                gr.Markdown("### Course Material")
                pdf_upload = gr.File(label="Upload your notes", file_types=[".pdf"], type="filepath")
                process_button = gr.Button("Process Notes", variant="primary")
                status = gr.Markdown("Upload a PDF to get started.")
                gr.Markdown("**Try:** lecture slides, class notes, textbook chapters, or a study guide.")
            with gr.Column(scale=2):
                gr.Markdown("### Study Session")
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="<strong>StudyAgent</strong><br>Process your notes, then ask me anything or say <em>Quiz me!</em>",
                )
                message = gr.Textbox(placeholder='Ask a question or say "Quiz me!"', label="Message", lines=2)
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear Chat")

        process_button.click(process_pdf, [pdf_upload, agent_state], [status, agent_state])
        io = dict(inputs=[message, chatbot, agent_state], outputs=[message, chatbot, agent_state])
        send_button.click(respond, **io)
        message.submit(respond, **io)
        clear_button.click(lambda: ("", []), outputs=[message, chatbot])
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft(), css=CSS)
