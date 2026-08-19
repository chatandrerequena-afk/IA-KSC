from __future__ import annotations

import os
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from agent import run_agent
from vision import analyze_image


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SKILLS_DIR = BASE_DIR / "skills"
OUTPUTS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")


def initial_state():
    return {
        "session_id": uuid.uuid4().hex[:12],
        "conversation": [],
    }


def session_files(session_dir: Path):
    if not session_dir.exists():
        return []
    return [str(p) for p in sorted(session_dir.iterdir()) if p.is_file()]


def send_message(message, image, history, state):
    message = (message or "").strip()
    if not message and not image:
        return "", None, history, state, []

    if not state or "session_id" not in state:
        state = initial_state()

    session_dir = OUTPUTS_DIR / state["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)

    display_user = message if message else "Analiza la imagen adjunta."
    agent_text = message or "Analiza esta imagen."

    if image:
        description = analyze_image(image, agent_text)
        agent_text += (
            "\n\n[ANÁLISIS VISUAL PROPORCIONADO POR EL MODELO DE VISIÓN]\n"
            + description
            + "\n[FIN DEL ANÁLISIS VISUAL]"
        )
        display_user += "\n\n🖼️ Imagen adjunta"

    reply, conversation = run_agent(
        user_text=agent_text,
        conversation=state.get("conversation", []),
        session_dir=session_dir,
        skills_dir=SKILLS_DIR,
    )
    state["conversation"] = conversation

    history = history or []
    history = history + [
        {"role": "user", "content": display_user},
        {"role": "assistant", "content": reply},
    ]
    return "", None, history, state, session_files(session_dir)


def clear_chat():
    return [], initial_state(), []


with gr.Blocks(title="Agente IA con Skills + Python") as demo:
    gr.Markdown(
        """
# 🤖 Agente IA con Skills, Python y archivos
Pídele código, juegos, documentos, hojas de cálculo, presentaciones, análisis de imágenes o investigación web.
"""
    )

    state = gr.State(initial_state())

    chatbot = gr.Chatbot(
        label="Conversación",
        type="messages",
        height=520,
    )

    with gr.Row():
        message = gr.Textbox(
            label="Escribe tu petición",
            placeholder="Ej.: Créame un juego Snake en Python y dame el archivo para descargar.",
            lines=3,
            scale=4,
        )
        image = gr.Image(
            label="Imagen opcional",
            type="filepath",
            sources=["upload"],
            scale=1,
        )

    with gr.Row():
        send = gr.Button("Enviar", variant="primary")
        clear = gr.Button("Nueva conversación")

    generated = gr.File(
        label="Archivos generados",
        file_count="multiple",
        interactive=False,
    )

    send.click(
        send_message,
        inputs=[message, image, chatbot, state],
        outputs=[message, image, chatbot, state, generated],
    )
    message.submit(
        send_message,
        inputs=[message, image, chatbot, state],
        outputs=[message, image, chatbot, state, generated],
    )
    clear.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, state, generated],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
