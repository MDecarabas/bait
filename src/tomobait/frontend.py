import os

import gradio as gr
import requests

from .config import BaitConfig

# Load configuration
config = BaitConfig()
BACKEND_URL = f"http://{config.server.backend_host}:{config.server.backend_port}/chat"
DOCS_DIR = None
if config.documentation.git_repos:
    repo_name = config.documentation.git_repos[0].split("/")[-1].removesuffix(".git")
    _docs_path = config.docs_output_dir / repo_name / "docs" / "_build" / "html"
    if _docs_path.exists():
        DOCS_DIR = os.path.abspath(_docs_path)


def chat_func(message, history):
    """
    This is the function that Gradio calls when the user sends a message.
    Uses the modern 'messages' format with role and content.
    """
    try:
        response = requests.post(BACKEND_URL, json={"query": message})
        response.raise_for_status()
        agent_response = response.json().get("response", "No response from agent.")

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Format agent response and add to history
        # For now, we'll just add the text response
        # Images will be embedded in the text if present
        history.append({"role": "assistant", "content": agent_response})

        return history

    except requests.exceptions.RequestException as e:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": f"Error connecting to backend: {e}"}
        )
        return history


def new_conversation():
    """
    Start a new conversation.
    """
    return []


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# TomoBait Chat")
    gr.Markdown("Ask questions about the 2-BM beamline documentation.")

    with gr.Tabs():
        # --- Tab 1: Chat Interface ---
        with gr.Tab("Chat"):
            with gr.Row():
                new_chat_btn = gr.Button("🆕 New Conversation", size="sm")

            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=500,
            )

            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Enter your question and press enter",
                    container=False,
                )

            # Connect chat function
            txt.submit(chat_func, [txt, chatbot], [chatbot]).then(
                lambda: "", None, txt
            )  # Clear input

            # Connect new conversation button
            new_chat_btn.click(new_conversation, [], chatbot)


def main():
    allowed_paths = [DOCS_DIR] if DOCS_DIR else []
    demo.launch(
        server_name=config.server.frontend_host,
        server_port=config.server.frontend_port,
        allowed_paths=allowed_paths,  # This is crucial for serving images
    )


if __name__ == "__main__":
    main()
