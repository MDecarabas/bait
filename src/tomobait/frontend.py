import os
import re

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

# Global state for current conversation
current_conversation_id = None


def format_response(text):
    """
    This function takes the raw text response from the agent and formats it for
    display in the Gradio interface. It converts image paths to local URLs
    that Gradio can serve.
    """
    # If we don't have a docs directory, we can't serve images.
    if not DOCS_DIR:
        return [(text, None)]

    # Find all image paths (markdown or raw)
    image_paths = re.findall(
        r"!\[.*?\]\((.*?)\)|([\w\-/.]+\.(?:png|jpg|jpeg|gif|svg))", text
    )

    # Flatten the list of tuples from findall
    flat_paths = [item for sublist in image_paths for item in sublist if item]

    # Create a list of tuples (original_text, image_path)
    # to be used in the Gradio chatbot component
    output_components = []

    # Start with the full text
    remaining_text = text

    for path in flat_paths:
        # We split the text by the image path to insert the image
        parts = remaining_text.split(path, 1)

        # Add the text before the image
        if parts[0].strip():
            # also remove the markdown remnant `![]()`
            clean_text = re.sub(r"!\[.*?\]\(\)", "", parts[0]).strip()
            if clean_text:
                output_components.append((clean_text, None))

        # Add the image
        # Gradio needs an absolute path to serve the file
        full_image_path = os.path.join(DOCS_DIR, path)
        if os.path.exists(full_image_path):
            output_components.append((None, full_image_path))
        else:
            # If the image path is broken, just append the text
            output_components.append((f"(Image not found: {path})", None))

        # The rest of the text
        remaining_text = parts[1] if len(parts) > 1 else ""

    # Add any remaining text after the last image
    if remaining_text.strip():
        output_components.append((remaining_text.strip(), None))

    # If no images were found, just return the original text
    if not output_components:
        return [(text, None)]

    return output_components


def chat_func(message, history):
    """
    This is the function that Gradio calls when the user sends a message.
    Uses the modern 'messages' format with role and content.
    """
    global current_conversation_id

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
    global current_conversation_id
    current_conversation_id = None
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
