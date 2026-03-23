import os

import gradio as gr
import requests

from .config import BaitConfig

# Load configuration
config = BaitConfig()
BACKEND_URL = f"http://{config.server.backend_host}:{config.server.backend_port}"
CHAT_URL = f"{BACKEND_URL}/chat"
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
        response = requests.post(CHAT_URL, json={"query": message})
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
    """Start a new conversation."""
    return [], None


def _extract_text(content):
    """Extract plain text from Gradio's content format.

    Content can be a string, or a list of dicts like
    [{'text': '...', 'type': 'text'}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def save_chat(history, current_chat_id):
    """Save current conversation to backend."""
    if not history:
        gr.Info("Nothing to save — chat is empty.")
        return current_chat_id

    # Strip extra Gradio fields (metadata, etc.) — backend expects only role+content
    clean_msgs = [
        {"role": str(m["role"]), "content": _extract_text(m.get("content", ""))}
        for m in history
    ]
    payload = {"messages": clean_msgs}
    if current_chat_id:
        payload["id"] = current_chat_id

    try:
        resp = requests.post(f"{BACKEND_URL}/chat/save", json=payload)
        if resp.status_code == 422:
            print("Validation error:", resp.json())
        resp.raise_for_status()
        data = resp.json()
        gr.Info(f"Chat saved: {data['title']}")
        return data["id"]
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Failed to save: {e}")
        return current_chat_id


def load_history_list():
    """Fetch list of saved chats from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history")
        resp.raise_for_status()
        chats = resp.json().get("chats", [])
        labels = [f"{c['title']} ({c['message_count']} msgs)" for c in chats]
        ids = [c["id"] for c in chats]
        return gr.update(choices=labels, value=None), ids
    except requests.exceptions.RequestException:
        return gr.update(choices=[], value=None), []


def load_selected_chat(selected, id_list):
    """Load a selected chat into the chatbot."""
    if selected is None or not id_list:
        return [], None

    try:
        # Match selected label to chat list to find the id
        resp = requests.get(f"{BACKEND_URL}/chat/history")
        resp.raise_for_status()
        chats = resp.json().get("chats", [])
        labels = [f"{c['title']} ({c['message_count']} msgs)" for c in chats]
        if selected not in labels:
            return [], None

        idx = labels.index(selected)
        chat_id = id_list[idx]
        resp = requests.get(f"{BACKEND_URL}/chat/history/{chat_id}")
        resp.raise_for_status()
        data = resp.json()
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
        ]
        return messages, chat_id
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Failed to load chat: {e}")
        return [], None


def delete_selected_chat(selected, id_list):
    """Delete the selected chat and refresh the list."""
    if selected is None or not id_list:
        gr.Info("No chat selected.")
        return gr.update(), id_list

    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history")
        resp.raise_for_status()
        chats = resp.json().get("chats", [])
        labels = [f"{c['title']} ({c['message_count']} msgs)" for c in chats]
        if selected not in labels:
            gr.Warning("Chat not found.")
            return gr.update(), id_list

        idx = labels.index(selected)
        chat_id = id_list[idx]
        requests.delete(f"{BACKEND_URL}/chat/history/{chat_id}")
        gr.Info("Chat deleted.")
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Failed to delete: {e}")

    return load_history_list()


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# TomoBait Chat")
    gr.Markdown("Ask questions about the 2-BM beamline documentation.")

    current_chat_id = gr.State(None)
    history_ids = gr.State([])

    with gr.Sidebar(label="Chat History", open=False, position="left"):
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        history_radio = gr.Radio(choices=[], label="Saved Chats", interactive=True)
        with gr.Row():
            load_btn = gr.Button("📂 Load", size="sm")
            delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

    with gr.Tabs():
        # --- Tab 1: Chat Interface ---
        with gr.Tab("Chat"):
            with gr.Row():
                new_chat_btn = gr.Button("🆕 New Conversation", size="sm")
                save_btn = gr.Button("💾 Save Chat", size="sm")

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
            new_chat_btn.click(new_conversation, [], [chatbot, current_chat_id])

            # Connect save button
            save_btn.click(save_chat, [chatbot, current_chat_id], [current_chat_id])

    # Sidebar event handlers
    refresh_btn.click(load_history_list, [], [history_radio, history_ids])

    load_btn.click(
        load_selected_chat,
        [history_radio, history_ids],
        [chatbot, current_chat_id],
    )

    delete_btn.click(
        delete_selected_chat,
        [history_radio, history_ids],
        [history_radio, history_ids],
    )


def main():
    allowed_paths = [DOCS_DIR] if DOCS_DIR else []
    demo.launch(
        server_name=config.server.frontend_host,
        server_port=config.server.frontend_port,
        allowed_paths=allowed_paths,  # This is crucial for serving images
    )


if __name__ == "__main__":
    main()
