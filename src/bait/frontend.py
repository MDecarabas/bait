"""Gradio frontend for Bait."""

from pathlib import Path
from typing import List

import gradio as gr
import requests

from .config import BaitConfig, get_config


def _backend_url(config: BaitConfig) -> str:
    return f"http://{config.server.backend_host}:{config.server.backend_port}"


def _chat_url(config: BaitConfig) -> str:
    return f"{_backend_url(config)}/chat"


def _resolve_allowed_paths(config: BaitConfig) -> List[str]:
    """Return absolute paths Gradio is allowed to serve as static assets.

    Walks every configured ``git_repos`` build directory and every
    ``local_folders`` entry. Missing build dirs are silently skipped — that
    just means the user hasn't run ``bait-ingest`` for that source yet, and
    the frontend should still come up.
    """
    paths: List[str] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        if not p.exists():
            return
        absolute = str(p.resolve())
        if absolute in seen:
            return
        seen.add(absolute)
        paths.append(absolute)

    for repo_url in config.documentation.git_repos:
        repo_name = repo_url.split("/")[-1].removesuffix(".git")
        _add(config.docs_output_dir / repo_name / "docs" / "_build" / "html")

    for folder in config.documentation.local_folders:
        _add(Path(folder))

    return paths


def _format_pending_writes(writes: list[dict]) -> str:
    """Render the proposed writes as a markdown table for the confirm group."""
    rows = ["| device | component | value |", "|---|---|---|"]
    for w in writes:
        name = w.get("name", "")
        component = w.get("component") or "—"
        value = w.get("value", "")
        rows.append(f"| `{name}` | `{component}` | `{value}` |")
    return "\n".join(rows)


def chat_func(message, history, pending_state):
    """Send a message to the backend, render response, surface any HITL writes."""
    config = get_config()
    try:
        response = requests.post(_chat_url(config), json={"query": message})
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": f"Error connecting to backend: {e}"}
        )
        return history, pending_state, gr.update(visible=False), "", ""

    agent_response = data.get("response", "No response from agent.")
    pending_writes = data.get("pending_writes")
    pending_id = data.get("pending_id")
    qs_alert = data.get("qs_alert")

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": agent_response})

    if pending_writes and pending_id:
        return (
            history,
            {"pending_id": pending_id, "writes": pending_writes},
            gr.update(visible=True),
            _format_pending_writes(pending_writes),
            f"⚠️ {qs_alert}" if qs_alert else "",
        )
    return history, None, gr.update(visible=False), "", ""


def confirm_writes(history, pending_state):
    """Send the staged writes to /chat/confirm with approved=True."""
    return _resolve_pending(history, pending_state, approved=True)


def cancel_writes(history, pending_state):
    """Send the staged writes to /chat/confirm with approved=False."""
    return _resolve_pending(history, pending_state, approved=False)


def _resolve_pending(history, pending_state, approved: bool):
    if not pending_state:
        return history, None, gr.update(visible=False), "", ""
    config = get_config()
    pending_id = pending_state["pending_id"]
    try:
        resp = requests.post(
            f"{_backend_url(config)}/chat/confirm",
            json={"pending_id": pending_id, "approved": approved},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        history.append(
            {"role": "assistant", "content": f"Error confirming writes: {e}"}
        )
        return history, None, gr.update(visible=False), "", ""

    if data.get("denied"):
        history.append(
            {"role": "assistant", "content": "Cancelled — no writes performed."}
        )
    else:
        lines = []
        for entry in data.get("results", []):
            r = entry["result"]
            w = entry["write"]
            label = f"{w['name']}{('.' + w['component']) if w.get('component') else ''}"
            if r.get("ok"):
                lines.append(f"✅ Set `{label}` to `{w['value']}`.")
            else:
                err = r.get("error", "unknown error")
                lines.append(f"❌ Failed to set `{label}` to `{w['value']}`: {err}")
                if r.get("qs_alert"):
                    lines.append(f"⚠️ {r['qs_alert']}")
        history.append({"role": "assistant", "content": "\n".join(lines)})
    return history, None, gr.update(visible=False), "", ""


def new_conversation():
    """Start a new conversation."""
    return [], None


def _extract_text(content):
    """Extract plain text from Gradio's content format."""
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

    config = get_config()
    clean_msgs = [
        {"role": str(m["role"]), "content": _extract_text(m.get("content", ""))}
        for m in history
    ]
    payload = {"messages": clean_msgs}
    if current_chat_id:
        payload["id"] = current_chat_id

    try:
        resp = requests.post(f"{_backend_url(config)}/chat/save", json=payload)
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
    config = get_config()
    try:
        resp = requests.get(f"{_backend_url(config)}/chat/history")
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

    config = get_config()
    try:
        resp = requests.get(f"{_backend_url(config)}/chat/history")
        resp.raise_for_status()
        chats = resp.json().get("chats", [])
        labels = [f"{c['title']} ({c['message_count']} msgs)" for c in chats]
        if selected not in labels:
            return [], None

        idx = labels.index(selected)
        chat_id = id_list[idx]
        resp = requests.get(f"{_backend_url(config)}/chat/history/{chat_id}")
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

    config = get_config()
    try:
        resp = requests.get(f"{_backend_url(config)}/chat/history")
        resp.raise_for_status()
        chats = resp.json().get("chats", [])
        labels = [f"{c['title']} ({c['message_count']} msgs)" for c in chats]
        if selected not in labels:
            gr.Warning("Chat not found.")
            return gr.update(), id_list

        idx = labels.index(selected)
        chat_id = id_list[idx]
        requests.delete(f"{_backend_url(config)}/chat/history/{chat_id}")
        gr.Info("Chat deleted.")
    except requests.exceptions.RequestException as e:
        gr.Warning(f"Failed to delete: {e}")

    return load_history_list()


def _build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("# Bait Chat")
        gr.Markdown("Ask questions about the beamline documentation and devices.")

        current_chat_id = gr.State(None)
        history_ids = gr.State([])
        pending_state = gr.State(None)

        with gr.Sidebar(label="Chat History", open=False, position="left"):
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            history_radio = gr.Radio(choices=[], label="Saved Chats", interactive=True)
            with gr.Row():
                load_btn = gr.Button("📂 Load", size="sm")
                delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    new_chat_btn = gr.Button("🆕 New Conversation", size="sm")
                    save_btn = gr.Button("💾 Save Chat", size="sm")

                chatbot = gr.Chatbot([], elem_id="chatbot", height=500)

                with gr.Group(visible=False) as pending_group:
                    gr.Markdown("### Confirm proposed device writes")
                    qs_banner = gr.Markdown("")
                    pending_table = gr.Markdown("")
                    with gr.Row():
                        confirm_btn = gr.Button("✅ Confirm", variant="primary")
                        cancel_btn = gr.Button("❌ Cancel", variant="stop")

                with gr.Row():
                    txt = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder="Enter your question and press enter",
                        container=False,
                    )

                txt.submit(
                    chat_func,
                    [txt, chatbot, pending_state],
                    [chatbot, pending_state, pending_group, pending_table, qs_banner],
                ).then(lambda: "", None, txt)
                confirm_btn.click(
                    confirm_writes,
                    [chatbot, pending_state],
                    [chatbot, pending_state, pending_group, pending_table, qs_banner],
                )
                cancel_btn.click(
                    cancel_writes,
                    [chatbot, pending_state],
                    [chatbot, pending_state, pending_group, pending_table, qs_banner],
                )
                new_chat_btn.click(new_conversation, [], [chatbot, current_chat_id])
                save_btn.click(save_chat, [chatbot, current_chat_id], [current_chat_id])

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

    return demo


def main():
    config = get_config()
    allowed_paths = _resolve_allowed_paths(config)
    demo = _build_demo()
    demo.launch(
        server_name=config.server.frontend_host,
        server_port=config.server.frontend_port,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
