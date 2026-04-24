import json
import re
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import route_question
from .config import BaitConfig

# --- FastAPI App ---
api = FastAPI()


class ChatQuery(BaseModel):
    query: str


def _safe_chat_path(chat_history_dir, chat_id: str):
    """Resolve a chat file path, guarding against path traversal."""
    filepath = (chat_history_dir / f"{chat_id}.json").resolve()
    if not filepath.is_relative_to(chat_history_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid chat ID")
    return filepath


class ChatMessage(BaseModel):
    role: str
    content: str


class SaveChatRequest(BaseModel):
    messages: List[ChatMessage]
    title: Optional[str] = None
    id: Optional[str] = Field(None, description="Pass existing id to update a chat")


@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    answer = route_question(chat_query.query)
    return {"response": answer}


@api.get("/config")
async def get_config_endpoint():
    """
    Get current configuration.
    """
    config = BaitConfig()
    return {"config": config.model_dump()}


@api.post("/config")
async def update_config_endpoint(new_config: dict):
    """
    Update configuration (requires restart to apply).
    """
    # This is a placeholder - in production you'd want to validate and save
    return {"message": "Configuration updated. Restart backend to apply changes."}


def _generate_title(messages: List[ChatMessage]) -> str:
    """Extract first user message and truncate to 60 chars at word boundary."""
    for msg in messages:
        if msg.role == "user" and msg.content.strip():
            text = msg.content.strip()
            if len(text) <= 60:
                return text
            return text[:60].rsplit(" ", 1)[0]
    return "Untitled Chat"


def _make_chat_id(title: str) -> str:
    """Combine ISO timestamp + sanitized title slug."""
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50]
    return f"{ts}_{slug}"


@api.post("/chat/save")
async def save_chat(request: SaveChatRequest):
    """Save or update a conversation."""
    config = BaitConfig()
    now = datetime.now().isoformat(timespec="seconds")
    title = request.title or _generate_title(request.messages)

    if request.id:
        chat_id = request.id
        filepath = _safe_chat_path(config.chat_history_dir, chat_id)
        if filepath.exists():
            data = json.loads(filepath.read_text())
            data["title"] = title
            data["updated_at"] = now
            data["message_count"] = len(request.messages)
            data["messages"] = [m.model_dump() for m in request.messages]
            filepath.write_text(json.dumps(data, indent=2))
            return {"id": chat_id, "title": title}
    else:
        chat_id = _make_chat_id(title)

    data = {
        "id": chat_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": len(request.messages),
        "messages": [m.model_dump() for m in request.messages],
    }
    filepath = _safe_chat_path(config.chat_history_dir, chat_id)
    filepath.write_text(json.dumps(data, indent=2))
    return {"id": chat_id, "title": title}


@api.get("/chat/history")
async def list_chats():
    """List all saved chats (metadata only, no messages)."""
    config = BaitConfig()
    chats = []
    for f in sorted(config.chat_history_dir.glob("*.json"), reverse=True):
        data = json.loads(f.read_text())
        chats.append(
            {
                "id": data["id"],
                "title": data["title"],
                "created_at": data["created_at"],
                "message_count": data["message_count"],
            }
        )
    return {"chats": chats}


@api.get("/chat/history/{chat_id:path}")
async def load_chat(chat_id: str):
    """Load a specific conversation with full messages."""
    config = BaitConfig()
    filepath = _safe_chat_path(config.chat_history_dir, chat_id)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    return json.loads(filepath.read_text())


@api.delete("/chat/history/{chat_id:path}")
async def delete_chat(chat_id: str):
    """Delete a saved conversation."""
    config = BaitConfig()
    filepath = _safe_chat_path(config.chat_history_dir, chat_id)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    filepath.unlink()
    return {"deleted": chat_id}


def main():
    """
    Main function to run the FastAPI application using uvicorn.
    """
    config = BaitConfig()
    uvicorn.run(api, host=config.server.backend_host, port=config.server.backend_port)
