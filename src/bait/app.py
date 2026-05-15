"""FastAPI backend for Bait."""

import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import device_io
from .agents import ARGO_UNAVAILABLE_MESSAGE, route_question
from .config import BaitConfig, get_config
from .device_io import check_queueserver

logger = logging.getLogger(__name__)

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError
except ImportError:  # pragma: no cover - openai is a hard dependency
    APIConnectionError = APIStatusError = APITimeoutError = ()  # type: ignore


# Pending writes staged by the bits_agent, keyed by UUID. The frontend echoes
# the pending_id back via /chat/confirm to either execute or discard them.
# Restart-clears (no TTL) — small, simple, sufficient for v1.
_pending_store: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ophyd devices into the in-process registry; warn if QS is down."""
    config = get_config()
    try:
        # bluesky's BestEffortCallback (instantiated by BITS' init_bec_peaks)
        # demands a Qt teleporter pre-created on the main thread before any
        # background-thread instantiation.
        from bluesky.callbacks.mpl_plotting import initialize_qt_teleporter
        initialize_qt_teleporter()
        # apsbits.make_devices() inside the BITS startup file calls
        # asyncio.run(), which forbids a running loop. The lifespan runs
        # inside uvicorn's loop, so dispatch to a worker thread that has
        # none.
        loaded = await asyncio.to_thread(device_io.load_devices, config)
        logger.info("[startup] loaded %d device(s): %s", len(loaded), loaded)
    except Exception:
        logger.exception("[startup] failed to load devices from %s",
                         config.oas_startup_file)
    qs_up, qs_alert = check_queueserver(config)
    if not qs_up:
        logger.warning("[startup] %s", qs_alert)
    yield


api = FastAPI(lifespan=lifespan)


class ChatQuery(BaseModel):
    query: str


class ChatMessage(BaseModel):
    role: str
    content: str


class SaveChatRequest(BaseModel):
    messages: List[ChatMessage]
    title: Optional[str] = None
    id: Optional[str] = Field(None, description="Pass existing id to update a chat")


def _safe_chat_path(chat_history_dir, chat_id: str):
    """Resolve a chat file path, guarding against path traversal."""
    filepath = (chat_history_dir / f"{chat_id}.json").resolve()
    if not filepath.is_relative_to(chat_history_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid chat ID")
    return filepath


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


class ConfirmRequest(BaseModel):
    pending_id: str
    approved: bool


@api.post("/chat")
async def chat_endpoint(
    chat_query: ChatQuery,
    config: BaitConfig = Depends(get_config),
):
    """Receive a query and return the agent's response.

    When the bits_agent stages writes, the response carries ``pending_writes``
    plus a ``pending_id`` the frontend echoes back via /chat/confirm. If the
    queue server is down at the time, ``qs_alert`` carries the operator-facing
    instruction so the user sees it before clicking Confirm.

    Surfaces Argo connectivity failures as 503 with a user-facing message
    instead of leaking a 500 + traceback.
    """
    try:
        answer, pending_writes = route_question(chat_query.query)
    except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
        logger.exception("LLM provider unreachable: %s", exc)
        raise HTTPException(status_code=503, detail=ARGO_UNAVAILABLE_MESSAGE)

    pending_id: Optional[str] = None
    qs_alert: Optional[str] = None
    if pending_writes:
        pending_id = uuid.uuid4().hex
        _pending_store[pending_id] = pending_writes
        qs_up, alert = check_queueserver(config)
        if not qs_up:
            qs_alert = alert

    return {
        "response": answer,
        "pending_writes": pending_writes or None,
        "pending_id": pending_id,
        "qs_alert": qs_alert,
    }


@api.post("/chat/confirm")
async def confirm_endpoint(
    request: ConfirmRequest,
    config: BaitConfig = Depends(get_config),
):
    """Execute or discard a previously-staged set of device writes."""
    writes = _pending_store.pop(request.pending_id, None)
    if writes is None:
        raise HTTPException(status_code=404, detail="pending_id not found")

    if not request.approved:
        return {"results": [], "denied": True}

    results: list[dict[str, Any]] = []
    for write in writes:
        result = device_io.set_device(
            config,
            name=write["name"],
            value=write["value"],
            component=write.get("component"),
        )
        results.append({"write": write, "result": result})
    return {"results": results, "denied": False}


@api.get("/config")
async def get_config_endpoint(config: BaitConfig = Depends(get_config)):
    """Return the loaded configuration (read-only)."""
    return {"config": config.model_dump()}


@api.post("/chat/save")
async def save_chat(
    request: SaveChatRequest,
    config: BaitConfig = Depends(get_config),
):
    """Save or update a conversation."""
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
async def list_chats(config: BaitConfig = Depends(get_config)):
    """List all saved chats (metadata only). Skips corrupted files."""
    chats = []
    for f in sorted(config.chat_history_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable chat history file %s: %s", f, exc)
            continue
        try:
            chats.append(
                {
                    "id": data["id"],
                    "title": data["title"],
                    "created_at": data["created_at"],
                    "message_count": data["message_count"],
                }
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed chat history file %s: %s", f, exc)
            continue
    return {"chats": chats}


@api.get("/chat/history/{chat_id:path}")
async def load_chat(chat_id: str, config: BaitConfig = Depends(get_config)):
    """Load a specific conversation with full messages."""
    filepath = _safe_chat_path(config.chat_history_dir, chat_id)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    return json.loads(filepath.read_text())


@api.delete("/chat/history/{chat_id:path}")
async def delete_chat(chat_id: str, config: BaitConfig = Depends(get_config)):
    """Delete a saved conversation."""
    filepath = _safe_chat_path(config.chat_history_dir, chat_id)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    filepath.unlink()
    return {"deleted": chat_id}


def main():
    """Run the FastAPI application via uvicorn."""
    config = get_config()
    uvicorn.run(api, host=config.server.backend_host, port=config.server.backend_port)
