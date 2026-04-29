"""Tests for FastAPI app endpoints (chat history CRUD + error paths)."""

import importlib

import pytest


def _make_client(monkeypatch, route_question=None):
    """Build a FastAPI TestClient with route_question patched."""
    import bait.agents
    import bait.app

    importlib.reload(bait.agents)
    importlib.reload(bait.app)

    if route_question is not None:
        monkeypatch.setattr(bait.app, "route_question", route_question)

    from fastapi.testclient import TestClient

    return TestClient(bait.app.api)


@pytest.fixture
def client(write_config, monkeypatch):
    write_config()
    return _make_client(monkeypatch, route_question=lambda q: "Test answer")


def test_chat_endpoint(client):
    """POST /chat returns the agent's response."""
    response = client.post("/chat", json={"query": "test question"})
    assert response.status_code == 200
    assert response.json()["response"] == "Test answer"


def test_save_and_load_chat(client):
    save_response = client.post(
        "/chat/save",
        json={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        },
    )
    assert save_response.status_code == 200
    chat_id = save_response.json()["id"]
    assert chat_id

    load_response = client.get(f"/chat/history/{chat_id}")
    assert load_response.status_code == 200
    data = load_response.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "hello"


def test_list_chats(client):
    client.post(
        "/chat/save",
        json={"messages": [{"role": "user", "content": "test"}]},
    )
    response = client.get("/chat/history")
    assert response.status_code == 200
    assert len(response.json()["chats"]) >= 1


def test_delete_chat(client):
    save_resp = client.post(
        "/chat/save",
        json={"messages": [{"role": "user", "content": "delete me"}]},
    )
    chat_id = save_resp.json()["id"]

    del_resp = client.delete(f"/chat/history/{chat_id}")
    assert del_resp.status_code == 200

    load_resp = client.get(f"/chat/history/{chat_id}")
    assert load_resp.status_code == 404


def test_load_missing_chat_returns_404(client):
    response = client.get("/chat/history/nonexistent-id")
    assert response.status_code == 404


def test_auto_title_generation(client):
    response = client.post(
        "/chat/save",
        json={
            "messages": [
                {"role": "user", "content": "How do I configure the sample stage?"},
                {"role": "assistant", "content": "Here's how..."},
            ]
        },
    )
    assert response.status_code == 200
    assert "sample stage" in response.json()["title"].lower()


# --- Phase 1a: Argo failure → 503 ---


def _argo_error(exc_cls):
    def _raise(_q):
        raise exc_cls(request=None, message="boom")  # type: ignore[arg-type]

    return _raise


def test_chat_endpoint_handles_api_timeout(write_config, monkeypatch):
    """APITimeoutError → 503 with the user-facing message."""
    write_config()
    from openai import APITimeoutError

    def _timeout(_q):
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    client = _make_client(monkeypatch, route_question=_timeout)
    resp = client.post("/chat", json={"query": "x"})
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_chat_endpoint_handles_api_connection_error(write_config, monkeypatch):
    """APIConnectionError → 503."""
    write_config()
    from openai import APIConnectionError

    def _conn(_q):
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    client = _make_client(monkeypatch, route_question=_conn)
    resp = client.post("/chat", json={"query": "x"})
    assert resp.status_code == 503


def test_chat_endpoint_handles_api_status_error(write_config, monkeypatch):
    """APIStatusError → 503."""
    write_config()
    from unittest.mock import MagicMock

    from openai import APIStatusError

    def _status(_q):
        response = MagicMock()
        response.status_code = 502
        raise APIStatusError(message="bad gateway", response=response, body=None)

    client = _make_client(monkeypatch, route_question=_status)
    resp = client.post("/chat", json={"query": "x"})
    assert resp.status_code == 503


# --- Phase 1a: POST /config removed → 405 ---


def test_post_config_returns_405(client):
    """POST /config was removed; only GET is allowed."""
    resp = client.post("/config", json={})
    assert resp.status_code == 405


# --- R3: path traversal defense ---


def test_path_traversal_rejected(client):
    """R3: a traversal-y chat_id returns 400, not 404 or success."""
    resp = client.get("/chat/history/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 400


def test_path_traversal_rejected_on_delete(client):
    """Same defense applies to DELETE."""
    resp = client.delete("/chat/history/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 400


# --- Corrupted chat history file is skipped, not crashed on ---


def test_list_chats_skips_corrupted_file(client):
    """A bogus .json file in the history dir doesn't take down the endpoint."""
    # Save one valid chat first.
    client.post("/chat/save", json={"messages": [{"role": "user", "content": "valid"}]})

    # Drop a corrupted file alongside it.
    import bait.app

    cfg = bait.app.get_config()
    (cfg.chat_history_dir / "broken.json").write_text("{ not valid json")

    resp = client.get("/chat/history")
    assert resp.status_code == 200
    titles = [c["title"] for c in resp.json()["chats"]]
    assert any("valid" in t for t in titles)
