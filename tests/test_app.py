"""Tests for FastAPI app endpoints (chat history CRUD).

These tests mock the agents module to avoid module-level LLM client initialization.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture(autouse=True)
def mock_agents(monkeypatch):
    """Mock the agents module before app.py imports it."""
    fake_agents = types.ModuleType("tomobait.agents")
    fake_agents.route_question = MagicMock(return_value="Test answer")
    monkeypatch.setitem(sys.modules, "tomobait.agents", fake_agents)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Create a temporary config environment for testing."""
    config_data = {
        "project": {"name": "test", "data_dir": str(tmp_path / ".bait-test")},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def client(tmp_config):
    """Create a test client."""
    # Import after agents is mocked and config is set up
    import importlib

    import tomobait.app

    importlib.reload(tomobait.app)
    from fastapi.testclient import TestClient

    return TestClient(tomobait.app.api)


def test_chat_endpoint(client):
    """POST /chat should return a response from the agent."""
    response = client.post("/chat", json={"query": "test question"})
    assert response.status_code == 200
    assert response.json()["response"] == "Test answer"


def test_save_and_load_chat(client):
    """Save a chat, then load it back."""
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
    """List chats should return saved conversations."""
    client.post(
        "/chat/save",
        json={"messages": [{"role": "user", "content": "test"}]},
    )
    response = client.get("/chat/history")
    assert response.status_code == 200
    chats = response.json()["chats"]
    assert len(chats) >= 1


def test_delete_chat(client):
    """Delete a chat should remove it."""
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
    """Loading a non-existent chat should return 404."""
    response = client.get("/chat/history/nonexistent-id")
    assert response.status_code == 404


def test_auto_title_generation(client):
    """Save without title should auto-generate from first user message."""
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
    title = response.json()["title"]
    assert "sample stage" in title.lower()
