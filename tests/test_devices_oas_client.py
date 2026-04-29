"""Tests for OASClient against an in-process FastAPI stub."""

from __future__ import annotations

import threading
import time

import pytest
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

# --- In-process stub server ---


class SetBody(BaseModel):
    device: str
    value: int | float | str
    component: str | None = None
    timeout: int | None = None


def _build_stub_app(state: dict) -> FastAPI:
    """Build a minimal FastAPI app mimicking the OAS device endpoints."""
    app = FastAPI()

    @app.get("/api/v1/devices/{name}")
    def get_device(name: str, response: Response):
        if name not in state["devices"]:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"error": "not found"}
        device = state["devices"][name]
        return {
            "name": name,
            "type": "EpicsSignal",
            "connected": True,
            "values": {f"{name}_{c}": {"value": v} for c, v in device.items()},
        }

    @app.put("/api/v1/devices")
    def put_device(body: SetBody, response: Response):
        if state.get("queue_locked"):
            raise HTTPException(status_code=423, detail="queue locked")
        if body.device not in state["devices"]:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"error": "not found"}
        state["sets"].append(body.model_dump())
        return {"success": True, "device": body.device, "value": body.value}

    return app


@pytest.fixture
def stub_server():
    """Spin up the stub on a random port for the duration of one test."""
    state: dict = {
        "devices": {"motor": {"readback": 1.5}},
        "sets": [],
        "queue_locked": False,
    }
    app = _build_stub_app(state)

    # Pick an ephemeral port by binding once.
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for readiness — up to 3s.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{port}/api/v1/devices/motor", timeout=0.5)
            break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        raise RuntimeError("stub server failed to start")

    try:
        yield port, state
    finally:
        server.should_exit = True
        thread.join(timeout=3)


# --- Tests ---


def test_read_device_with_component(stub_server):
    from bait.devices import OASClient

    port, _ = stub_server
    client = OASClient(f"http://127.0.0.1:{port}")
    out = client.read_device("motor", component="readback")
    assert out["ok"]
    assert out["value"] == 1.5


def test_read_device_404(stub_server):
    from bait.devices import OASClient

    port, _ = stub_server
    client = OASClient(f"http://127.0.0.1:{port}")
    out = client.read_device("ghost")
    assert not out["ok"]
    assert "not registered" in out["error"]


def test_set_device_happy(stub_server):
    from bait.devices import OASClient

    port, state = stub_server
    client = OASClient(f"http://127.0.0.1:{port}")
    out = client.set_device("motor", 7.0, component="readback")
    assert out["ok"]
    assert state["sets"] == [
        {"device": "motor", "value": 7.0, "component": "readback", "timeout": 5}
    ]


def test_set_device_queue_locked(stub_server):
    from bait.devices import OASClient

    port, state = stub_server
    state["queue_locked"] = True
    client = OASClient(f"http://127.0.0.1:{port}")
    out = client.set_device("motor", 1)
    assert not out["ok"]
    assert out["queue_locked"] is True
    assert "queue server" in out["error"].lower()


def test_set_device_unknown_device(stub_server):
    from bait.devices import OASClient

    port, _ = stub_server
    client = OASClient(f"http://127.0.0.1:{port}")
    out = client.set_device("ghost", 1)
    assert not out["ok"]
    assert "not registered" in out["error"]


def test_oas_unreachable_returns_error():
    """No server running → ok=False with a connection error."""
    from bait.devices import OASClient

    client = OASClient("http://127.0.0.1:9", timeout=0.5)  # nothing listens on port 9
    out = client.read_device("motor")
    assert not out["ok"]
    assert "unreachable" in out["error"].lower()
