"""OAS server lifecycle helpers: spawn-if-needed and queue-server probe."""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen
from typing import Optional

import httpx
import requests

from ..config import BaitConfig

logger = logging.getLogger(__name__)

# Hard cap on how long we wait for the spawned OAS server to come up before
# giving up. First ophyd connect can take a few seconds.
_SERVER_READY_TIMEOUT = 15.0
_SERVER_POLL_INTERVAL = 0.5

# Queue-server defaults match the OAS server's own defaults so the probe
# targets the same instance the OAS would talk to.
_QSERVER_HOST = os.getenv("QSERVER_HTTP_SERVER_HOST", "localhost")
_QSERVER_PORT = int(os.getenv("QSERVER_HTTP_SERVER_PORT", "60610"))
_QSERVER_API_KEY = os.getenv("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "test")


def _server_url(config: BaitConfig) -> str:
    return f"http://{config.ophyd_websocket.host}:{config.ophyd_websocket.port}"


def _is_server_up(config: BaitConfig) -> bool:
    try:
        resp = requests.get(f"{_server_url(config)}/api/v1/devices", timeout=1.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def ensure_server(config: BaitConfig) -> Optional[Popen]:
    """Return a Popen iff this call spawned the server; None if already up or disabled.

    Sequence:
      1. If OAS is disabled, no-op.
      2. Probe /api/v1/devices; reuse if already running.
      3. Else spawn `python server.py --startup-dir <file>` from the OAS repo.
      4. Poll until ready or timeout.
      5. POST /load-devices to populate the registry.
    """
    if not config.ophyd_websocket.enabled:
        logger.info("[OAS] disabled in config; skipping")
        return None

    if _is_server_up(config):
        logger.info(
            "[OAS] server already reachable at %s; reusing", _server_url(config)
        )
        _trigger_load_devices(config)
        return None

    if not config.ophyd_websocket.auto_start:
        logger.warning(
            "[OAS] not reachable and auto_start=false; device tools will fail"
        )
        return None

    startup_file = config.oas_startup_file
    repo_path = Path(config.ophyd_websocket.repo_path)
    server_dir = repo_path / "src" / "ophyd_websocket"
    if not (server_dir / "server.py").is_file():
        logger.error(
            "[OAS] server.py not found at %s; check ophyd_websocket.repo_path",
            server_dir,
        )
        return None
    if not Path(startup_file).is_file():
        logger.error("[OAS] startup file %s does not exist", startup_file)
        return None

    env = {
        **os.environ,
        "OAS_PORT": str(config.ophyd_websocket.port),
        "OAS_HOST": config.ophyd_websocket.host,
        "OAS_REQUIRE_QSERVER": "true"
        if config.ophyd_websocket.require_qserver
        else "false",
        "OAS_STARTUP_DIR": str(startup_file),
    }
    python = config.ophyd_websocket.python_executable or sys.executable
    logger.info(
        "[OAS] spawning server from %s with python=%s startup_file=%s on port %d",
        server_dir,
        python,
        startup_file,
        config.ophyd_websocket.port,
    )
    proc = subprocess.Popen(
        [python, "server.py", "--startup-dir", str(startup_file)],
        cwd=str(server_dir),
        env=env,
    )

    deadline = time.monotonic() + _SERVER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _is_server_up(config):
            logger.info("[OAS] server is up at %s", _server_url(config))
            _trigger_load_devices(config)
            return proc
        if proc.poll() is not None:
            logger.error(
                "[OAS] subprocess exited prematurely with code %d", proc.returncode
            )
            return None
        time.sleep(_SERVER_POLL_INTERVAL)

    logger.error(
        "[OAS] server failed to come up within %.1fs; killing subprocess",
        _SERVER_READY_TIMEOUT,
    )
    proc.terminate()
    return None


def _trigger_load_devices(config: BaitConfig) -> None:
    """Tell the OAS to populate its registry from OAS_STARTUP_DIR."""
    try:
        resp = requests.post(f"{_server_url(config)}/api/v1/load-devices", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "[OAS] loaded %d devices: %s",
                data.get("new_device_count", 0),
                data.get("devices_loaded", []),
            )
        else:
            logger.warning(
                "[OAS] load-devices returned %d: %s", resp.status_code, resp.text
            )
    except requests.RequestException as exc:
        logger.warning("[OAS] load-devices request failed: %s", exc)


def check_queueserver(config: BaitConfig) -> tuple[bool, Optional[str]]:
    """Probe the Bluesky queue server. Returns (is_up, alert_message_if_down)."""
    url = f"http://{_QSERVER_HOST}:{_QSERVER_PORT}/api/status"
    headers = {"Authorization": f"Apikey {_QSERVER_API_KEY}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=1.0)
        if resp.status_code == 200:
            return True, None
        alert = (
            f"Queue server responded HTTP {resp.status_code}; "
            f"writes may be blocked. Start it with: bash {config.queueserver_script}"
        )
        return False, alert
    except httpx.RequestError:
        alert = (
            "Queue server is not running. Writes will fail with HTTP 423 until "
            f"you start it: bash {config.queueserver_script}"
        )
        return False, alert
