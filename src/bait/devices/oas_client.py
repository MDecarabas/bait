"""Sync REST client for the ophyd-websocket OAS server.

Wraps `GET /api/v1/devices/{name}` and `PUT /api/v1/devices` in a tiny
synchronous interface so the LangGraph bits_agent_node can call read/set
without async wrapping. The OAS server itself does the ophyd work.
"""

from typing import Any, Optional

import requests


def _coerce_numeric(value):
    """Convert numeric-looking strings ('5.0', '7') to float/int. Pass-through otherwise."""
    if not isinstance(value, str):
        return value
    try:
        return int(value) if value.lstrip("-").isdigit() else float(value)
    except ValueError:
        return value


class OASClient:
    """Thin wrapper around the OAS REST API."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        # base_url like "http://localhost:8002"; the /api/v1 prefix is added per-call.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def read_device(self, name: str, component: Optional[str] = None) -> dict[str, Any]:
        """Read a registered device. Optionally extract a single component value."""
        try:
            resp = requests.get(self._url(f"/devices/{name}"), timeout=self.timeout)
        except requests.RequestException as exc:
            return {"ok": False, "error": f"OAS unreachable: {exc}"}

        if resp.status_code == 404:
            return {"ok": False, "error": f"device {name!r} not registered"}
        if resp.status_code != 200:
            return {"ok": False, "error": f"OAS HTTP {resp.status_code}: {resp.text}"}

        info = resp.json()
        if component is None:
            return {"ok": True, "info": info}

        # OAS get_device_info returns device.read() under "values", keyed by
        # f"{device_name}_{component_name}".
        values = info.get("values") or {}
        key = f"{name}_{component}"
        if key not in values:
            return {
                "ok": False,
                "error": (
                    f"component {component!r} not found in device {name!r} "
                    f"(available: {sorted(values.keys())})"
                ),
            }
        entry = values[key]
        return {
            "ok": True,
            "value": entry.get("value"),
            "timestamp": entry.get("timestamp"),
            "connected": info.get("connected"),
        }

    def set_device(
        self,
        name: str,
        value: int | float | str,
        component: Optional[str] = None,
        timeout: int = 5,
    ) -> dict[str, Any]:
        """Set a value on a registered device. Surfaces 423 (queue locked) plainly."""
        # The LLM sometimes emits numeric values as JSON strings ("5.0"). OAS
        # passes the value straight to ophyd which then does arithmetic and
        # explodes with TypeError. Coerce here so device.set() sees a number.
        body = {
            "device": name,
            "value": _coerce_numeric(value),
            "component": component,
            "timeout": timeout,
        }
        try:
            resp = requests.put(self._url("/devices"), json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            return {"ok": False, "error": f"OAS unreachable: {exc}"}

        if resp.status_code == 200:
            return {"ok": True, "result": resp.json()}
        if resp.status_code == 423:
            return {
                "ok": False,
                "queue_locked": True,
                "error": (
                    "queue server is running an experiment; cannot modify devices"
                ),
                "detail": resp.json(),
            }
        if resp.status_code == 404:
            return {"ok": False, "error": f"device {name!r} not registered"}
        return {
            "ok": False,
            "error": f"OAS HTTP {resp.status_code}: {resp.text}",
        }
