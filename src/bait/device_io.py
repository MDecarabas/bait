"""In-process ophyd device read/write.

Bait imports ophyd directly (see pyproject.toml) and reuses the vendored
``device_registry`` from ``bait.ophyd_websocket``. No subprocess, no HTTP
roundtrip — the bits_agent's read tool and the HITL confirm endpoint both
call these helpers directly.
"""

import logging
import os
import sys
from typing import Any, Optional

import httpx

from .config import BaitConfig
from .ophyd_websocket.device_registry import device_registry

logger = logging.getLogger(__name__)

_QSERVER_HOST = os.getenv("QSERVER_HTTP_SERVER_HOST", "localhost")
_QSERVER_PORT = int(os.getenv("QSERVER_HTTP_SERVER_PORT", "60610"))
_QSERVER_API_KEY = os.getenv("QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY", "test")


def _coerce_numeric(value):
    """LLM emits '5.0' as a JSON string; ophyd's set then does target - '5.0'
    and explodes with TypeError. Coerce numeric-looking strings."""
    if not isinstance(value, str):
        return value
    try:
        return int(value) if value.lstrip("-").isdigit() else float(value)
    except ValueError:
        return value


def load_devices(config: BaitConfig) -> list[str]:
    """Populate device_registry from config.oas_startup_file. Returns names.

    Two harvest paths run in sequence:
      1. The vendored device_registry scans the loaded module's top-level
         names — works for example_startup.py-style files.
      2. apsbits.make_devices writes Devices into a guarneri Instrument's
         registry rather than the module namespace, so we additionally
         pull from apsbits.core.instrument_init._instrument when present.
    """
    startup_file = str(config.oas_startup_file)
    device_registry.clear()
    device_registry.load_startup_files(startup_file)
    extra = _harvest_apsbits_oregistry()
    if extra:
        logger.info("[load_devices] harvested %d apsbits device(s)", extra)
    return device_registry.list_devices()


def _harvest_apsbits_oregistry() -> int:
    """Pick up devices created by apsbits.make_devices.

    Returns the count newly added to device_registry. No-op when apsbits
    isn't loaded or hasn't initialized an instrument yet.
    """
    module = sys.modules.get("apsbits.core.instrument_init")
    if module is None:
        return 0
    instrument = getattr(module, "_instrument", None)
    if instrument is None or not hasattr(instrument, "devices"):
        return 0
    oregistry = instrument.devices
    if not hasattr(oregistry, "device_names"):
        return 0
    existing = set(device_registry.list_devices())
    added = 0
    for name in oregistry.device_names:
        if name in existing:
            continue
        try:
            device_registry.add_device(name, oregistry[name])
            added += 1
        except (ValueError, KeyError) as exc:
            logger.warning(
                "[harvest_apsbits] skipping %r: %s", name, exc
            )
    return added


def read_device(name: str, component: Optional[str] = None) -> dict[str, Any]:
    """Read a registered device. Same return shape as the old OASClient."""
    device = device_registry.get_device(name)
    if device is None:
        return {"ok": False, "error": f"device {name!r} not registered"}

    info = device_registry.get_device_info(name) or {}
    if component is None:
        return {"ok": True, "info": info}

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


def check_queueserver(config: BaitConfig) -> tuple[bool, Optional[str]]:
    """Sync probe — returns (safe_to_write, alert_if_not).

    safe_to_write is True only when the QS is reachable AND idle. Mirrors the
    semantics of OAS's ``@queue_safety_required`` decorator without going via
    the subprocess.
    """
    url = f"http://{_QSERVER_HOST}:{_QSERVER_PORT}/api/status"
    headers = {"Authorization": f"Apikey {_QSERVER_API_KEY}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=1.0)
    except httpx.RequestError:
        return False, (
            "Queue server is not running. Writes will fail until "
            f"you start it: bash {config.queueserver_script}"
        )
    if resp.status_code != 200:
        return False, (
            f"Queue server responded HTTP {resp.status_code}; "
            f"start it: bash {config.queueserver_script}"
        )
    status = resp.json()
    running = status.get("running_item_uid") is not None or status.get(
        "manager_state"
    ) in ("running", "paused")
    if running:
        return False, (
            "Queue server is currently running an experiment "
            f"(state={status.get('manager_state')!r}); cannot modify devices."
        )
    return True, None


def set_device(
    config: BaitConfig,
    name: str,
    value,
    component: Optional[str] = None,
    timeout: int = 5,
) -> dict[str, Any]:
    """Set a device value. Runs the QS safety check first when require_qserver."""
    if config.ophyd_websocket.require_qserver:
        safe, alert = check_queueserver(config)
        if not safe:
            return {"ok": False, "queue_locked": True, "error": alert}

    device = device_registry.get_device(name)
    if device is None:
        return {"ok": False, "error": f"device {name!r} not registered"}

    target = device
    target_name = name
    if component is not None:
        if not hasattr(device, component):
            return {
                "ok": False,
                "error": f"component {component!r} not on device {name!r}",
            }
        target = getattr(device, component)
        target_name = f"{name}.{component}"

    if not hasattr(target, "set"):
        return {"ok": False, "error": f"{target_name!r} does not support set"}

    coerced = _coerce_numeric(value)
    try:
        status = target.set(coerced)
        status.wait(timeout=timeout)
    except Exception as exc:
        return {"ok": False, "error": f"set failed: {exc}"}

    return {
        "ok": True,
        "result": {
            "device": name,
            "component": component,
            "value": coerced,
            "message": f"Set {target_name} to {coerced}",
        },
    }
