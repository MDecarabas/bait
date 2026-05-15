"""Vendored OAS (Ophyd-as-a-Service) FastAPI server.

Only the REST surface (/api/v1/devices*, /api/v1/load-devices) used by bait's
device tools is included here — the original ophyd-websocket repo also exposes
WebSocket endpoints for live PV/camera/console streaming that bait does not use.

Spawned by `bait.devices.oas_server.ensure_server` as a subprocess via
`python -m bait.ophyd_websocket.server --startup-dir <file>`. The Python
interpreter must have `ophyd` and `pyepics` installed; bait itself does not
depend on them.
"""
