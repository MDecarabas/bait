# ophyd_websocket/ — vendored OAS package map

This is a **vendored** copy of upstream OAS (Ophyd-as-a-Service):
https://github.com/anl-aps/ophyd-device-control

It is a standalone FastAPI server that exposes ophyd Devices and EPICS PVs
over REST + four WebSocket protocols. It also contains a reusable
`DeviceRegistry` class that bait imports as a Python library — bait does **not**
launch the FastAPI server.

Read this file before touching anything in `src/bait/ophyd_websocket/`.

---

## TL;DR — what bait actually uses

| File | bait uses it? | How |
|------|---------------|-----|
| `device_registry.py` | **YES** | `bait.device_io` imports the global `device_registry` singleton |
| `queue_safety.py` | No | bait has its own sync version in `device_io.check_queueserver` |
| `server.py` | No | Standalone FastAPI app; **broken** when run as `python -m bait.ophyd_websocket.server` (see "Known broken parts" below) |
| `routers/*.py` | No | All four WebSocket routers + REST router are vendored but unused by bait |
| `examples/example_startup.py` | No (reference only) | Shows the device-declaration pattern the registry expects |

**Single import path bait uses:**

```python
from bait.ophyd_websocket.device_registry import device_registry
```

If you need anything else from this package, you are probably going outside
bait's current scope — stop and re-evaluate.

---

## Data flow when bait reads/sets a device

```
LangGraph bits_agent
  └─> tool call: read_device(name) / set_device(name, value)
       └─> bait.device_io.read_device / set_device
            └─> device_registry.get_device(name)        # in-process Python lookup
                 └─> ophyd Device.read() / .set()       # Channel Access via pyepics
                      └─> EPICS IOC                     # the actual hardware
```

For writes, bait additionally hits the bluesky queueserver's REST API
(`http://localhost:60610/api/status`) **before** calling `Device.set()`, to
ensure no plan is currently running. That path goes through `bait.device_io`,
not through this package.

---

## File-by-file map

### `__init__.py`
Empty package marker. Do not put logic here — the upstream refresh process
overwrites it.

### `device_registry.py`
The only file bait depends on.

- `class DeviceRegistry`
  - `_devices: dict[str, ophyd.Device]` — the in-memory map
  - `load_startup_files(path)` — executes a Python file (or every `.py` in a
    dir) via `importlib.util.spec_from_file_location`, then iterates
    `vars(loaded_module)` and adds anything that passes `_is_ophyd_device`
  - `_is_ophyd_device(obj, name)` — returns True iff `obj` is an instance
    (not a class) of `Device` or `EpicsSignal` AND the name doesn't start
    with `_`
  - `get_device(name)` / `add_device(name, device)` / `remove_device(name)`
  - `list_devices()` — list of registered names
  - `get_device_info(name)` — type, prefix, connected-bool, `describe()`,
    `read()` (cached values dict)
  - `clear()` / `reload_devices()`
- `device_registry = DeviceRegistry()` — module-level singleton bait imports

**Discovery rule (critical):** A device is only registered if it exists
**at module top level** in the loaded startup file. Devices created inside
functions, attached to other objects, or written to a different module's
namespace are invisible to this registry. See "Known concerns" below for
how this collides with apsbits.

### `queue_safety.py`
`async` HTTPx-based check that the bluesky queueserver is reachable and
not running an experiment. Reads:

- `QSERVER_HTTP_SERVER_HOST` (default `localhost`)
- `QSERVER_HTTP_SERVER_PORT` (default `60610`)
- `QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY` (default `test`)
- `OAS_REQUIRE_QSERVER` (default `false`) — strict mode

Exposes `check_queue_server_safety()` and a `@queue_safety_required`
decorator for FastAPI endpoints.

bait does **not** use this — it ships an equivalent **sync** version in
`bait.device_io.check_queueserver` because the FastAPI lifespan runs inside
a uvicorn loop and dispatching async-from-sync was complicating the safety
check.

### `server.py`
Standalone FastAPI server. Mounts the REST router and all four websocket
routers. Reads many env vars (see "Environment variables" below) and runs
under `uvicorn.run("server:app", host=..., port=..., reload=True)`.

**bait does not launch this.** If you ever need to, see "Known broken parts."

### `examples/example_startup.py`
Reference for the kind of file `device_registry.load_startup_files()` expects:
module-level instances of `Device` or `EpicsSignal`. Read this if you're
debugging "why isn't my device showing up?"

### `routers/core_api.py` (REST)
All routes are mounted under `/api/v1/` by `server.py`.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/load-devices` | Re-runs `device_registry.load_startup_files(startup_dir)` |
| `GET`  | `/devices` | Returns `{"devices": [...names], "count": N}` |
| `GET`  | `/devices/{device_name}` | Per-device info (`get_device_info`) |
| `GET`  | `/devices-info` | All devices' info in one payload |
| `PUT`  | `/devices` | Set a device value (queue-safety-gated) |
| `GET`  | `/queue-server/status` | Proxy to `<qserver>/api/status` |
| `GET`  | `/pvs` | List connected raw PVs (in-memory `pv_dict`) |
| `GET`  | `/pvs/{pv}` | Read a PV value |
| `POST` | `/pvs/{pv}` | Open an `EpicsSignal` for a raw PV |
| `PUT`  | `/pvs` | Set a raw PV value (queue-safety-gated) |

Request payloads (pydantic models in the router file):

```python
DeviceSetInstruction(device: str, value: str|int|float, component: str|None, timeout: int|None)
EpicsPVInstruction(pv: str, set_value: str|int|float, timeout: int|None)
```

### `routers/device_socket.py` (WebSocket)
Endpoint: `/api/v1/device-socket`. Live ophyd Device monitoring + control by
device name. Subscribes to ophyd callbacks (`callbackMd` for metadata,
`callbackValue` for value updates) and pushes JSON over the socket.

Supported client actions: `subscribe`, `subscribeSafely`, `subscribeReadOnly`,
`unsubscribe`, `refresh`, `set`.

Example client message: `{"action": "subscribe", "device": "tomoscan"}`

### `routers/pv_socket.py` (WebSocket)
Endpoint: `/api/v1/pv-socket`. Same protocol as `device_socket` but operates
on raw EPICS PV names instead of registered ophyd Devices.

Example: `{"action": "subscribe", "pv": "IOC:m1"}`

### `routers/camera_socket.py` (WebSocket)
Endpoint: `/api/v1/camera-socket`. Streams area-detector image arrays as
base64-encoded JPEGs. Initialization message specifies the image array PV:
`{"imageArray_PV": "MYDET:image1:ArrayData"}`.

### `routers/qs_console_socket.py` (WebSocket)
Endpoint: `/api/v1/qs-console-socket`. Bridges the bluesky queueserver's
ZMQ console-publish socket (default `tcp://localhost:60625`) to the
WebSocket. One-way — server pushes QS console lines as text frames.

---

## Environment variables

Read by various files in this package. Bait does not set any of these
itself — they're inherited from the shell.

| Variable | Read by | Default | Purpose |
|----------|---------|---------|---------|
| `OAS_HOST` | `server.py` | `localhost` | Standalone server bind host |
| `OAS_PORT` | `server.py` | `8001` | Standalone server bind port |
| `OAS_REQUIRE_QSERVER` | `queue_safety.py`, `server.py` | `false` | Strict mode for QS gating |
| `OAS_ALLOWED_ORIGINS` | `server.py` | empty | Extra CORS origins |
| `OAS_STARTUP_DIR` | `device_registry.py`, `server.py` | unset | Default startup path |
| `OAS_LOG_LEVEL` | `server.py` | `INFO` | Logging level |
| `QSERVER_HTTP_SERVER_HOST` | `queue_safety.py` | `localhost` | bluesky-httpserver host |
| `QSERVER_HTTP_SERVER_PORT` | `queue_safety.py` | `60610` | bluesky-httpserver port |
| `QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY` | `queue_safety.py` | `test` | bluesky-httpserver API key |
| `ZMQ_HOST` | `qs_console_socket.py` | `localhost` | ZMQ console host |
| `ZMQ_PORT` | `qs_console_socket.py` | `60625` | ZMQ console port |
| `EPICS_CA_ADDR_LIST` | ophyd/pyepics | unset | EPICS Channel Access broadcast list |
| `EPICS_CA_AUTO_ADDR_LIST` | ophyd/pyepics | unset | EPICS auto-discovery toggle |
| `EPICS_CA_MAX_ARRAY_BYTES` | ophyd/pyepics | unset | EPICS max waveform size |

bait reads `QSERVER_HTTP_SERVER_*` directly in `bait.device_io` for its
own queue-safety check.

---

## Known concerns

### 1. apsbits namespace mismatch — RESOLVED in `bait.device_io.load_devices`

**Background:** the upstream `device_registry.load_startup_files()` discovers
devices by iterating `vars(loaded_startup_module)` — works for
`example_startup.py` (which declares `sim_motor1 = EpicsSignal(...)` at
module top level). BITS's `startup.py` does NOT: it calls
`apsbits.make_devices(file="devices.yml", device_manager=instrument)`, which
in `apsbits/core/instrument_init.py:135-167` writes constructed devices into
`sys.modules["__main__"]` AND into the guarneri Instrument's own registry
(`_instrument.devices`) — neither of those is the loaded startup module's
namespace, so the upstream scanner finds zero devices.

**Fix:** `bait.device_io.load_devices` runs the upstream scan first (so
example-style startups still work), then calls `_harvest_apsbits_oregistry()`
which checks `sys.modules["apsbits.core.instrument_init"]._instrument.devices`
and adds anything new to the registry. No-op when apsbits isn't loaded, so
example-style startups are unaffected.

**Verify:** `tests/test_device_io.py` covers the harvest logic in isolation
(apsbits-not-loaded, instrument-unset, fresh-add, dedup, invalid-device-skip).
End-to-end against the real beamline: start bait against
`/Users/ecodrea/tomo-bits/configs/bait_config.yaml`, watch the lifespan log
for `[load_devices] harvested N apsbits device(s)` and `[startup] loaded N
device(s): [...]`, then call `device_registry.list_devices()` from a Python
repl in the same env — should show `tomoscan`, `sim_motor`, `sim_det`.

**If devices are still missing** in the live env: print
`sys.modules["apsbits.core.instrument_init"]._instrument.devices.device_names`
to see what guarneri actually loaded — the harvester only adds what's
already there.

### 2. `server.py` bare imports — broken in this layout

```python
from routers.pv_socket import router       # line 11
from device_registry import device_registry # line 18
```

These resolve only when CWD is `src/bait/ophyd_websocket/`. Running
`python -m bait.ophyd_websocket.server` fails with
`ModuleNotFoundError: No module named 'routers'`. The `core_api.py` and
`device_socket.py` routers have the same problem (`from device_registry
import device_registry`).

bait doesn't run this server, so it's not a runtime issue today. If you
ever want to launch the standalone OAS server alongside bait, either
patch the imports to be relative (`from .routers.pv_socket import router`,
`from .device_registry import device_registry`) or run from the package
directory: `cd src/bait/ophyd_websocket && python server.py --startup-dir <file>`.

### 3. `__init__.py` is intentionally empty

Earlier docstring edits were wiped by an upstream refresh of the vendored
code. Do not re-add a docstring here — put package-level documentation in
this CLAUDE.md instead.

---

## Diagnostic checklist

When something doesn't work, check in this order:

1. **Is the queueserver up?**
   ```bash
   curl -H "Authorization: Apikey test" http://localhost:60610/api/status
   ```
   Must return JSON with `manager_state: "idle"` for writes to be allowed.
   If down, run `bash /Users/ecodrea/tomo-bits/scripts/tomo_2bm_qs_host.sh start`.

2. **Are devices loaded into the registry?**
   In a Python repl with the bait env active:
   ```python
   from bait.ophyd_websocket.device_registry import device_registry
   from bait.config import get_config
   from bait.device_io import load_devices
   load_devices(get_config())
   print(device_registry.list_devices())
   ```
   If this returns `[]`, see "Concern #1" above.

3. **Is EPICS reachable?**
   ```bash
   echo $EPICS_CA_ADDR_LIST
   caget 2bmb:TomoScan:RotationStart   # or any PV from devices.yml
   ```
   If `caget` times out, the device's `connected` will be False and reads
   return stale/None.

4. **Is bait pointing at the right config?**
   Default config is `/Users/ecodrea/tomo-bits/configs/bait_config.yaml`.
   Verify `bits.path`, `bits.instrument_name`, and the resolved
   `oas_startup_file` (printed at lifespan startup).

5. **Looking at logs?**
   Bait's startup logs go via the `logging` module — backend startup logs
   the device count (`[startup] loaded N device(s): [...]`). If N is 0,
   `device_registry` is empty (back to step 2).

---

## How to extend

- **Add a new ophyd Device type:** create the class in `tomo-bits/src/tomo_2bm/devices/`, add an entry to `tomo-bits/src/tomo_2bm/configs/devices.yml`, restart bait. The registry auto-discovers it on next `load_devices`.
- **Add a new REST endpoint to the standalone server:** edit `routers/core_api.py`. Not used by bait.
- **Replace bait's facade:** the only contract `bait.device_io` requires from this package is `device_registry.load_startup_files(path)`, `get_device(name)`, `get_device_info(name)`, `list_devices()`, `clear()`. Anything that satisfies that interface drops in.

---

## When to update this file

- A vendored upstream refresh changed file sizes / contents (compare against this map)
- bait starts using a new file from this package (update the TL;DR table)
- The apsbits namespace concern gets resolved (update the "Known concerns" section)
- A new env var is introduced upstream
