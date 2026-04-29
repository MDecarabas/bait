# BITS/Ophyd Device Integration Plan for TomoBait

## Context & Motivation

TomoBait currently only answers questions from Sphinx documentation via ChromaDB retrieval. The user wants agents that can also **interact with live beamline hardware** — ophyd devices defined in their BITS (Bluesky Instrument) installation.

**BITS** is the general term for the beamline's Bluesky/ophyd instrument configuration. In this case, the user's BITS installation is at `/Users/ecodrea/usaxs-bits`, packaged as `bits_usaxs` (importable as `usaxs`). It uses the `apsbits` framework for device management.

**The ask:** "I should have a set of agents which know how to engage with the ophyd devices that the user's BITS has."
**Interaction level:** Live read AND control (not just documentation).

---

## What's in a BITS Installation

The user's BITS at `/Users/ecodrea/usaxs-bits` contains:

```
src/usaxs/
├── devices/          # 25 ophyd Device modules (stages, amplifiers, detectors, etc.)
├── plans/            # 26 Bluesky plan modules (scans, tuning, mode changes)
├── configs/          # 9 YAML files defining device instances and PV prefixes
├── callbacks/        # Data acquisition callbacks
├── suspenders/       # Beam suspension logic
├── startup.py        # Full session initialization (RunEngine, devices, plans)
└── utils/            # Helper functions
```

### How devices are defined

Ophyd Device classes in Python, using Component descriptors with EPICS PV suffixes:

```python
class UsaxsSampleStageDevice(MotorBundle):
    x: Component[EpicsMotor] = Component(EpicsMotor, "usxAERO:m8", labels=("sample",))
    y: Component[EpicsMotor] = Component(EpicsMotor, "usxAERO:m9", labels=("sample",))
```

### How devices are instantiated

YAML configs in `configs/` define which classes to instantiate and with what names:

```yaml
# configs/devices.yml
usaxs.devices.stages.UsaxsSampleStageDevice:
  - name: s_stage

ophyd.EpicsMotor:
  - name: waxsx
    prefix: usxAERO:m3
    labels: ["motor", "baseline"]
```

At startup, `apsbits.core.instrument_init.make_devices()` reads these YAMLs, imports the classes, instantiates them, and registers them in `oregistry` — a global device registry.

### How devices are accessed

```python
from apsbits.core.instrument_init import oregistry

device = oregistry["s_stage"]       # Get by name
device.x.position                   # Read motor position
device.x.move(5.0)                  # Move motor
signal.get()                        # Read signal value
signal.put(new_value)               # Write signal value
```

---

## Design Decisions

### Decision 1: Extend existing agents vs. create new ones

**Chosen: Extend existing agent pair with new tools.**

The current architecture has `doc_expert` (LLM) + `tool_worker` (executor). Adding new tools to `tool_worker` is trivial — just more `@register_for_execution` decorators. The LLM's `tools` list gets extended, and the system message is updated to describe device capabilities.

**Why not separate agents:**
- AG2 GroupChat adds significant complexity (routing, turn management)
- A separate agent pair would need its own chat endpoint and routing logic
- The tools are simple functions — the complexity is in `bits.py`, not in the agent wiring
- The same LLM can handle both documentation and device queries; it just needs the right tools

**Tradeoff:** If device interactions become much more complex (multi-step plans, safety workflows), a dedicated agent might make sense later. But for read/set/inspect, extending the existing pair is the right call.

### Decision 2: How to initialize devices (avoiding startup.py)

**Chosen: Parse YAML configs + instantiate devices directly.**

The BITS `startup.py` does heavy initialization (RunEngine, callbacks, data management, suspenders). We don't need any of that for device read/write — we just need live ophyd Device instances connected to EPICS.

The approach:
1. Add BITS `src/` directory to `sys.path`
2. Parse each YAML config file to extract `(class_path, name, prefix, kwargs)` tuples
3. Dynamically import each class with `importlib`
4. Instantiate directly: `DeviceClass(name=name, prefix=prefix)`
5. Store in a module-level dict for tool access

This gives us live EPICS-connected devices without the Bluesky session overhead.

**Why not import startup.py:**
- It creates a RunEngine (side effect)
- It connects to data management services
- It runs beam suspender logic
- It imports all plans into the module namespace
- Any of these can fail and block TomoBait from starting

**Why not use oregistry from a running session:**
- Would require TomoBait to share a process with the Bluesky session
- Tight coupling to a running IPython/queueserver session

**Tradeoff:** Some devices may need special initialization that only happens in startup.py (e.g., devices that reference other devices via `oregistry["other_device"]` in their `__init__`). These would fail to instantiate in isolation. We handle this with try/except and skip problematic devices.

### Decision 3: Static parsing vs. live-only

**Chosen: Two-layer approach — static parsing always works, live interaction when EPICS is available.**

**Layer 1 (static, always works):**
- Parse YAML configs with Python's `yaml` module — no ophyd import needed
- Parse Python source with `ast` module to extract class hierarchies, component names, docstrings
- Parse plan files to extract function signatures and docstrings
- This powers `list_devices`, `inspect_device`, and `list_plans` tools even without EPICS

**Layer 2 (live, requires EPICS):**
- Instantiate actual ophyd devices
- Powers `read_device` and `set_device` tools
- Falls back gracefully if EPICS isn't available

**Why both:** The user is developing locally (macOS) where EPICS IOCs may not be running. Static parsing lets them work with device metadata offline. At the beamline, live mode kicks in automatically.

### Decision 4: Safety for write operations

**Chosen: LLM-level guardrails in the system message.**

The system message instructs the agent to describe what it will do before calling `set_device` and to ask for confirmation for large moves. The `set_device` tool itself returns a confirmation string with old → new values.

**Why not code-level confirmation:**
- Tool calls are executed by `tool_worker` without human input (`human_input_mode="NEVER"`)
- Adding a confirmation step would require changing the AG2 conversation flow
- The LLM is already the decision-maker — telling it to be careful is simpler and matches how beamline scientists work (they tell the control system what to do, it does it)

**Tradeoff:** The LLM might occasionally call `set_device` without confirming. For a production deployment at a real beamline, we'd want code-level safety (move limits, rate limiting, confirmation protocol). For this first implementation, system message guardrails are sufficient.

### Decision 5: Ingesting BITS source into ChromaDB

**Chosen: Yes, optionally.**

When data ingestion runs with BITS enabled, we also ingest:
- Device class docstrings + component structure
- Plan function docstrings + signatures
- YAML device definitions

This lets `query_documentation` find device-related information alongside Sphinx docs. The agent can answer "what does the linkam controller do?" from the docstring, and "what is the linkam's current temperature?" from the live `read_device` tool.

---

## Implementation Plan

### Step 1: Add BITS Configuration

**File:** `src/tomobait/config.py`

```python
class BITSConfig(BaseModel):
    """Configuration for BITS instrument integration."""
    enabled: bool = False
    path: str = ""                    # /Users/ecodrea/usaxs-bits
    package_name: str = ""            # usaxs
    src_subdir: str = "src"           # subdirectory with the package
```

Add `bits: BITSConfig` field to `BaitConfig`.

**File:** `config.yaml`

```yaml
bits:
  enabled: true
  path: /Users/ecodrea/usaxs-bits
  package_name: usaxs
  src_subdir: src
```

### Step 2: Create `bits.py` Module

**New file:** `src/tomobait/bits.py`

#### Static functions (no EPICS needed):

| Function | Purpose |
|----------|---------|
| `parse_device_yamls(bits_path, configs_dir)` | Parse all `*.yml` files in configs/ to extract device definitions: `{name, class_path, prefix, labels, kwargs}` |
| `parse_device_source(bits_path, devices_dir)` | AST-parse Python files in devices/ to extract: class names, docstrings, base classes, Component definitions |
| `parse_plans_source(bits_path, plans_dir)` | AST-parse Python files in plans/ to extract: function names, signatures, docstrings |
| `get_device_summary()` | Formatted text summary of all discovered devices |
| `get_plan_summary()` | Formatted text summary of all discovered plans |

#### Live functions (require EPICS):

| Function | Purpose |
|----------|---------|
| `init_bits_live(config)` | Import device classes, instantiate from YAML, store in `_live_devices` dict |
| `read_device_value(name, attr)` | Call `.position` / `.get()` on live device |
| `set_device_value(name, attr, value)` | Call `.put()` / `.move()` on live device |
| `is_live()` | Returns whether live mode is active |

Module-level state:
```python
_device_defs: list[dict] = []       # From YAML parsing
_device_classes: dict = {}           # From AST parsing
_plan_defs: list[dict] = []          # From AST parsing
_live_devices: dict = {}             # Instantiated ophyd objects
_live_mode: bool = False
```

Initialization happens when `bits.py` is first imported with a valid config. Static parsing runs always; live init is attempted but failures are caught and logged.

### Step 3: Register BITS Tools on Agent

**File:** `src/tomobait/agents.py`

Add 5 new tools, conditionally registered when `config.bits.enabled`:

**`list_devices()`** — Returns table of device names, types, PV prefixes, labels. Uses static parsing. Always works.

**`inspect_device(device_name)`** — Returns detailed component tree from AST parsing + live values if available.

**`read_device(device_name, attribute="")`** — Reads live device values. Returns error if not in live mode.

**`set_device(device_name, attribute, value)`** — Sets live device value. Returns old→new confirmation.

**`list_plans()`** — Returns plan names, signatures, and first-line descriptions from parsed source.

Tool schema dictionaries are built dynamically:
```python
tools_list = [query_documentation_tool_dict]
if config.bits.enabled:
    tools_list.extend(bits_tool_dicts)
```

### Step 4: Update System Message & Prompt

**File:** `src/tomobait/agents.py`

When BITS is enabled, append to system message:
```
You also have beamline device tools:
- list_devices: See all available devices
- inspect_device: Get detailed device info
- read_device: Read current device values
- set_device: Set device values (always state what you'll do first)
- list_plans: See available Bluesky plans

Use query_documentation for documentation questions.
Use device tools for real-time device questions.
```

Update `run_agent_chat()` prompt to mention device tools when appropriate.

### Step 5: Ingest BITS Source (Optional)

**File:** `src/tomobait/data_ingestion.py`

Add `ingest_bits_source(config)` function:
- Creates LangChain `Document` objects from parsed device/plan info
- Adds them to ChromaDB alongside Sphinx documentation
- Called from `__main__` when `config.bits.enabled is True`

### Step 6: Dependencies

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
bits = ["ophyd", "apsbits"]
```

`bits.py` uses lazy imports so TomoBait still works without these packages installed.

---

## Files Summary

| File | Action |
|------|--------|
| `src/tomobait/config.py` | Add `BITSConfig` model |
| `src/tomobait/bits.py` | **New** — BITS parsing + live interaction |
| `src/tomobait/agents.py` | Register 5 new tools, update system message |
| `src/tomobait/data_ingestion.py` | Add BITS source ingestion |
| `config.yaml` | Add `bits:` section |
| `pyproject.toml` | Add optional `bits` dependency group |

---

## How to Verify

1. `ruff check . && ruff format .` — no lint errors
2. With `bits.enabled: false` — app starts and behaves identically to before
3. With `bits.enabled: true` and valid path:
   - `list_devices` returns all devices from YAML configs
   - `inspect_device("s_stage")` returns component tree (x, y motors with PV names)
   - `list_plans` returns plan signatures from source parsing
4. With EPICS running:
   - `read_device("s_stage", "x")` returns current motor position
   - `set_device("s_stage", "x", "5.0")` moves the motor and reports result
5. Without EPICS:
   - Static tools work normally
   - `read_device` / `set_device` return clear "EPICS not available" message
6. Chat test: "What devices are on this beamline?" → agent calls `list_devices`
7. Chat test: "Where is the sample stage?" → agent calls `read_device("s_stage")`

---

## Open Questions / Future Work

- **Plan execution:** Not included in this plan. Running Bluesky plans requires a RunEngine, which is a much larger integration. Could be added later as a `run_plan` tool.
- **Device safety limits:** For production use, the `set_device` tool should check software limits before moving motors. The current plan relies on EPICS soft limits and LLM-level caution.
- **Multi-BITS support:** The config supports one BITS installation. Could be extended to multiple beamlines.
- **oregistry integration:** If TomoBait runs in the same Python environment as a Bluesky session, it could use oregistry directly instead of re-instantiating devices. This would be a future optimization.


  Prompt for /skill-creator:

  ▎ Create a skill called ophyd-device-control that enables an agent to fully control any ophyd EPICS device in any BITS (Bluesky Instrument for Tomo Scanning) package from an IPython session. The skill should NOT
  hardcode any specific device names, signals, or PV prefixes — instead it should teach the agent how to dynamically discover and interact with whatever devices are loaded.

  ▎ What it should cover:

  ▎ 1. Device Discovery — How to find what devices exist:
  ▎   - Access oregistry from apsbits.core.instrument_init to list all registered devices
  ▎   - Devices are defined in configs/devices.yml (Guarneri-style YAML: class path, name, prefix, labels)
  ▎   - After startup, device variables are available directly in the IPython namespace
  ▎ 2. Device Exploration — How to inspect any device's structure without knowing it in advance:
  ▎   - device.summary() for human-readable overview
  ▎   - device.component_names to list all signal attributes
  ▎   - device.walk_signals() to recursively walk all signals including nested sub-devices, getting name, PV, kind, connection status
  ▎   - device.prefix for the EPICS PV prefix
  ▎   - signal.pvname for the full EPICS PV name
  ▎   - device._sig_attrs to filter signals by kind (config/normal/omitted/hinted)
  ▎   - device.__class__.__mro__ to understand the inheritance chain
  ▎ 3. Reading Values — Interactive only, no Bluesky plans:
  ▎   - device.signal_name.get() for single values
  ▎   - device.signal_name.get(as_string=True) for string-type PVs
  ▎   - device.read() for all kind="normal" signals
  ▎   - device.read_configuration() for all kind="config" signals
  ▎   - device.describe() and device.describe_configuration() for metadata
  ▎ 4. Setting Values — Interactive only:
  ▎   - device.signal_name.put(value) — synchronous, blocks until acknowledged
  ▎   - device.signal_name.put(value, wait=False) — fire-and-forget
  ▎   - device.signal_name.set(value) — returns Status object for async tracking
  ▎ 5. Connection Checking:
  ▎   - device.connected — True only if ALL signals connected
  ▎   - device.signal_name.connected — individual signal check
  ▎   - device.wait_for_connection(timeout=5.0) — blocks until connected or raises TimeoutError
  ▎   - Walk signals to find specifically which ones are disconnected
  ▎ 6. Monitoring/Subscriptions:
  ▎   - signal.subscribe(callback) returns a callback ID
  ▎   - signal.unsubscribe(cbid) to stop
  ▎   - signal.clear_sub(signal.SUB_VALUE) to clear all
  ▎ 7. Important ophyd concepts the agent should understand:
  ▎   - Signal kind values: "config" (read once), "normal" (read each scan), "omitted" (not auto-read), "hinted" (primary data)
  ▎   - .read() returns normal+hinted, .read_configuration() returns config signals
  ▎   - Omitted signals must be accessed directly via .get()
  ▎   - EpicsSignal is read-write, EpicsSignalRO is read-only
  ▎   - string=True signals represent enum/text PVs
  ▎   - Device hierarchy via inheritance — child classes can override parent signal kinds
  ▎   - PV naming: {device.prefix}{Component_suffix}

  ▎ Critical constraints:
  ▎ - This skill is for INTERACTIVE ophyd control only — absolutely NO Bluesky plan syntax (yield from, bps.mv, bps.rd, @with_registry, RunEngine)
  ▎ - Must be generic — no hardcoded device names, signal names, or PV prefixes
  ▎ - The agent should always start by discovering what devices exist and exploring their structure before trying to interact
  ▎ - All introspection works WITHOUT a live EPICS connection; only .get(), .put(), and .connected require one

  ▎ When to trigger: When the user mentions ophyd, EPICS, PVs, device signals, .get(), .put(), .read(), .connected, checking values, setting parameters, device status, walk signals, or any beamline device
  interaction.