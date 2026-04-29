# TODOS

## Safety model for device write operations
**Status:** Open
**Priority:** High (blocks write operations)
**Depends on:** ophyd-websocket integration complete

Before enabling `set_device()` in the device agent, design and implement a safety model:
- Confirmation prompt before hardware writes? (LLM suggests action, user confirms)
- Allowlist of safe devices/operations?
- Read-only by default, opt-in for write access?
- What happens if the LLM hallucinates a device name or value?

This is Open Question #3 from the design doc. Read-only operations (`read_device()`) can ship without this. Write operations (`set_device()`) must not ship until this is resolved.

**Context:** An LLM issuing commands to real synchrotron hardware without human approval could damage equipment or invalidate experiments. The safety model needs to balance usability (scientists want fast device control) with protection (prevent accidental or hallucinated commands).

## Phase 1b: BITS source introspection
**Status:** Open
**Priority:** Medium (Phase 1b)
**Depends on:** Phase 1a 2nd-beamline test confirms BITS folder format is stable across at least 2 installations

Parse the BITS installation into an in-memory `DeviceInfo` data model so the device agent can answer questions beyond what `device_skills.md` says (e.g., "what plans take a sample stage parameter?").

Files to parse (per design doc lines 296-323):
- `configs/*.yml` — device instantiation configs (PyYAML)
- `devices/*.py` — ophyd Device class definitions (Python `ast` for class hierarchies + Component fields)
- `plans/*.py` — Bluesky plan signatures + docstrings (`ast`)
- `skills/*.md` — already loaded (read as plain text)

**Pros:** dynamic device discovery without manual `device_skills.md` updates; agent can introspect plans and devices the operator hasn't documented.
**Cons:** AST parsing is fragile against version drift; BITS folder schema may vary across beamlines.
**Where to start:** new module `src/bait/bits/introspection.py` returning `list[DeviceInfo]`. Wire into `bits_agent_node` system prompt. Validate against tomo-bits AND the second beamline's BITS before relying on the format.

## Phase 1b: ophyd-websocket read-only integration
**Status:** Open
**Priority:** Medium (Phase 1b)
**Depends on:** Phase 1a complete; one beamline workstation with ophyd-websocket server running

Implement `read_device(device_name: str) -> str` as an agent tool that subscribes to the ophyd-websocket server and returns the current value + metadata. Read-only — no `set_device()`. Write operations are blocked on the safety-model TODO above.

Per design doc lines 146-191:
- WebSocket endpoint: `ws://{host}:{port}/api/v1/device-socket`
- Send `{"action": "subscribe", "device": "name"}`, receive `{"device", "value", "timestamp", "connected", "read_access", "write_access"}`
- New config block: `ophyd_websocket: {url, enabled}` — when `enabled: false` or connection fails, agent falls back to `device_skills.md` knowledge only.

**Pros:** device agent moves from "tells you what's possible" to "tells you the current value." Big UX win for the scientist asking "where is the sample stage right now?"
**Cons:** WebSocket lifecycle (connect, subscribe, timeout, cleanup) adds operational complexity; depends on the ophyd-websocket server actually running on the workstation.
**Where to start:** new module `src/bait/devices/websocket_client.py` with the subscribe/read flow. Add `ophyd_websocket` block to `BaitConfig`. Register `read_device` as a tool in `bits_agent`.

## Router classification accuracy eval
**Status:** Open
**Priority:** Medium (quality gate for prompt changes)
**Depends on:** Phase 1a complete; ideally one beamline beyond tomo for cross-domain test cases

Build a parametrized pytest suite (or LangSmith eval) covering router classification accuracy: at minimum 5 documentation questions, 5 device questions, 5 ambiguous, 5 nonsense. Assert exact match for clear cases; collect-only for ambiguous (track over time).

**Pros:** any future change to a system prompt (router, doc_agent, bits_agent) is gated by this; baseline for measuring prompt improvements.
**Cons:** ambiguous/nonsense cases are subjective and need maintenance; running it costs Argo tokens per test invocation; needs Argo creds in CI.
**Where to start:** `tests/test_router_eval.py` parametrized with the 20 cases; mark `@pytest.mark.eval` so it only runs with an explicit flag. Document the run command in CLAUDE.md.

## CI/CD distribution pipeline for `bait` package
**Status:** Open
**Priority:** Low (manual install fine for first 2 beamlines)
**Depends on:** decision on publish target (PyPI / private index / GitHub Releases) + enough beamline adoption to justify the overhead

Build a GitHub Actions workflow that builds the wheel, runs tests on every push, and publishes on tag. Each beamline upgrades by version pin (`uv pip install bait==0.2.0`) instead of git URL.

**Pros:** reproducible deployments; clear release cadence; rollback is `pip install bait==<previous>`.
**Cons:** needs a publish target decided (PyPI is public — is that OK? Internal index needs hosting; GitHub Releases is intermediate); release process means somebody owns version bumps.
**Where to start:** `.github/workflows/release.yml` triggered on tag push, builds wheel via `uv build`, attaches to GitHub Release. Decide PyPI vs GitHub Releases vs internal index based on who needs to install (only ANL? external collaborators?).
