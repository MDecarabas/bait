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
