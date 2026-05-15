"""Default OAS startup file: simulated ophyd devices for development.

The ophyd-websocket OAS server walks `vars(module)` of this file and registers
any module-level Device or EpicsSignal instance. Anything not EPICS-connected
(like the ophyd.sim devices below) lets the system come up green without a
real IOC.

Override `bits.oas_startup_file` in config.yaml to point at your own startup
file with EPICS-connected devices for real beamline use.
"""

from ophyd.sim import noisy_det as sim_det
from ophyd.sim import motor as sim_motor


__all__ = ["motor", "noisy_det"]
