"""Tests for the apsbits-aware device harvesting in bait.device_io.

The vendored device_registry's load_startup_files() scans the loaded
module's top-level namespace for Device / EpicsSignal instances. apsbits's
make_devices() instead writes Devices into a guarneri Instrument's registry,
so the upstream scanner misses them. _harvest_apsbits_oregistry() bridges
that gap.
"""

from __future__ import annotations

import sys
import types

import pytest
from ophyd.sim import motor as sim_motor

from bait.device_io import _harvest_apsbits_oregistry
from bait.ophyd_websocket.device_registry import device_registry


class _FakeOregistry:
    """Mimics guarneri's Instrument.devices: ordered names + getitem."""

    def __init__(self, devices: dict):
        self._devices = devices

    @property
    def device_names(self):
        return list(self._devices)

    def __getitem__(self, key):
        return self._devices[key]


@pytest.fixture(autouse=True)
def clear_registry():
    device_registry.clear()
    yield
    device_registry.clear()


def _install_apsbits(monkeypatch, instrument):
    fake = types.ModuleType("apsbits.core.instrument_init")
    fake._instrument = instrument
    monkeypatch.setitem(sys.modules, "apsbits.core.instrument_init", fake)


def test_harvest_skips_when_apsbits_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "apsbits.core.instrument_init", raising=False)
    assert _harvest_apsbits_oregistry() == 0
    assert device_registry.list_devices() == []


def test_harvest_skips_when_instrument_unset(monkeypatch):
    _install_apsbits(monkeypatch, instrument=None)
    assert _harvest_apsbits_oregistry() == 0


def test_harvest_skips_when_instrument_has_no_devices_attr(monkeypatch):
    _install_apsbits(monkeypatch, instrument=types.SimpleNamespace())
    assert _harvest_apsbits_oregistry() == 0


def test_harvest_adds_new_devices(monkeypatch):
    instrument = types.SimpleNamespace(
        devices=_FakeOregistry({"motor1": sim_motor})
    )
    _install_apsbits(monkeypatch, instrument)

    added = _harvest_apsbits_oregistry()

    assert added == 1
    assert device_registry.list_devices() == ["motor1"]
    assert device_registry.get_device("motor1") is sim_motor


def test_harvest_skips_already_registered(monkeypatch):
    instrument = types.SimpleNamespace(
        devices=_FakeOregistry({"motor1": sim_motor})
    )
    _install_apsbits(monkeypatch, instrument)
    device_registry.add_device("motor1", sim_motor)

    added = _harvest_apsbits_oregistry()

    assert added == 0
    assert device_registry.list_devices() == ["motor1"]


def test_harvest_warns_and_continues_on_invalid_device(monkeypatch, caplog):
    """add_device raises ValueError on non-Device objects; harvester should
    skip and keep going rather than abort the whole load."""
    instrument = types.SimpleNamespace(
        devices=_FakeOregistry({"bad": object(), "good": sim_motor})
    )
    _install_apsbits(monkeypatch, instrument)

    with caplog.at_level("WARNING"):
        added = _harvest_apsbits_oregistry()

    assert added == 1
    assert device_registry.list_devices() == ["good"]
    assert any("bad" in rec.message for rec in caplog.records)
