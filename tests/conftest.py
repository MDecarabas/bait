"""Shared test fixtures for Bait."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# Minimal valid config: every required field present, every default left alone.
# Tests can override sections by passing extras to ``write_config``.
# Per-agent prompts are tagged so test scripts can dispatch by substring match.
_BASE_CONFIG: Dict[str, Any] = {
    "project": {"name": "test"},
    "bits": {"path": ""},
    # Tests must not spawn the OAS server subprocess by default.
    "ophyd_websocket": {"enabled": False, "auto_start": False},
    "agents": {
        "router": {"system_prompt": "test classifier prompt"},
        "doc_agent": {"system_prompt": "test expert on this project prompt"},
        "bits_agent": {"system_prompt": "test expert on BITS prompt"},
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursive dict merge — overrides win, lists/scalars are replaced."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a callable that writes a config.yaml inside ``tmp_path``.

    Tests can call ``write_config(extra={"bits": {"path": "/x"}})`` to
    override the defaults. Always chdirs into ``tmp_path`` and clears
    ``BAIT_CONFIG`` so the resolver picks up the local file.
    """
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    def _write(extra: dict | None = None) -> Path:
        merged = _deep_merge(_BASE_CONFIG, extra or {})
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(merged))
        return path

    return _write


@pytest.fixture
def base_config(write_config):
    """Convenience: minimal valid config, returned as a fresh BaitConfig."""
    write_config()
    from bait.config import BaitConfig

    return BaitConfig()


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset all process-wide singletons between tests."""
    yield
    try:
        from bait import agents
        from bait.config import reset_config_cache
        from bait.utils import _client_cache

        reset_config_cache()
        agents._graph = None
        _client_cache.clear()
    except Exception:
        # Modules may not be imported yet during collection; that's fine.
        pass
