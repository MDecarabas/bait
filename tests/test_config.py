"""Tests for configuration loading and path resolution."""

from pathlib import Path

import pytest
import yaml


def test_required_fields_loaded(write_config):
    """A minimal valid config loads with all defaults applied."""
    write_config()
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.project.name == "test"
    assert config.server.backend_port == 8001
    assert config.server.frontend_port == 8000
    assert config.retriever.k == 3


def test_get_config_returns_singleton(write_config):
    """get_config() returns the same instance across calls."""
    write_config()
    from bait.config import get_config

    assert get_config() is get_config()


def test_path_resolution_defaults_to_bait_dot_name(write_config):
    """data_dir derives from project.name when not explicitly set."""
    write_config()
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.data_dir == Path(".bait-test")
    assert config.db_path == config.data_dir / "chroma_db"
    assert config.docs_output_dir == config.data_dir / "documentation"
    assert config.chat_history_dir == config.data_dir / "chat_history"


def test_path_resolution_explicit_data_dir(write_config):
    """An explicit project.data_dir wins over the .bait-{name} default."""
    write_config(extra={"project": {"name": "test", "data_dir": ".bait-explicit"}})
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.data_dir == Path(".bait-explicit")


def test_get_agent_llm_settings(write_config):
    """Per-agent overrides are exposed via get_agent_llm_settings."""
    write_config(
        extra={
            "agents": {
                "router": {
                    "system_prompt": "test",
                    "model": "custom-model",
                    "api_type": "openai",
                    "api_key": "testuser",
                    "argo_base_url": "http://custom",
                }
            }
        }
    )
    from bait.config import BaitConfig

    config = BaitConfig()
    settings = config.get_agent_llm_settings("router")
    assert settings["model"] == "custom-model"
    assert settings["api_type"] == "openai"
    assert settings["api_key"] == "testuser"
    assert settings["argo_base_url"] == "http://custom"


def test_bits_skills_dir_default(write_config):
    """No override → {bits.path}/.claude/skills/ophyd-device-control."""
    write_config(extra={"bits": {"path": "/some/path"}})
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.bits_skills_dir == Path(
        "/some/path/.claude/skills/ophyd-device-control"
    )


def test_bits_skills_dir_override(write_config):
    """`bits.skills_dir` overrides the computed default."""
    write_config(extra={"bits": {"path": "/some/path", "skills_dir": "/custom/skills"}})
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.bits_skills_dir == Path("/custom/skills")


def test_queueserver_script_default(write_config):
    write_config(extra={"bits": {"path": "/some/path", "instrument_name": "tomo_2bm"}})
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.queueserver_script == Path("/some/path/scripts/tomo_2bm_qs_host.sh")


def test_queueserver_script_override(write_config):
    write_config(
        extra={"bits": {"path": "/some/path", "queueserver_script": "/x/qs.sh"}}
    )
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.queueserver_script == Path("/x/qs.sh")


def test_oas_startup_file_default(write_config):
    """No override → {bits.path}/src/{instrument_name}/startup.py (BITS convention)."""
    write_config(
        extra={"bits": {"path": "/some/path", "instrument_name": "tomo_2bm"}}
    )
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.oas_startup_file == Path("/some/path/src/tomo_2bm/startup.py")


def test_oas_startup_file_override(write_config):
    write_config(
        extra={"bits": {"path": "/some/path", "oas_startup_file": "/x/start.py"}}
    )
    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.oas_startup_file == Path("/x/start.py")


# --- BAIT_CONFIG env override ---


def test_bait_config_env_points_to_other_dir(tmp_path, monkeypatch):
    """BAIT_CONFIG resolves to a yaml in any directory, not just cwd."""
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    cfg = other_dir / "alt.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "project": {"name": "elsewhere"},
                "bits": {"path": ""},
                "agents": {
                    "router": {"system_prompt": "x"},
                    "doc_agent": {"system_prompt": "x"},
                    "bits_agent": {"system_prompt": "x"},
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)  # cwd has no config.yaml
    monkeypatch.setenv("BAIT_CONFIG", str(cfg))

    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.project.name == "elsewhere"


def test_bait_config_env_missing_file_raises(tmp_path, monkeypatch):
    """BAIT_CONFIG pointing at a missing file raises with the offending path."""
    bogus = tmp_path / "nope" / "no.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BAIT_CONFIG", str(bogus))

    from bait.config import BaitConfig

    with pytest.raises(FileNotFoundError, match=str(bogus)):
        BaitConfig()


def test_no_config_anywhere_raises(tmp_path, monkeypatch):
    """R4: no BAIT_CONFIG and no ./config.yaml → loud error, not silent defaults."""
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    from bait.config import BaitConfig

    with pytest.raises(FileNotFoundError, match="No Bait configuration found"):
        BaitConfig()


def test_bits_root_configs_dir_fallback(tmp_path, monkeypatch):
    """No cwd config.yaml, but configs/bait_config.yaml present → resolved."""
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bait_config.yaml").write_text(
        yaml.dump(
            {
                "project": {"name": "from-bits-configs"},
                "bits": {"path": ""},
                "agents": {
                    "router": {"system_prompt": "x"},
                    "doc_agent": {"system_prompt": "x"},
                    "bits_agent": {"system_prompt": "x"},
                },
            }
        )
    )

    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.project.name == "from-bits-configs"


def test_cwd_config_yaml_preferred_over_bits_configs(tmp_path, monkeypatch):
    """If both ./config.yaml and ./configs/bait_config.yaml exist, ./config.yaml wins."""
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    def _minimal(name):
        return yaml.dump(
            {
                "project": {"name": name},
                "bits": {"path": ""},
                "agents": {
                    "router": {"system_prompt": "x"},
                    "doc_agent": {"system_prompt": "x"},
                    "bits_agent": {"system_prompt": "x"},
                },
            }
        )

    (tmp_path / "config.yaml").write_text(_minimal("from-cwd"))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bait_config.yaml").write_text(_minimal("from-bits-configs"))

    from bait.config import BaitConfig

    config = BaitConfig()
    assert config.project.name == "from-cwd"


# --- extra="forbid" enforcement ---


def test_unknown_top_level_key_rejected(write_config):
    """A typo at the top level surfaces as a ValidationError naming the key."""
    write_config(extra={"embeddings": {}})  # typo: should be "embedding"
    from pydantic import ValidationError

    from bait.config import BaitConfig

    with pytest.raises(ValidationError, match="embeddings"):
        BaitConfig()


def test_unknown_nested_key_in_embedding_rejected(write_config):
    """A typo inside embedding: surfaces as ValidationError naming the key."""
    write_config(extra={"embedding": {"providr": "anl_argo"}})  # typo
    from pydantic import ValidationError

    from bait.config import BaitConfig

    with pytest.raises(ValidationError, match="providr"):
        BaitConfig()


def test_unknown_nested_key_in_router_rejected(write_config):
    """A typo inside agents.router: surfaces as ValidationError naming the key."""
    write_config(
        extra={
            "agents": {
                "router": {"system_prompt": "x", "api_typ": "openai"}  # typo
            }
        }
    )
    from pydantic import ValidationError

    from bait.config import BaitConfig

    with pytest.raises(ValidationError, match="api_typ"):
        BaitConfig()


# --- Required fields ---


def test_missing_project_name_raises(tmp_path, monkeypatch):
    """project.name has no default — missing it raises ValidationError."""
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "project": {},
                "bits": {"path": ""},
                "agents": {
                    "router": {"system_prompt": "x"},
                    "doc_agent": {"system_prompt": "x"},
                    "bits_agent": {"system_prompt": "x"},
                },
            }
        )
    )
    from pydantic import ValidationError

    from bait.config import BaitConfig

    with pytest.raises(ValidationError, match="name"):
        BaitConfig()


def test_missing_bits_path_raises(tmp_path, monkeypatch):
    """bits.path has no default — missing it raises ValidationError."""
    monkeypatch.delenv("BAIT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "project": {"name": "t"},
                "bits": {},
                "agents": {
                    "router": {"system_prompt": "x"},
                    "doc_agent": {"system_prompt": "x"},
                    "bits_agent": {"system_prompt": "x"},
                },
            }
        )
    )
    from pydantic import ValidationError

    from bait.config import BaitConfig

    with pytest.raises(ValidationError, match="path"):
        BaitConfig()
