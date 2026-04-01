"""Tests for configuration loading and path resolution."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml


def test_bait_config_defaults():
    """BaitConfig should provide sensible defaults without a config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        assert config.project.name == "tomo"
        assert config.server.backend_port == 8001
        assert config.server.frontend_port == 8000
        assert config.retriever.k == 3


def test_bait_config_path_resolution():
    """Computed paths should be derived from project settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        assert config.data_dir == Path(config.project.data_dir)
        assert config.db_path == config.data_dir / "chroma_db"
        assert config.docs_output_dir == config.data_dir / "documentation"
        assert config.chat_history_dir == config.data_dir / "chat_history"


def test_get_agent_llm_settings_uses_defaults():
    """Agent LLM settings should fall back to global llm config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        settings = config.get_agent_llm_settings("router")
        assert settings["model"] == config.llm.model
        assert settings["api_type"] == config.llm.api_type


def test_get_agent_llm_settings_with_override():
    """Agent-specific overrides should take precedence over global settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        config_data = {
            "llm": {"model": "default-model", "api_type": "openai", "api_key": "key1", "argo_base_url": "http://default"},
            "agents": {
                "router": {
                    "system_prompt": "test",
                    "model": "custom-model",
                }
            },
        }
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        from tomobait.config import BaitConfig

        config = BaitConfig()
        settings = config.get_agent_llm_settings("router")
        assert settings["model"] == "custom-model"
        assert settings["api_key"] == "key1"  # falls back to global


def test_bits_skills_dir_with_path():
    """bits_skills_dir should return the configured BITS path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        config.bits.path = "/some/path"
        assert config.bits_skills_dir == Path("/some/path")


def test_bits_skills_dir_empty():
    """bits_skills_dir should return cwd when no path is configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        config.bits.path = ""
        assert config.bits_skills_dir == Path(".")
