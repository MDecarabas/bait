"""Tests for configuration loading and path resolution."""

import os
import tempfile
from pathlib import Path

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


def test_get_agent_llm_settings_defaults():
    """Agent LLM settings should read from the agent's own config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        from tomobait.config import BaitConfig

        config = BaitConfig()
        settings = config.get_agent_llm_settings("router")
        assert settings["model"] == config.agents.router.model
        assert settings["api_type"] == config.agents.router.api_type
        assert settings["api_type"] == "anthropic"
        assert settings["model"] == "claudeopus46"


def test_get_agent_llm_settings_per_agent():
    """Each agent can have its own LLM settings in config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        config_data = {
            "agents": {
                "router": {
                    "system_prompt": "test",
                    "model": "custom-model",
                    "api_type": "openai",
                    "api_key": "testuser",
                    "argo_base_url": "http://custom",
                }
            },
        }
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        from tomobait.config import BaitConfig

        config = BaitConfig()
        settings = config.get_agent_llm_settings("router")
        assert settings["model"] == "custom-model"
        assert settings["api_type"] == "openai"
        assert settings["api_key"] == "testuser"
        assert settings["argo_base_url"] == "http://custom"


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
