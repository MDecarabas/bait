"""Centralized configuration management for Bait, using pydantic-settings."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


def _resolve_config_path() -> Path:
    """Resolve the config.yaml location.

    Order:
      1. ``BAIT_CONFIG`` env var (must point to an existing file).
      2. ``./config.yaml`` in the current working directory.
      3. ``./configs/bait_config.yaml`` (BITS-repo-root convention).

    Raises ``FileNotFoundError`` if none are found, so a missing config is
    a loud startup failure instead of silent fallback to defaults.
    """
    env_path = os.environ.get("BAIT_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"BAIT_CONFIG points to a missing file: {p}")
        return p
    cwd_config = Path("config.yaml")
    if cwd_config.is_file():
        return cwd_config
    bits_config = Path("configs") / "bait_config.yaml"
    if bits_config.is_file():
        return bits_config
    raise FileNotFoundError(
        "No Bait configuration found. Either set "
        "BAIT_CONFIG=/abs/path/config.yaml, place config.yaml in the "
        "current working directory, or create configs/bait_config.yaml "
        "(BITS-repo-root convention)."
    )


# --- Pydantic Models for Configuration Sections ---
class ProjectConfig(BaseModel):
    """Project identity and base directories."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Project identifier name (used in directory naming)",
    )
    data_dir: Optional[str] = Field(
        default=None,
        description=("Base directory for all project data. Defaults to .bait-{name}."),
    )


class DocumentationSourceConfig(BaseModel):
    """Documentation source list."""

    model_config = ConfigDict(extra="forbid")

    git_repos: List[str] = Field(
        default_factory=list,
        description="List of Git repository URLs to clone and index",
    )
    local_folders: List[str] = Field(
        default_factory=list, description="List of local folder paths to index"
    )
    resources: Optional[Dict] = Field(
        default=None,
        description="Reference resources (beamlines, software, organizations, etc.)",
    )


class RetrieverConfig(BaseModel):
    """Document retriever configuration."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(
        default=3, description="Number of documents to retrieve per query", ge=1, le=20
    )
    search_type: str = Field(
        default="similarity",
        description="Search type: similarity, mmr, or similarity_score_threshold",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum relevance score (for similarity_score_threshold)",
        ge=0.0,
        le=1.0,
    )


class AgentConfig(BaseModel):
    """Per-agent configuration: prompt, token limit, and LLM settings."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(description="System prompt for this agent")
    model: str = Field(
        default="claudeopus46",
        description="Model name for this agent",
    )
    api_type: str = Field(
        default="openai",
        description="API protocol: 'openai' or 'anthropic'",
    )
    api_key: str = Field(
        default="",
        description="API key (ANL username for Argo)",
    )
    argo_base_url: str = Field(
        default="https://apps.inside.anl.gov/argoapi/v1",
        description="Base URL for the LLM API",
    )
    max_tokens: int = Field(
        default=4096,
        description="Max tokens for this agent's responses",
        ge=1,
    )


class AgentsConfig(BaseModel):
    """Configuration for all agents in the system."""

    model_config = ConfigDict(extra="forbid")

    router: AgentConfig
    doc_agent: AgentConfig
    bits_agent: AgentConfig


class TextProcessingConfig(BaseModel):
    """Document text processing configuration."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(
        default=1000, description="Size of text chunks in characters", ge=100, le=5000
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks in characters", ge=0, le=1000
    )


class ServerConfig(BaseModel):
    """Server settings."""

    model_config = ConfigDict(extra="forbid")

    backend_host: str = Field(default="127.0.0.1", description="Backend server host")
    backend_port: int = Field(default=8001, description="Backend server port")
    frontend_host: str = Field(default="0.0.0.0", description="Frontend server host")
    frontend_port: int = Field(default=8000, description="Frontend server port")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(
        default="",
        description="API key for the embedding provider",
    )
    provider: str = Field(
        default="huggingface",
        description="Embedding provider: 'huggingface' or 'anl_argo'",
    )
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name (HuggingFace model or Argo model)",
    )
    device: str = Field(
        default="cpu",
        description="Device for local embeddings: 'cpu', 'cuda', 'mps', or 'auto'",
    )
    argo_user: Optional[str] = Field(
        default=None,
        description="ANL username for Argo API (if provider is 'anl_argo')",
    )
    argo_base_url: str = Field(
        default="https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/",
        description="Base URL for Argo embedding API",
    )


class BITSConfig(BaseModel):
    """BITS instrument integration."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Absolute path to the BITS root folder")
    instrument_name: str = Field(
        default="", description="Python package name (e.g., 'tomo_2bm')"
    )
    skills_dir: Optional[str] = Field(
        default=None,
        description=(
            "Override path to the device-skills directory. "
            "Defaults to {path}/.claude/skills/ophyd-device-control."
        ),
    )
    queueserver_script: Optional[str] = Field(
        default=None,
        description=(
            "Override path to the queue-server start script. "
            "Defaults to {path}/scripts/{instrument_name}_qs_host.sh."
        ),
    )
    oas_startup_file: Optional[str] = Field(
        default=None,
        description=(
            "Override path to the ophyd-websocket startup .py file. "
            "Defaults to bait's bundled default_oas_startup.py (sim devices)."
        ),
    )


class OphydWebsocketConfig(BaseModel):
    """ophyd-websocket OAS server settings (server lifecycle, not BITS paths)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Whether to enable OAS at all")
    auto_start: bool = Field(
        default=True,
        description="If true, spawn the OAS server on startup when not reachable",
    )
    repo_path: str = Field(
        default="/Users/ecodrea/ophyd-websocket",
        description="Filesystem path to the ophyd-websocket repository checkout",
    )
    host: str = Field(default="localhost", description="OAS server host")
    port: int = Field(
        default=8002,
        description="OAS server port (NOT 8001 — that conflicts with bait backend)",
    )
    require_qserver: bool = Field(
        default=True,
        description="OAS strict mode: block writes if queue server not reachable",
    )
    python_executable: Optional[str] = Field(
        default=None,
        description=(
            "Python interpreter to use when spawning the OAS subprocess. "
            "Defaults to sys.executable (bait's venv). Override to point at "
            "the conda env that has ophyd installed."
        ),
    )


class BaitConfig(BaseSettings):
    """Main configuration for Bait, loaded from config.yaml."""

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        extra="forbid",
    )

    project: ProjectConfig
    bits: BITSConfig
    ophyd_websocket: OphydWebsocketConfig = Field(default_factory=OphydWebsocketConfig)
    documentation: DocumentationSourceConfig = Field(
        default_factory=DocumentationSourceConfig
    )
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    agents: AgentsConfig
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def model_post_init(self, __context) -> None:
        """Create all necessary directories after the model is initialized."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)

    # --- Computed Path Properties ---

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Resolved data directory path. Defaults to .bait-{project.name}."""
        if self.project.data_dir:
            return Path(self.project.data_dir)
        return Path(f".bait-{self.project.name}")

    @computed_field
    @property
    def docs_output_dir(self) -> Path:
        """Resolved documentation output directory path."""
        return self.data_dir / "documentation"

    @computed_field
    @property
    def db_path(self) -> Path:
        """Resolved ChromaDB path."""
        return self.data_dir / "chroma_db"

    @computed_field
    @property
    def chat_history_dir(self) -> Path:
        """Resolved chat history directory path."""
        return self.data_dir / "chat_history"

    @computed_field
    @property
    def bits_skills_dir(self) -> Path:
        """Directory holding device-skill markdown files loaded into bits_agent."""
        if self.bits.skills_dir:
            return Path(self.bits.skills_dir).expanduser()
        return Path(self.bits.path) / ".claude" / "skills" / "ophyd-device-control"

    @computed_field
    @property
    def queueserver_script(self) -> Path:
        """Path to the queue-server start script."""
        if self.bits.queueserver_script:
            return Path(self.bits.queueserver_script).expanduser()
        return (
            Path(self.bits.path) / "scripts" / f"{self.bits.instrument_name}_qs_host.sh"
        )

    @computed_field
    @property
    def oas_startup_file(self) -> Path:
        """Path to the .py file the OAS server loads to populate its registry."""
        if self.bits.oas_startup_file:
            return Path(self.bits.oas_startup_file).expanduser()
        # Don't import the module — it imports ophyd, which is only available
        # in the OAS subprocess, not in tomo-bait itself.
        import bait.devices as _devices_pkg

        return Path(_devices_pkg.__file__).parent / "default_oas_startup.py"

    def get_agent_llm_settings(self, agent_name: str) -> dict:
        """Return LLM settings for a given agent."""
        agent_config: AgentConfig = getattr(self.agents, agent_name)
        return {
            "model": agent_config.model,
            "api_type": agent_config.api_type,
            "api_key": agent_config.api_key,
            "argo_base_url": agent_config.argo_base_url,
        }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Resolve the YAML source from BAIT_CONFIG or ./config.yaml.

        Resolution happens on each instantiation (not at class-definition
        time) so tests and runtime config switching both work.
        """
        yaml_path = _resolve_config_path()
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=str(yaml_path)),
            file_secret_settings,
        )


# Process-wide BaitConfig singleton. A plain module global is honest about what
# `@lru_cache(maxsize=1)` was already doing and gives tests one place to reset.

_config: BaitConfig | None = None


def get_config() -> BaitConfig:
    """Return the process-wide BaitConfig, instantiating on first call."""
    global _config
    if _config is None:
        _config = BaitConfig()
    return _config


def reset_config_cache() -> None:
    """Test-only: drop the cached BaitConfig so the next call re-reads YAML."""
    global _config
    _config = None
