"""
Centralized configuration management for TomoBait, using pydantic-settings.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


# --- Pydantic Models for Configuration Sections ---
class ProjectConfig(BaseModel):
    """Configuration for project identity and base directories."""

    name: str = Field(
        default="tomo",
        description="Project identifier name (used in directory naming)",
    )
    data_dir: str = Field(
        default=".bait-tomo",
        description="Base directory for all project data",
    )


class DocumentationSourceConfig(BaseModel):
    """Configuration for documentation sources."""

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
    """Configuration for the document retriever."""

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

    system_prompt: str = Field(description="System prompt for this agent")
    model: str = Field(
        default="claudeopus46",
        description="Model name for this agent",
    )
    api_type: str = Field(
        default="anthropic",
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


class AgentsConfig(BaseModel):
    """Configuration for all agents in the system."""

    router: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            system_prompt=(
                "You are a question classifier for a beamline instrument system.\n\n"
                "Classify the user's question into exactly one category:\n"
                '- "documentation": Questions about beamline documentation, '
                "experimental procedures, how-to guides, configuration, "
                "or general usage.\n"
                '- "device": Questions about ophyd devices, EPICS PVs, signals, '
                "motors, detectors, shutters, scan parameters, Bluesky plans, "
                "or hardware interaction.\n\n"
                "Respond with ONLY the category name, nothing else."
            ),
        )
    )
    doc_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            system_prompt=(
                "You are an expert on this project's documentation. "
                "When answering questions: "
                "1. Answer based *only* on the context from your "
                "'query_documentation' tool. "
                "2. Provide concise but complete responses (2-3 paragraphs). "
                "3. For 'how to' questions, provide step-by-step numbered "
                "instructions. "
                "4. Include relevant source links from the context. "
                "5. If the context is insufficient, say so. "
                "Do not make up answers."
            ),
        )
    )
    bits_agent: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            system_prompt=(
                "You are an expert on BITS (Beamline Instrument and Tool Suite) "
                "devices, ophyd signals, EPICS PVs, scan parameters, and Bluesky "
                "plans for this beamline.\n\n"
                "Answer questions about devices, their signals, PV names, scan "
                "configuration, and how to interact with hardware using ophyd "
                "and Bluesky.\n\n"
                "Base your answers on the device reference below. If the reference "
                "does not contain enough information, say so. Do not invent PV "
                "names or device attributes."
            ),
        )
    )


class TextProcessingConfig(BaseModel):
    """Configuration for document text processing."""

    chunk_size: int = Field(
        default=1000, description="Size of text chunks in characters", ge=100, le=5000
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between chunks in characters", ge=0, le=1000
    )


class ServerConfig(BaseModel):
    """Configuration for server settings."""

    backend_host: str = Field(default="127.0.0.1", description="Backend server host")
    backend_port: int = Field(default=8001, description="Backend server port")
    frontend_host: str = Field(default="0.0.0.0", description="Frontend server host")
    frontend_port: int = Field(default=8000, description="Frontend server port")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""

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
    """Configuration for BITS instrument integration."""

    path: str = Field(default="", description="Absolute path to the BITS root folder")
    instrument_name: str = Field(
        default="", description="Python package name (e.g., 'tomo_2bm')"
    )


class BaitConfig(BaseSettings):
    """Main configuration for TomoBait, loaded from config.yaml."""

    model_config = SettingsConfigDict(yaml_file="config.yaml")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    documentation: DocumentationSourceConfig = Field(
        default_factory=DocumentationSourceConfig
    )
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    bits: BITSConfig = Field(default_factory=BITSConfig)

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
        """Get the resolved data directory path."""
        return Path(self.project.data_dir)

    @computed_field
    @property
    def docs_output_dir(self) -> Path:
        """Get the resolved documentation output directory path."""
        return self.data_dir / "documentation"

    @computed_field
    @property
    def db_path(self) -> Path:
        """Get the resolved ChromaDB path."""
        return self.data_dir / "chroma_db"

    @computed_field
    @property
    def chat_history_dir(self) -> Path:
        """Get the resolved chat history directory path."""
        return self.data_dir / "chat_history"

    @computed_field
    @property
    def bits_skills_dir(self) -> Path:
        """Get the BITS root folder where skills files are written."""
        return Path(self.bits.path) if self.bits.path else Path(".")

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
        """Define the source loading priority, using YAML as the primary source."""
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
