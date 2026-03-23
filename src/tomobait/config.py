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


class LLMConfig(BaseModel):
    """Configuration for the LLM and agents."""

    provider: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable name containing the API key",
    )
    argo_base_url: str = Field(
        default="https://apps-dev.inside.anl.gov/argoapi/v1/",
        description="Base URL for the LLM API (if applicable)",
    )
    api_key: str = Field(
        default="",
        description="API key for the LLM provider",
    )
    model: str = Field(
        default="gemini-2.5-flash", description="Model name (e.g., gemini-2.5-flash)"
    )
    api_type: str = Field(
        default="google", description="API type (google, openai, etc.)"
    )
    system_message: str = Field(
        default=(
            "You are an expert on this project's documentation. "
            "A user will ask a question. Your 'query_documentation' tool "
            "will provide you with the *only* relevant context. "
            "**You must answer the user's question based *only* on that context.** "
            "If the context is not sufficient, say so. Do not make up answers."
        ),
        description="System message for the documentation expert agent",
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
        description="Model name (HuggingFace model or Argo model like 'ada002')",
    )
    device: str = Field(
        default="cpu",
        description="Device for local embeddings: 'cpu', 'cuda', 'mps', or 'auto'",
    )
    argo_user: Optional[str] = Field(
        default=None,
        description="ANL username for Argo API (if provider is 'argo')",
    )
    argo_base_url: str = Field(
        default="https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/",
        description="Base URL for Argo embedding API",
    )


class BITSConfig(BaseModel):
    """Configuration for BITS instrument integration."""

    enabled: bool = Field(default=True, description="Enable BITS integration")
    path: str = Field(default="", description="Absolute path to the BITS root folder")
    package_name: str = Field(
        default="", description="Python package name (e.g., 'tomo_2bm')"
    )
    src_subdir: str = Field(
        default="src",
        description="Subdirectory containing Python packages",
    )


class BaitConfig(BaseSettings):
    """Main configuration for TomoBait, loaded from config.yaml."""

    model_config = SettingsConfigDict(
        # Ensure the default config.yaml is loaded if it exists
        yaml_file="config.yaml"
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    documentation: DocumentationSourceConfig = Field(
        default_factory=DocumentationSourceConfig
    )
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
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
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
