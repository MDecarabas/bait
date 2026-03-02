"""
Centralized configuration management for TomoBait, using pydantic-settings.
"""

import shutil
from datetime import datetime
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
    base_url: str = Field(
        default= "https://apps-dev.inside.anl.gov/argoapi/v1/",
        description="Base URL for the LLM API (if applicable)"
    )
    api_key: str = Field(
        default="ecodrea",
        description="api key itself",
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
    anl_api_url: Optional[str] = Field(
        default=None,
        description="ANL Argo API endpoint URL (only for api_type='anl_argo')",
    )
    anl_user: Optional[str] = Field(
        default=None,
        description="ANL username for API requests (only for api_type='anl_argo')",
    )
    anl_model: Optional[str] = Field(
        default=None,
        description="ANL model name (only for api_type='anl_argo')",
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

    provider: str = Field(
        default="huggingface",
        description="Embedding provider: 'huggingface' (local) or 'argo' (ANL Argo API)",
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

    def model_post_init(self, __context) -> None:
        """Create all necessary directories after the model is initialized."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

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


# --- Standalone Utility Functions ---


def backup_config(path: str = "config.yaml") -> str:
    """
    Backup current config file with a timestamp.
    Returns the backup file path.
    """
    config_path = Path(path)
    if not config_path.exists():
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.parent / f"{config_path.name}.backup.{timestamp}"
    shutil.copy2(config_path, backup_path)

    # Keep only the last 5 backups
    backups = sorted(config_path.parent.glob(f"{config_path.name}.backup.*"))
    if len(backups) > 5:
        for old_backup in backups[:-5]:
            old_backup.unlink()

    return str(backup_path)
