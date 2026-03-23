"""Shared utility/factory functions for TomoBait."""

import os

from .config import BaitConfig


def get_embeddings(config: BaitConfig):
    """Create an embedding model instance based on config."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings

    if config.embedding.provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=config.embedding.model)
    elif config.embedding.provider == "anl_argo":
        return OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_base=config.embedding.argo_base_url,
            openai_api_key=config.embedding.api_key,
            check_embedding_ctx_length=False,
        )
    raise ValueError(f"Unknown embedding provider: {config.embedding.provider}")


def build_llm_config(config: BaitConfig):
    """Build an AG2 LLMConfig dict list (without tools).

    Returns an autogen LLMConfig object. Caller can inject tools after.
    """
    from autogen import LLMConfig

    llm_config_dict: dict = {
        "api_type": config.llm.api_type,
        "model": config.llm.model,
    }

    if config.llm.provider == "anl_argo":
        llm_config_dict["api_key"] = config.llm.api_key
        llm_config_dict["base_url"] = config.llm.argo_base_url
    else:
        api_key = os.getenv(config.llm.provider)
        if not api_key:
            raise RuntimeError(f"{config.llm.provider} environment variable not set.")
        llm_config_dict["api_key"] = api_key

    return LLMConfig(config_list=[llm_config_dict])
