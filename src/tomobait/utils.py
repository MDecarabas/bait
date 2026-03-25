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


def build_llm_client(config: BaitConfig):
    """Build an Anthropic client based on config.

    For ANL Argo, uses the Argo base URL with the user's ANL username as api_key.
    For direct Anthropic, reads the API key from the environment variable
    specified in config.llm.provider.
    """

    if config.llm.api_type == "anthropic":
        from anthropic import Anthropic

        return Anthropic(
            api_key=config.llm.api_key,
            base_url=config.llm.argo_base_url,
        )

    elif config.llm.api_type == "openai":
        from openai import OpenAI
        return OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.argo_base_url,
        )

