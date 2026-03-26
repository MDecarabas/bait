"""Shared utility/factory functions for TomoBait."""

from .config import BaitConfig

_client_cache: dict[tuple, object] = {}


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


def build_llm_client(llm_settings: dict):
    """Build an LLM client from resolved per-agent settings.

    ``llm_settings`` is a dict with keys: api_type, api_key, argo_base_url.
    Clients are cached so agents sharing the same provider reuse one instance.
    """
    cache_key = (
        llm_settings["api_type"],
        llm_settings["api_key"],
        llm_settings["argo_base_url"],
    )
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    if llm_settings["api_type"] == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(
            api_key=llm_settings["api_key"],
            base_url=llm_settings["argo_base_url"],
        )
    elif llm_settings["api_type"] == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=llm_settings["api_key"],
            base_url=llm_settings["argo_base_url"],
        )
    else:
        raise ValueError(f"Unknown api_type: {llm_settings['api_type']}")

    _client_cache[cache_key] = client
    return client
