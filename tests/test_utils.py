"""Tests for utility functions."""

import pytest


def test_build_llm_client_openai():
    """build_llm_client with api_type='openai' returns an OpenAI client."""
    from bait.utils import _client_cache, build_llm_client

    _client_cache.clear()
    settings = {
        "api_type": "openai",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client = build_llm_client(settings)
    from openai import OpenAI

    assert isinstance(client, OpenAI)


def test_build_llm_client_openai_has_timeout():
    """OpenAI client built by Bait carries the 30s timeout."""
    from bait.utils import LLM_TIMEOUT_SECONDS, _client_cache, build_llm_client

    _client_cache.clear()
    client = build_llm_client(
        {"api_type": "openai", "api_key": "k", "argo_base_url": "http://u"}
    )
    assert client.timeout == LLM_TIMEOUT_SECONDS


def test_build_llm_client_anthropic():
    """build_llm_client with api_type='anthropic' returns an Anthropic client."""
    from bait.utils import _client_cache, build_llm_client

    _client_cache.clear()
    settings = {
        "api_type": "anthropic",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client = build_llm_client(settings)
    from anthropic import Anthropic

    assert isinstance(client, Anthropic)


def test_build_llm_client_anthropic_has_timeout():
    """Anthropic client built by Bait carries the 30s timeout."""
    from bait.utils import LLM_TIMEOUT_SECONDS, _client_cache, build_llm_client

    _client_cache.clear()
    client = build_llm_client(
        {"api_type": "anthropic", "api_key": "k", "argo_base_url": "http://u"}
    )
    assert client.timeout == LLM_TIMEOUT_SECONDS


def test_build_llm_client_unknown_raises():
    """build_llm_client with unknown api_type raises ValueError."""
    from bait.utils import _client_cache, build_llm_client

    _client_cache.clear()
    settings = {
        "api_type": "unknown",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    with pytest.raises(ValueError, match="Unknown api_type"):
        build_llm_client(settings)


def test_build_llm_client_caching():
    """Same settings return the same cached client instance."""
    from bait.utils import _client_cache, build_llm_client

    _client_cache.clear()
    settings = {
        "api_type": "openai",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client1 = build_llm_client(settings)
    client2 = build_llm_client(settings)
    assert client1 is client2
