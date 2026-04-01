"""Tests for utility functions."""

import pytest


def test_build_llm_client_openai():
    """build_llm_client with api_type='openai' should return an OpenAI client."""
    from tomobait.utils import build_llm_client, _client_cache

    _client_cache.clear()
    settings = {
        "api_type": "openai",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client = build_llm_client(settings)
    from openai import OpenAI

    assert isinstance(client, OpenAI)


def test_build_llm_client_anthropic():
    """build_llm_client with api_type='anthropic' should return an Anthropic client."""
    from tomobait.utils import build_llm_client, _client_cache

    _client_cache.clear()
    settings = {
        "api_type": "anthropic",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client = build_llm_client(settings)
    from anthropic import Anthropic

    assert isinstance(client, Anthropic)


def test_build_llm_client_unknown_raises():
    """build_llm_client with unknown api_type should raise ValueError."""
    from tomobait.utils import build_llm_client, _client_cache

    _client_cache.clear()
    settings = {
        "api_type": "unknown",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    with pytest.raises(ValueError, match="Unknown api_type"):
        build_llm_client(settings)


def test_build_llm_client_caching():
    """Same settings should return the same client instance (cached)."""
    from tomobait.utils import build_llm_client, _client_cache

    _client_cache.clear()
    settings = {
        "api_type": "openai",
        "api_key": "test-key",
        "argo_base_url": "http://test-url",
    }
    client1 = build_llm_client(settings)
    client2 = build_llm_client(settings)
    assert client1 is client2
