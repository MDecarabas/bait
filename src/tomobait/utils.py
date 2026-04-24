"""Shared utility/factory functions for TomoBait."""

import json

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
    """Build an LLM client from per-agent settings.

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

        # The Anthropic SDK adds /v1/messages internally, so strip /v1
        # from the base URL to avoid /v1/v1/messages double-path.
        base = llm_settings["argo_base_url"].rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        client = Anthropic(
            api_key=llm_settings["api_key"],
            base_url=base,
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


def _is_anthropic(client) -> bool:
    """Check if client is an Anthropic client."""
    return type(client).__name__ == "Anthropic"


def llm_chat(
    client,
    model: str,
    max_tokens: int,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """SDK-agnostic LLM chat call.

    Returns a normalized dict:
        {"text": str|None, "tool_calls": list|None, "stop": bool}

    tool_calls (when present):
        [{"id": str, "name": str, "arguments": dict}, ...]
    """
    if _is_anthropic(client):
        return _chat_anthropic(
            client, model, max_tokens, system_prompt, messages, tools
        )
    return _chat_openai(
        client, model, max_tokens, system_prompt, messages, tools
    )


def _chat_anthropic(client, model, max_tokens, system_prompt, messages, tools):
    """Call Anthropic Messages API and normalize the response."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

    response = client.messages.create(**kwargs)

    text = None
    tool_calls = None
    for block in response.content:
        if block.type == "text":
            text = block.text
        elif block.type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                }
            )

    # Build raw assistant message for conversation history
    raw_content = [
        {"type": b.type, **({"text": b.text} if b.type == "text" else
         {"id": b.id, "name": b.name, "input": b.input})}
        for b in response.content
    ]

    return {
        "text": text,
        "tool_calls": tool_calls,
        "stop": response.stop_reason == "end_turn",
        "_assistant_msg": {"role": "assistant", "content": raw_content},
    }


def _chat_openai(client, model, max_tokens, system_prompt, messages, tools):
    """Call OpenAI Chat Completions API and normalize the response."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": full_messages,
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    text = choice.message.content
    tool_calls = None
    if choice.message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in choice.message.tool_calls
        ]

    return {
        "text": text,
        "tool_calls": tool_calls,
        "stop": choice.finish_reason == "stop",
        "_assistant_msg": choice.message,
    }


def build_tool_result_messages(
    client,
    tool_calls: list[dict],
    results: list[str],
    assistant_message=None,
) -> list[dict]:
    """Build tool result messages for the appropriate SDK.

    For Anthropic: returns messages with tool_result content blocks.
    For OpenAI: returns the assistant message + tool role messages.
    """
    if _is_anthropic(client):
        msgs = []
        if assistant_message is not None:
            msgs.append(assistant_message)
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    }
                    for tc, result in zip(tool_calls, results)
                ],
            }
        )
        return msgs
    else:
        msgs = []
        if assistant_message is not None:
            msgs.append(assistant_message)
        for tc, result in zip(tool_calls, results):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )
        return msgs
