"""Tests for the agent orchestration layer.

These tests inject a fake ``llm_chat_fn`` into ``build_graph`` so nothing
hits the network. The retriever and clients are also stubbed.
"""

from __future__ import annotations

from typing import List

import pytest

# --- Fakes ---


class FakeRetriever:
    """Minimal retriever: returns one fake chunk per query."""

    def __init__(self, results: List[str] | None = None):
        from langchain_core.documents import Document

        self.results = [
            Document(page_content=r, metadata={"source": "fake"})
            for r in (results or ["fake chunk content"])
        ]
        self.calls: list[str] = []

    def invoke(self, query: str):
        self.calls.append(query)
        return self.results


def _scripted_llm_chat(scripts: dict[str, list[dict]]):
    """Return a llm_chat_fn that pops scripted responses keyed by prompt key."""
    counters: dict[str, int] = {k: 0 for k in scripts}

    def _fn(client, model, max_tokens, system_prompt, messages, tools=None):
        # Pick the first key that the system prompt starts with — lets tests
        # key by router/doc/bits without re-pasting the whole prompt.
        for key, script in scripts.items():
            if key in system_prompt:
                idx = counters[key]
                counters[key] = min(idx + 1, len(script) - 1)
                return script[idx]
        raise AssertionError(f"No script matched system_prompt: {system_prompt[:60]}")

    return _fn


# --- Fixtures ---


@pytest.fixture
def fake_clients():
    """Return three sentinel objects standing in for the per-agent clients."""
    return object(), object(), object()


@pytest.fixture
def make_graph(base_config, fake_clients):
    """Factory: build a graph wired with fakes + a scripted llm_chat_fn."""
    from bait.agents import build_graph

    router_c, doc_c, bits_c = fake_clients

    def _build(scripts, retriever=None, device_skills="some device skills"):
        return build_graph(
            base_config,
            llm_chat_fn=_scripted_llm_chat(scripts),
            retriever=retriever or FakeRetriever(),
            router_client=router_c,
            doc_client=doc_c,
            bits_client=bits_c,
            device_skills=device_skills,
        )

    return _build


# --- Cached factory identity ---


def test_get_config_is_cached(base_config):
    """get_config() returns the same instance across calls."""
    from bait.config import get_config

    assert get_config() is get_config()


# --- build_graph wiring ---


def test_build_graph_returns_compiled(base_config, fake_clients):
    """build_graph yields a graph that can invoke without errors."""
    from bait.agents import build_graph

    router_c, doc_c, bits_c = fake_clients
    graph = build_graph(
        base_config,
        llm_chat_fn=_scripted_llm_chat(
            {
                "classifier": [
                    {"text": "documentation", "tool_calls": None, "stop": True}
                ],
                "expert on this project": [
                    {"text": "answer", "tool_calls": None, "stop": True}
                ],
                "expert on BITS": [
                    {"text": "device answer", "tool_calls": None, "stop": True}
                ],
            }
        ),
        retriever=FakeRetriever(),
        router_client=router_c,
        doc_client=doc_c,
        bits_client=bits_c,
        device_skills="",
    )
    assert graph is not None
    assert hasattr(graph, "invoke")


# --- R1: doc routing ---


def test_route_question_doc_path(make_graph):
    """R1: a doc question routes to doc_agent and uses query_documentation."""
    from bait.agents import route_question

    retriever = FakeRetriever(["how to configure the sample stage"])
    graph = make_graph(
        scripts={
            "classifier": [{"text": "documentation", "tool_calls": None, "stop": True}],
            "expert on this project": [
                # First call: tool_call.
                {
                    "text": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "query_documentation",
                            "arguments": {"query": "sample stage"},
                        }
                    ],
                    "stop": False,
                    "_assistant_msg": {"role": "assistant", "content": ""},
                },
                # Second call: final text.
                {
                    "text": "The sample stage is configured via the X motor.",
                    "tool_calls": None,
                    "stop": True,
                },
            ],
            "expert on BITS": [
                {"text": "should not run", "tool_calls": None, "stop": True}
            ],
        },
        retriever=retriever,
    )

    answer = route_question("how do I configure the sample stage?", graph=graph)
    assert "sample stage" in answer.lower()
    assert retriever.calls == ["sample stage"]


# --- R2: bits routing ---


def test_route_question_bits_path(make_graph):
    """R2: a device question routes to bits_agent and answers from skills."""
    from bait.agents import route_question

    graph = make_graph(
        scripts={
            "classifier": [{"text": "device", "tool_calls": None, "stop": True}],
            "expert on this project": [
                {"text": "should not run", "tool_calls": None, "stop": True}
            ],
            "expert on BITS": [
                {
                    "text": "Available motors: m1, m2, m3.",
                    "tool_calls": None,
                    "stop": True,
                }
            ],
        },
        device_skills="m1, m2, m3 are the motors.",
    )

    answer = route_question("what motors are available?", graph=graph)
    assert "motor" in answer.lower()
    assert "m1" in answer


# --- Doc tool loop cap ---


def test_doc_tool_loop_capped_at_max_iterations(make_graph):
    """A model that keeps calling the tool returns the fallback message."""
    from bait.agents import DOC_LOOP_FALLBACK, MAX_DOC_TOOL_ITERATIONS, route_question

    # Always return a tool call — the loop should bail after MAX iterations.
    looping_call = {
        "text": None,
        "tool_calls": [
            {
                "id": "loop",
                "name": "query_documentation",
                "arguments": {"query": "again"},
            }
        ],
        "stop": False,
        "_assistant_msg": {"role": "assistant", "content": ""},
    }
    retriever = FakeRetriever()
    graph = make_graph(
        scripts={
            "classifier": [{"text": "documentation", "tool_calls": None, "stop": True}],
            "expert on this project": [looping_call] * (MAX_DOC_TOOL_ITERATIONS + 5),
            "expert on BITS": [{"text": "n/a", "tool_calls": None, "stop": True}],
        },
        retriever=retriever,
    )

    answer = route_question("infinite tool loop", graph=graph)
    assert answer == DOC_LOOP_FALLBACK
    # Sanity: we capped, so the retriever was hit exactly MAX times.
    assert len(retriever.calls) == MAX_DOC_TOOL_ITERATIONS


# --- Router fallback on garbage classification ---


def test_router_fallback_on_unknown_class(make_graph):
    """Router returning gibberish defaults to the doc agent."""
    from bait.agents import route_question

    graph = make_graph(
        scripts={
            "classifier": [{"text": "marshmallows", "tool_calls": None, "stop": True}],
            "expert on this project": [
                {"text": "doc answer", "tool_calls": None, "stop": True}
            ],
            "expert on BITS": [
                {"text": "should not run", "tool_calls": None, "stop": True}
            ],
        }
    )
    answer = route_question("???", graph=graph)
    assert answer == "doc answer"
