"""Agent orchestration using LangGraph.

The compiled graph is built lazily by ``get_graph()`` so importing this
module never touches the network or filesystem. Tests can call
``build_graph(config, llm_chat_fn=fake, retriever=...)`` directly to inject
fakes; ``build_graph`` constructs every component itself when not given one.
"""

import json
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from .config import BaitConfig, get_config
from .devices import OASClient
from .retriever import get_documentation_retriever
from .utils import build_llm_client, build_tool_result_messages, llm_chat

# Hard cap on each agent's tool-call loop. Stops runaway searches/queries when
# the model keeps re-calling tools instead of answering.
MAX_DOC_TOOL_ITERATIONS = 5
MAX_BITS_TOOL_ITERATIONS = 5

DOC_LOOP_FALLBACK = (
    "I couldn't find a confident answer after several searches. "
    "Try rephrasing your question or consulting the documentation directly."
)

BITS_LOOP_FALLBACK = (
    "I couldn't form a confident answer about the device after several attempts. "
    "Try rephrasing your question."
)

ARGO_UNAVAILABLE_MESSAGE = (
    "The AI service is temporarily unavailable. Please try again in a moment."
)


def _load_device_skills(skills_dir: Path) -> str:
    """Concatenate every .md file under skills_dir (recursive) into one string."""
    if not skills_dir.is_dir():
        return ""
    parts = []
    for md_path in sorted(skills_dir.rglob("*.md")):
        try:
            parts.append(md_path.read_text())
        except OSError:
            continue
    return "\n\n---\n\n".join(parts)


def _build_bits_system_prompt(config: BaitConfig, device_skills: str) -> str:
    prompt = config.agents.bits_agent.system_prompt
    if device_skills:
        prompt += (
            "\n\n--- DEVICE REFERENCE ---\n"
            + device_skills
            + "\n--- END DEVICE REFERENCE ---"
        )
    return prompt


# --- LangGraph State ---


class AgentState(TypedDict):
    question: str
    answer: str
    route: str
    pending_writes: list[dict]


# --- Doc agent tool definition ---

DOC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_documentation",
            "description": "Search the project documentation for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the documentation",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# --- BITS agent tool definitions ---

BITS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_device",
            "description": (
                "Read the current value of a registered ophyd device. "
                "Optionally narrow to one component (e.g. name='tomoscan', "
                "component='rotation_start'). Returns JSON with the value, "
                "timestamp, and connection status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Device name as registered in the OAS server.",
                    },
                    "component": {
                        "type": "string",
                        "description": "Optional component attribute name.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_device",
            "description": (
                "Propose a write to a registered ophyd device. The write is "
                "NOT executed immediately — it is staged and the user must "
                "confirm via the UI. After calling, briefly summarise the "
                "proposed change in plain language and STOP."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Device name as registered in the OAS server.",
                    },
                    "value": {
                        "type": ["number", "string"],
                        "description": "New value (number or string) to assign.",
                    },
                    "component": {
                        "type": "string",
                        "description": "Optional component attribute name.",
                    },
                },
                "required": ["name", "value"],
            },
        },
    },
]


def _format_retrieved_chunks(results) -> str:
    formatted_chunks = []
    for doc in results:
        chunk = doc.page_content
        sources = []
        if "source" in doc.metadata:
            source_path = doc.metadata["source"]
            if "config_resources" not in str(source_path):
                sources.append(f"Source: {source_path}")
        url_fields = [
            "documentation",
            "docs",
            "official_page",
            "website",
            "github",
            "pypi",
            "url",
        ]
        for field in url_fields:
            if field in doc.metadata and doc.metadata[field]:
                url = doc.metadata[field]
                field_name = field.replace("_", " ").title()
                sources.append(f"{field_name}: {url}")
        if sources:
            formatted_chunks.append(f"{chunk}\n\n[Sources: {' | '.join(sources)}]")
        else:
            formatted_chunks.append(chunk)
    return "\n\n---\n\n".join(formatted_chunks)


# --- Graph builder (config + llm_chat injectable for tests) ---


def build_graph(
    config: BaitConfig,
    llm_chat_fn: Callable = llm_chat,
    retriever=None,
    router_client=None,
    doc_client=None,
    bits_client=None,
    device_skills: str | None = None,
    oas_client: OASClient | None = None,
):
    """Build a compiled LangGraph for the configured agents.

    Every dependency is injectable so tests can pass fakes instead of hitting
    the network. Production callers should use ``get_graph()`` which wires
    in the cached lazy factories.
    """
    if retriever is None:
        retriever = get_documentation_retriever(config)
    if router_client is None:
        router_client = build_llm_client(config.get_agent_llm_settings("router"))
    if doc_client is None:
        doc_client = build_llm_client(config.get_agent_llm_settings("doc_agent"))
    if bits_client is None:
        bits_client = build_llm_client(config.get_agent_llm_settings("bits_agent"))
    if device_skills is None:
        device_skills = _load_device_skills(config.bits_skills_dir)
    if oas_client is None:
        oas_client = OASClient(
            f"http://{config.ophyd_websocket.host}:{config.ophyd_websocket.port}"
        )

    router_settings = config.get_agent_llm_settings("router")
    doc_settings = config.get_agent_llm_settings("doc_agent")
    bits_settings = config.get_agent_llm_settings("bits_agent")
    bits_system_prompt = _build_bits_system_prompt(config, device_skills)

    def query_documentation(query: str) -> str:
        print(f"\n--- TOOL: Querying for '{query}' ---")
        results = retriever.invoke(query)
        print(f"--- TOOL: Found {len(results)} chunks. ---")
        return _format_retrieved_chunks(results)

    def router_node(state: AgentState) -> dict:
        response = llm_chat_fn(
            client=router_client,
            model=router_settings["model"],
            max_tokens=config.agents.router.max_tokens,
            system_prompt=config.agents.router.system_prompt,
            messages=[{"role": "user", "content": state["question"]}],
        )
        route = (response["text"] or "").strip().lower()
        if route not in ("documentation", "device"):
            route = "documentation"
        print(f"Router: classified as '{route}'")
        return {"route": route}

    def route_decision(state: AgentState) -> str:
        return "bits_agent" if state["route"] == "device" else "doc_agent"

    def doc_agent_node(state: AgentState) -> dict:
        print("Starting doc agent chat...")
        prompt = (
            f"Please answer this question: '{state['question']}'. "
            "You *must* use the 'query_documentation' tool to find the "
            "relevant context first. "
            "Provide a concise but complete answer (2-3 paragraphs). "
            "If the question asks 'how to' do something, provide step-by-step "
            "instructions as a numbered list. "
            "Include relevant source links from the context at the end of "
            "your response."
        )
        messages = [{"role": "user", "content": prompt}]

        for _ in range(MAX_DOC_TOOL_ITERATIONS):
            response = llm_chat_fn(
                client=doc_client,
                model=doc_settings["model"],
                max_tokens=config.agents.doc_agent.max_tokens,
                system_prompt=config.agents.doc_agent.system_prompt,
                messages=messages,
                tools=DOC_TOOLS,
            )

            if response["tool_calls"]:
                tool_results = []
                for tc in response["tool_calls"]:
                    result = query_documentation(tc["arguments"]["query"])
                    tool_results.append(result)
                tool_msgs = build_tool_result_messages(
                    client=doc_client,
                    tool_calls=response["tool_calls"],
                    results=tool_results,
                    assistant_message=response["_assistant_msg"],
                )
                messages.extend(tool_msgs)
                continue

            final_text = response["text"] or ""
            print("\n--- FINAL ANSWER ---")
            print(final_text)
            return {"answer": final_text or "Sorry, I couldn't find an answer."}

        # Loop exhausted — return graceful fallback rather than spinning forever.
        print(
            f"\n--- DOC AGENT: hit MAX_DOC_TOOL_ITERATIONS "
            f"({MAX_DOC_TOOL_ITERATIONS}); returning fallback. ---"
        )
        return {"answer": DOC_LOOP_FALLBACK}

    def bits_agent_node(state: AgentState) -> dict:
        print("Starting BITS device agent chat...")
        messages = [{"role": "user", "content": state["question"]}]
        proposed_writes: list[dict] = []

        for _ in range(MAX_BITS_TOOL_ITERATIONS):
            response = llm_chat_fn(
                client=bits_client,
                model=bits_settings["model"],
                max_tokens=config.agents.bits_agent.max_tokens,
                system_prompt=bits_system_prompt,
                messages=messages,
                tools=BITS_TOOLS,
            )

            if response["tool_calls"]:
                tool_results = []
                for tc in response["tool_calls"]:
                    args = tc["arguments"]
                    name = tc["name"]
                    if name == "read_device":
                        result = oas_client.read_device(
                            args["name"], args.get("component")
                        )
                        tool_results.append(json.dumps(result))
                    elif name == "set_device":
                        write = {
                            "name": args["name"],
                            "value": args["value"],
                            "component": args.get("component"),
                            "tool_call_id": tc["id"],
                        }
                        proposed_writes.append(write)
                        placeholder = (
                            f"PROPOSED_WRITE: device={write['name']!r} "
                            f"component={write['component']!r} "
                            f"value={write['value']!r}. NOT EXECUTED. "
                            "Tell the user what this will do and stop. "
                            "The user will confirm via the UI."
                        )
                        tool_results.append(placeholder)
                    else:
                        tool_results.append(f"unknown tool: {name}")
                tool_msgs = build_tool_result_messages(
                    client=bits_client,
                    tool_calls=response["tool_calls"],
                    results=tool_results,
                    assistant_message=response["_assistant_msg"],
                )
                messages.extend(tool_msgs)
                continue

            final_text = response["text"] or ""
            if not final_text and proposed_writes:
                final_text = (
                    f"I'd like to make {len(proposed_writes)} change(s); "
                    "please confirm below."
                )
            if final_text:
                print("\n--- BITS ANSWER ---")
                print(final_text)
            fallback = "Sorry, I couldn't find an answer about that device."
            return {
                "answer": final_text or fallback,
                "pending_writes": proposed_writes,
            }

        # Loop exhausted — return graceful fallback rather than spinning forever.
        print(
            f"\n--- BITS AGENT: hit MAX_BITS_TOOL_ITERATIONS "
            f"({MAX_BITS_TOOL_ITERATIONS}); returning fallback. ---"
        )
        return {"answer": BITS_LOOP_FALLBACK, "pending_writes": proposed_writes}

    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("doc_agent", doc_agent_node)
    builder.add_node("bits_agent", bits_agent_node)
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_decision)
    builder.add_edge("doc_agent", END)
    builder.add_edge("bits_agent", END)
    return builder.compile()


# --- Process-wide compiled graph singleton ---

_graph = None


def get_graph():
    """Return the cached compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph(get_config())
    return _graph


# --- Public API ---


def route_question(user_question: str, graph=None) -> tuple[str, list[dict]]:
    """Route a user question to the appropriate agent via LangGraph.

    Returns ``(answer, pending_writes)``. ``pending_writes`` is non-empty
    only when the bits_agent has staged one or more writes for HITL confirmation.
    Pass ``graph`` for tests; otherwise the cached production graph is used.
    """
    if graph is None:
        graph = get_graph()
    result = graph.invoke(
        {
            "question": user_question,
            "answer": "",
            "route": "",
            "pending_writes": [],
        }
    )
    return result["answer"], result.get("pending_writes", [])
