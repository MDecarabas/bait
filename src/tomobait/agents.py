"""Agent orchestration using LangGraph."""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import BaitConfig
from .retriever import get_documentation_retriever
from .utils import build_llm_client, build_tool_result_messages, llm_chat

# --- Configuration & Per-Agent Clients ---
config = BaitConfig()
retriever = get_documentation_retriever()

_router_settings = config.get_agent_llm_settings("router")
_doc_settings = config.get_agent_llm_settings("doc_agent")
_bits_settings = config.get_agent_llm_settings("bits_agent")

router_client = build_llm_client(_router_settings)
doc_client = build_llm_client(_doc_settings)
bits_client = build_llm_client(_bits_settings)


# --- LangGraph State ---
class AgentState(TypedDict):
    question: str
    answer: str
    route: str


# --- Router ---


def router_node(state: AgentState) -> dict:
    """Use the LLM to classify the question and decide which agent to use."""
    response = llm_chat(
        client=router_client,
        model=_router_settings["model"],
        max_tokens=50,
        system_prompt=config.agents.router.system_prompt,
        messages=[{"role": "user", "content": state["question"]}],
    )
    route = (response["text"] or "").strip().lower()
    if route not in ("documentation", "device"):
        route = "documentation"
    print(f"Router: classified as '{route}'")
    return {"route": route}


def route_decision(state: AgentState) -> str:
    """Conditional edge: dispatch to the correct agent node."""
    if state["route"] == "device":
        return "bits_agent"
    return "doc_agent"


# --- Documentation Agent ---

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


def query_documentation(query: str) -> str:
    """Execute the documentation retrieval tool."""
    print(f"\n--- TOOL: Querying for '{query}' ---")

    results = retriever.invoke(query)

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
            chunk_with_source = (
                f"{chunk}\n\n[Sources: {' | '.join(sources)}]"
            )
        else:
            chunk_with_source = chunk

        formatted_chunks.append(chunk_with_source)

    context_str = "\n\n---\n\n".join(formatted_chunks)

    print(f"--- TOOL: Found {len(results)} chunks. ---")
    return context_str


def doc_agent_node(state: AgentState) -> dict:
    """Documentation expert agent with tool use loop."""
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

    while True:
        response = llm_chat(
            client=doc_client,
            model=_doc_settings["model"],
            max_tokens=4096,
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
        else:
            final_text = response["text"] or ""
            print("\n--- FINAL ANSWER ---")
            print(final_text)
            return {
                "answer": final_text or "Sorry, I couldn't find an answer."
            }


# --- BITS Device Expert Agent ---

_device_skills_content = ""
_skills_path = config.bits_skills_dir / "device_skills.md"
if _skills_path.is_file():
    _device_skills_content = _skills_path.read_text()
    print(f"Loaded device skills from: {_skills_path}")
else:
    print(f"Warning: device_skills.md not found at {_skills_path}")

_bits_system_prompt = config.agents.bits_agent.system_prompt
if _device_skills_content:
    _bits_system_prompt += (
        "\n\n--- DEVICE REFERENCE ---\n"
        + _device_skills_content
        + "\n--- END DEVICE REFERENCE ---"
    )


def bits_agent_node(state: AgentState) -> dict:
    """BITS device expert agent (knowledge-only, no tools yet)."""
    print("Starting BITS device agent chat...")

    response = llm_chat(
        client=bits_client,
        model=_bits_settings["model"],
        max_tokens=4096,
        system_prompt=_bits_system_prompt,
        messages=[{"role": "user", "content": state["question"]}],
    )

    final_text = response["text"] or ""

    if final_text:
        print("\n--- BITS ANSWER ---")
        print(final_text)

    fallback = "Sorry, I couldn't find an answer about that device."
    return {"answer": final_text or fallback}


# --- Build LangGraph ---

graph_builder = StateGraph(AgentState)

graph_builder.add_node("router", router_node)
graph_builder.add_node("doc_agent", doc_agent_node)
graph_builder.add_node("bits_agent", bits_agent_node)

graph_builder.set_entry_point("router")
graph_builder.add_conditional_edges("router", route_decision)
graph_builder.add_edge("doc_agent", END)
graph_builder.add_edge("bits_agent", END)

graph = graph_builder.compile()


# --- Public API ---


def route_question(user_question: str) -> str:
    """Route user question to the appropriate agent via LangGraph.

    This is the main entry point called by the FastAPI backend.
    """
    result = graph.invoke(
        {"question": user_question, "answer": "", "route": ""}
    )
    return result["answer"]
