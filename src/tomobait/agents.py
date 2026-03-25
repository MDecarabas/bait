import re
from typing import Annotated

import autogen
from dotenv import load_dotenv

from .config import BaitConfig
from .retriever import get_documentation_retriever
from .utils import build_llm_config

load_dotenv()

# Load configuration
config = BaitConfig()

# --- 1. Shared LLM Config (no tools) ---
base_llm_config = build_llm_config(config)

# --- 2. Doc Expert LLM Config (with query_documentation tool) ---
query_documentation_tool_dict = {
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

# Clone the base config and inject the tool
doc_llm_config_list = []
for entry in base_llm_config.config_list:
    d = dict(entry)
    d["tools"] = [query_documentation_tool_dict]
    doc_llm_config_list.append(d)

doc_llm_config = autogen.LLMConfig(config_list=doc_llm_config_list)

# --- 3. Load Retriever ---
retriever = get_documentation_retriever()


# --- 4. Doc Expert Agents ---
technician_agent = autogen.AssistantAgent(
    "doc_expert", llm_config=doc_llm_config, system_message=config.llm.system_message
)

worker_agent = autogen.UserProxyAgent(
    "tool_worker",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: not msg.get("tool_calls"),
    code_execution_config=False,
)

# captain_agent = autogen.CaptainAgent(
#     "captain",


@worker_agent.register_for_execution(name="query_documentation")
def query_documentation(
    query: Annotated[str, "The search query for the documentation"],
) -> str:
    """
    A tool that takes a user's query, retrieves relevant
    document chunks, and returns them as a single string with source links.
    """
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
            chunk_with_source = f"{chunk}\n\n[Sources: {' | '.join(sources)}]"
        else:
            chunk_with_source = chunk

        formatted_chunks.append(chunk_with_source)

    context_str = "\n\n---\n\n".join(formatted_chunks)

    print(f"--- TOOL: Found {len(results)} chunks. ---")
    return context_str


def run_agent_chat(user_question: str) -> str:
    """
    Runs a chat between doc agents to answer a documentation question.
    """
    print("Starting doc agent chat...")
    chat_result = worker_agent.initiate_chat(
        recipient=technician_agent,
        message=(
            f"Please answer this question: '{user_question}'. "
            "You *must* use the 'query_documentation' tool to find the "
            "relevant context first. "
            "Provide a concise but complete answer (2-3 paragraphs). "
            "If the question asks 'how to' do something, provide step-by-step "
            "instructions as a numbered list. "
            "Include relevant source links from the context at the end of "
            "your response."
        ),
    )

    final_answer = chat_result.summary
    if final_answer:
        print("\n--- FINAL ANSWER ---")
        print(final_answer)
        return final_answer
    return "Sorry, I couldn't find an answer."


# --- 5. BITS Device Expert Agent ---

# Load device_skills.md if it exists
_device_skills_content = ""
_skills_path = config.bits_skills_dir / "device_skills.md"
if _skills_path.is_file():
    _device_skills_content = _skills_path.read_text()
    print(f"Loaded device skills from: {_skills_path}")
else:
    print(f"Warning: device_skills.md not found at {_skills_path}")

BITS_SYSTEM_PROMPT = """\
You are an expert on BITS (Beamline Instrument and Tool Suite) devices, \
ophyd signals, EPICS PVs, scan parameters, and Bluesky plans for this beamline.

Answer questions about devices, their signals, PV names, scan configuration, \
and how to interact with hardware using ophyd and Bluesky.

Base your answers on the device reference below. If the reference does not \
contain enough information, say so. Do not invent PV names or device \
attributes.

--- DEVICE REFERENCE ---
{skills}
--- END DEVICE REFERENCE ---
""".format(skills=(_device_skills_content or "(no device skills loaded)"))

bits_device_expert = autogen.AssistantAgent(
    "bits_device_expert",
    llm_config=base_llm_config,
    system_message=BITS_SYSTEM_PROMPT,
)

bits_worker = autogen.UserProxyAgent(
    "bits_worker",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: True,
    code_execution_config=False,
)


def run_bits_chat(user_question: str) -> str:
    """
    Runs a chat with the BITS device expert to answer a device question.
    """
    print("Starting BITS device agent chat...")
    chat_result = bits_worker.initiate_chat(
        recipient=bits_device_expert,
        message=user_question,
    )

    final_answer = chat_result.summary
    if final_answer:
        print("\n--- BITS ANSWER ---")
        print(final_answer)
        return final_answer
    return "Sorry, I couldn't find an answer about that device."


# --- 6. Question Router ---

# Keywords that indicate a device-related question
_DEVICE_KEYWORDS = re.compile(
    r"\b("
    r"device|signal|pv\b|epics|ophyd|bluesky|motor|detector|shutter|"
    r"\.get\(\)|\.put\(|prefix|suffix|component|"
    r"scan[_ ]?parameter|rotation[_ ]?start|exposure[_ ]?time|"
    r"tomoscan|tomooptics|mctoptics"
    r")\b",
    re.IGNORECASE,
)


def route_question(user_question: str) -> str:
    """Route user question to the appropriate agent.

    Uses keyword matching to decide between the device expert
    and the documentation expert.
    """
    if _device_skills_content and _DEVICE_KEYWORDS.search(user_question):
        print("Router: dispatching to BITS device expert")
        return run_bits_chat(user_question)
    else:
        print("Router: dispatching to documentation expert")
        return run_agent_chat(user_question)
