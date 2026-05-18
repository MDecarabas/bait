# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomoBait is a RAG (Retrieval-Augmented Generation) system for tomography beamline documentation. It ingests Sphinx documentation from the 2-BM beamline, stores it in a vector database (ChromaDB), and provides a conversational interface for querying the documentation using LangGraph-orchestrated AI agents.

## Development Environment

This project uses **uv** for dependency management and task running.

### Initial Setup

```bash
uv venv
uv pip install -e .
```

## Common Commands

### Running the Application

```bash
# Start the FastAPI backend (port 8001) — also spawns the OAS subprocess on
# port 8002 (configurable via ophyd_websocket.port). Set
# ophyd_websocket.auto_spawn: false in config.yaml to run OAS yourself.
uv run start-backend

# Start the Gradio frontend (port 8000)
uv run start-frontend
```

### Code Quality

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .
```

### Data Ingestion

```bash
# Ingest documentation (clones repo, builds Sphinx docs, creates vector DB)
uv run python -m tomobait.data_ingestion
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tomobait
```

## Architecture

### Process topology

Four processes cooperate when bait is fully running. Two of them (bait
backend + OAS) are spawned by `uv run start-backend`; the other two you
start yourself.

```
┌─────────────────────────┐   EPICS CA   ┌─────────────┐
│ queueserver (60610)     │◄────────────►│             │
│  + start-re-manager     │              │  EPICS IOCs │
│  loads startup.py       │              │             │
└─────────────────────────┘              │             │
                                         │             │
┌─────────────────────────┐   EPICS CA   │             │
│ OAS server (8002)       │◄────────────►│             │
│  bait subprocess        │              └─────────────┘
│  loads startup.py       │
└────────────▲────────────┘
             │  ws://localhost:8002/api/v1/device-socket
             │  {action:'subscribe'|'set'|'unsubscribe'}
┌────────────┴────────────┐
│ bait backend (8001)     │  HTTP GET queueserver/api/status (safety gate)
│  spawns + supervises OAS in lifespan
└────────────▲────────────┘
             │  HTTP /chat, /chat/confirm
┌────────────┴────────────┐
│ frontend (8000)         │
└─────────────────────────┘
```

The queueserver and OAS each load `startup.py` and hold their own ophyd
sessions — two independent sets of Device instances connected to the same
EPICS PVs. EPICS is pub/sub and tolerates this. The bait-side QS safety
check (`bait.device_io.check_queueserver`) is what prevents bait from
pushing writes through OAS while the RunEngine is mid-plan.

### Multi-Agent System (LangGraph)

TomoBait uses a LangGraph StateGraph with three nodes:

1. **Router** — classifies incoming questions as "documentation" or "device"
2. **Doc Agent** — retrieves documentation chunks from ChromaDB via tool calling, then synthesizes an answer
3. **BITS Agent** — answers device-related questions using a preloaded skills reference file

The flow: `router → (conditional edge) → doc_agent | bits_agent → END`

### Project-Based Data Isolation

All data lives under `.bait-{project.name}/` (e.g., `.bait-tomo/`):

```
.bait-tomo/
├── chroma_db/          # Vector database
├── documentation/      # Cloned repos and built docs
└── chat_history/       # Saved conversation JSON files
```

### Modules

1. **Configuration** (`config.py`)
   - Centralized configuration via pydantic-settings, loaded from `config.yaml`
   - `BaitConfig` is the main settings class with computed path properties (`data_dir`, `docs_output_dir`, `db_path`, `chat_history_dir`, `bits_skills_dir`)
   - Per-agent LLM settings with global defaults and per-agent overrides via `get_agent_llm_settings(agent_name)`

2. **Utilities** (`utils.py`)
   - `get_embeddings(config)` — shared factory for embedding models (HuggingFace or ANL Argo)
   - `build_llm_client(llm_settings)` — cached factory for OpenAI/Anthropic clients
   - `llm_chat()` — SDK-agnostic wrapper that normalizes OpenAI and Anthropic call/response formats
   - `build_tool_result_messages()` — formats tool results for either SDK

3. **Data Ingestion** (`data_ingestion.py`)
   - Clones/updates documentation repositories from GitHub
   - Builds Sphinx documentation to HTML
   - Uses `ReadTheDocsLoader` to load HTML documentation
   - Chunks documents using `RecursiveCharacterTextSplitter` (configurable size/overlap)
   - Stores in ChromaDB at `.bait-{project.name}/chroma_db`

4. **Retriever** (`retriever.py`)
   - Shared utility for accessing ChromaDB
   - Returns top-k most relevant document chunks (configurable, default k=3)
   - Can be tested standalone: `python -m tomobait.retriever "test query"`

5. **Agents** (`agents.py`)
   - LangGraph StateGraph with three nodes: `router`, `doc_agent`, `bits_agent`
   - Router classifies questions; conditional edge dispatches to the right agent
   - Doc agent uses OpenAI-compatible tool calling to invoke `query_documentation`
   - BITS agent uses preloaded device skills reference for device questions
   - `route_question(user_question)` — main entry point called by the backend

6. **Backend** (`app.py`)
   - FastAPI server with endpoints:
     - `POST /chat` — send a question, get an agent response
     - `GET /config` — read current configuration
     - `POST /config` — update configuration (placeholder)
     - `POST /chat/save` — save or update a conversation
     - `GET /chat/history` — list saved chats (metadata only)
     - `GET /chat/history/{chat_id}` — load a full conversation
     - `DELETE /chat/history/{chat_id}` — delete a conversation
   - Path traversal protection on chat history endpoints

7. **Frontend** (`frontend.py`)
   - Gradio chatbot interface with sidebar chat history management
   - Makes HTTP requests to the FastAPI backend
   - Save, load, and delete conversation sessions
   - Serves documentation images through Gradio's `allowed_paths` mechanism

8. **Ophyd WebSocket integration**
   - `ophyd_websocket/` — vendored OAS FastAPI server (REST + four WS routers).
     See `src/bait/ophyd_websocket/CLAUDE.md` for the module map, lifecycle,
     and WS protocol cheat sheet. **Read that file before touching anything
     under that folder.**
   - `ophyd_websocket_supervisor.py` — `OASSupervisor` class that the bait
     backend's lifespan uses to spawn the OAS subprocess (env-configured
     `OAS_STARTUP_DIR`/`HOST`/`PORT`), poll `/api/v1/devices` until devices
     are loaded, and terminate cleanly on shutdown.
   - `ophyd_ws_client.py` — sync WebSocket client (`read_device`, `set_device`)
     used by the bits_agent's tool handlers and the `/chat/confirm` endpoint.
     Speaks the `device-socket` protocol. No `component` support (WS-protocol
     limitation; use the OAS REST `PUT /devices` endpoint if you need it).
   - `device_io.py` — **just** the bait-side queueserver safety probe
     (`check_queueserver`). All device I/O now goes through OAS over WS.

## Key Configuration

All configuration is centralized in `config.yaml`:

- **Project Settings**: `project.name` and `project.data_dir`
- **Documentation**: Git repos, local folders, and reference resources
- **Retriever**: k, search_type, score_threshold
- **Embedding**: Provider (`huggingface` or `anl_argo`), model name, device
- **Agents**: Per-agent config (router, doc_agent, bits_agent) with system prompts and LLM settings (model, api_type, api_key, argo_base_url)
- **Text Processing**: chunk_size, chunk_overlap
- **Server**: Ports (backend 8001, frontend 8000)
- **BITS**: bits_skills_dir for device reference files

## API Keys

All API keys are configured per-agent in `config.yaml` under `agents.{agent_name}.api_key`. For ANL Argo, the API key is your ANL username. Each agent can use a different provider/model independently.

## Code Style

- Ruff for linting and formatting (config in `ruff.toml`)
- Line length: 88 characters
- Linting rules: Pyflakes (F), pycodestyle (E), isort (I)
- Double quotes, space indentation

## Important Implementation Details

### Agent Routing and Tool Calling

The LangGraph router classifies questions as "documentation" or "device". The doc agent uses tool calling (`query_documentation`) via the SDK-agnostic `llm_chat()` wrapper in `utils.py`, which handles both OpenAI and Anthropic protocols. The tool-call loop continues until the LLM returns a final text answer. The BITS agent answers from a preloaded device skills markdown file without tool calls.

### Chat History Persistence

Conversations are saved as JSON files in `.bait-{project.name}/chat_history/`. Each file stores messages, title, timestamps, and a unique ID. The backend guards against path traversal attacks on chat IDs.

### Documentation Sources

The system can ingest from multiple sources:
- **Git Repositories**: Cloned to `.bait-{name}/documentation/`, Sphinx docs built automatically
- **Local Folders**: Pre-built documentation can be loaded directly

### Image Handling in Frontend

The Gradio frontend resolves documentation image paths to absolute filesystem paths and serves them through Gradio's `allowed_paths` mechanism.

### Path Resolution

All paths are computed properties on `BaitConfig`:
- `config.data_dir` → `.bait-{project.name}/`
- `config.db_path` → `.bait-{project.name}/chroma_db`
- `config.docs_output_dir` → `.bait-{project.name}/documentation`
- `config.chat_history_dir` → `.bait-{project.name}/chat_history`
- `config.bits_skills_dir` → `.bait-{project.name}/bits_skills`

### Ophyd device flow (end to end)

1. User starts the bluesky queueserver themselves:
   `bash {bits.path}/scripts/{bits.instrument_name}_qs_host.sh start`
2. User starts bait: `uv run start-backend`. Lifespan spawns
   `python -m bait.ophyd_websocket.server` on `ophyd_websocket.port`
   (default 8002) with `OAS_STARTUP_DIR=config.oas_startup_file`. The OAS
   subprocess imports `startup.py` and populates its device registry from
   the same Guarneri YAMLs the queueserver uses.
3. `OASSupervisor.wait_ready` polls `/api/v1/devices` until count > 0
   (timeout from `ophyd_websocket.ready_timeout`).
4. The bits_agent's `read_device`/`set_device` tool calls open a short-lived
   WebSocket to `ws://localhost:8002/api/v1/device-socket` per call.
5. `set_device` calls are staged (HITL) and confirmed via `/chat/confirm`,
   which re-checks `device_io.check_queueserver` before sending the WS `set`
   so writes never collide with a running RE plan.

Run OAS manually instead (e.g. for debugging, or to share with another
client) by setting `ophyd_websocket.auto_spawn: false` and launching it
yourself:
`python -m bait.ophyd_websocket.server --startup-dir <startup.py>`.

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
