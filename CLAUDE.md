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
# Start the FastAPI backend (port 8001)
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
