# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomoBait is a RAG (Retrieval-Augmented Generation) system for tomography beamline documentation. It ingests Sphinx documentation from the 2-BM beamline, stores it in a vector database (ChromaDB), and provides a conversational interface for querying the documentation using AI agents.

## Development Environment

This project uses **uv** for dependency management and task running.

### Initial Setup

```bash
uv venv # Creates a virtual environment
uv pip install -e . # Installs dependencies
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
ruff check .

# Format code
ruff format .
```

### Data Ingestion

```bash
# Ingest documentation (clones repo, builds Sphinx docs, creates vector DB)
uv run python -m tomobait.data_ingestion
```

## Architecture

### Project-Based Data Isolation

TomoBait uses a project-based directory structure to isolate all data:
- Each project is defined in `config.yaml` with a `project.name` (e.g., "tomo")
- All data is stored in `.bait-{name}/` directory (e.g., `.bait-tomo/`)
- Directory structure:
  ```
  .bait-tomo/
  ├── chroma_db/          # Vector database
  └── documentation/      # Cloned repos and built docs
  ```

### Modules

1. **Configuration** (`config.py`)
   - Centralized configuration via pydantic-settings, loaded from `config.yaml`
   - `BaitConfig` is the main settings class with computed path properties (`data_dir`, `docs_output_dir`, `db_path`)
   - `get_embeddings(config)` — shared factory that creates the correct embedding model (HuggingFace or ANL Argo) based on config

2. **Data Ingestion** (`data_ingestion.py`)
   - Clones/updates documentation repositories from GitHub
   - Builds Sphinx documentation to HTML
   - Uses `ReadTheDocsLoader` to load HTML documentation
   - Chunks documents using `RecursiveCharacterTextSplitter` (configurable size/overlap)
   - Embeds using the shared `get_embeddings()` factory
   - Stores in ChromaDB at `.bait-{project.name}/chroma_db`

3. **Retriever** (`retriever.py`)
   - Shared utility for accessing ChromaDB
   - Returns top-k most relevant document chunks (configurable, default k=3)
   - Can be tested standalone: `python -m tomobait.retriever "test query"`

4. **Agents** (`agents.py`)
   - Defines the AG2 (Autogen) two-agent system:
     - `doc_expert` (AssistantAgent): LLM-powered agent that answers questions
     - `tool_worker` (UserProxyAgent): Executes the `query_documentation` tool
   - Handles LLM provider switching (standard API key providers vs ANL Argo)
   - `run_agent_chat(user_question)` — main entry point for running a Q&A session

5. **Backend** (`app.py`)
   - FastAPI server exposing `/chat` endpoint (POST)
   - Also exposes `/config` GET/POST endpoints
   - Bridges HTTP requests to `run_agent_chat()`

6. **Frontend** (`frontend.py`)
   - Gradio chatbot interface with a single Chat tab
   - Makes HTTP requests to the FastAPI backend
   - Serves documentation images through Gradio's `allowed_paths` mechanism

## Key Configuration

All configuration is centralized in `config.yaml`:

- **Project Settings**: `project.name` and `project.data_dir` (e.g., ".bait-tomo")
- **Documentation**: Git repos and local folders
- **Retriever**: k, search_type, score_threshold
- **Embedding**: Provider (`huggingface` or `anl_argo`), model name, device. Must match between ingestion and retrieval.
- **LLM**: Provider, model, api_type, argo_base_url, system_message
- **Text Processing**: chunk_size, chunk_overlap
- **Server**: Ports (backend 8001, frontend 8000)

## Environment Variables

For standard LLM providers, set the API key as an environment variable (or in a `.env` file):

- `GEMINI_API_KEY` — Google Gemini (default)
- `OPENAI_API_KEY` — OpenAI
- `ANTHROPIC_API_KEY` — Anthropic
- `AZURE_OPENAI_API_KEY` — Azure OpenAI

ANL Argo uses `api_key` and `argo_base_url` fields directly in `config.yaml` instead of environment variables.

## Code Style

- Ruff for linting and formatting
- Line length: 88 characters
- Linting rules: Pyflakes (F), pycodestyle (E), isort (I)
- Double quotes, space indentation

## Important Implementation Details

### Agent Termination Logic

The `tool_worker` agent terminates when it receives a message WITHOUT tool calls. This means the conversation flow is:
1. User question sent to doc_expert
2. doc_expert generates tool call
3. tool_worker executes tool, returns results
4. doc_expert generates final answer (no tool calls)
5. Conversation terminates

### Documentation Sources

The system can ingest from multiple sources:
- **Git Repositories**: Cloned to `.bait-{name}/documentation/`, Sphinx docs built automatically
- **Local Folders**: Pre-built documentation can be loaded directly

Default configuration includes the 2-BM tomography beamline documentation. The ingestion process expects a Sphinx documentation structure with a `docs/` directory.

### Image Handling in Frontend

The Gradio frontend resolves documentation image paths to absolute filesystem paths and serves them through Gradio's `allowed_paths` mechanism.

### Path Resolution

All paths are computed properties on `BaitConfig`:
- `config.data_dir` → `.bait-{project.name}/`
- `config.db_path` → `.bait-{project.name}/chroma_db`
- `config.docs_output_dir` → `.bait-{project.name}/documentation`

No hardcoded paths exist outside of `config.yaml`.
