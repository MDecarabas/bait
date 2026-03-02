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

# Run CLI interface
uv run python -m tomobait.cli "Your question here"
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
  ├── conversations/      # Saved chat history
  └── documentation/      # Cloned repos and built docs
  ```

### Three-Layer System

1. **Data Ingestion Layer** (`data_ingestion.py`)
   - Clones/updates documentation repositories from GitHub
   - Builds Sphinx documentation to HTML
   - Uses `ReadTheDocsLoader` to load HTML documentation
   - Chunks documents using `RecursiveCharacterTextSplitter` (configurable size/overlap)
   - Embeds using HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, no API)
   - Stores in ChromaDB at `.bait-{project.name}/chroma_db`
   - Embeds resource definitions from `config.yaml`

2. **Backend/Agent Layer** (`app.py`)
   - FastAPI server exposing `/chat` endpoint
   - Uses Autogen (AG2) multi-agent framework with Gemini 2.5 Flash
   - **Two-agent system**:
     - `doc_expert` (AssistantAgent): LLM-powered agent that answers questions
     - `tool_worker` (UserProxyAgent): Executes the `query_documentation` tool
   - Agent workflow: User question → doc_expert calls tool → tool_worker retrieves from ChromaDB → doc_expert synthesizes answer
   - Requires `GEMINI_API_KEY` environment variable

3. **Frontend Layer** (`frontend.py`)
   - Gradio chatbot interface with four tabs: Chat, History, Configuration, Setup
   - Makes HTTP requests to FastAPI backend
   - Handles image rendering from documentation (parses markdown image paths)
   - Serves static files from project documentation directory
   - Provides hot-reload configuration editing
   - AI-powered configuration generation

### Retriever Module (`retriever.py`)

- Shared utility for accessing ChromaDB
- Returns top 3 most relevant document chunks (k=3)
- Can be tested standalone: `python src/tomobait/retriever.py "test query"`

### CLI Interface (`cli.py`)

- Simple argparse wrapper around `run_agent_chat()`
- Provides command-line access to the agent system

## Key Configuration

All configuration is centralized in `config.yaml`:

- **Project Settings**: Define `project.name` and `project.data_dir` (e.g., ".bait-tomo")
- **Storage**: Conversations directory (defaults to `{data_dir}/conversations`)
- **Documentation**: Git repos, local folders, and resource definitions all in one section
- **Retriever**: ChromaDB path (defaults to `{data_dir}/chroma_db`), embedding model, search parameters
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (must match between ingestion and retrieval)
- **LLM Providers**: Supports Gemini (default), OpenAI, Anthropic, Azure, and ANL Argo via Autogen
- **Hot-Reload**: Configuration changes are detected and applied automatically
- **Ports**: Backend on 8001, Frontend on 8000

All paths can be explicitly set in config.yaml or left null to use computed defaults based on `project.data_dir`.

## Environment Variables

The system supports multiple LLM providers. Set up the appropriate API key based on your configuration:

### Gemini (Default)
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your key from: https://aistudio.google.com/app/apikey

### OpenAI
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
Get your key from: https://platform.openai.com/api-keys

Supported models: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-3.5-turbo`

### Anthropic (Claude)
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
Get your key from: https://console.anthropic.com/settings/keys

Supported models: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### Azure OpenAI
```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
```
Get it from your Azure portal.

### ANL Argo
ANL Argo is an internal LLM service that doesn't require API keys. Instead, configure in `config.yaml`:
```yaml
llm:
  api_type: anl_argo
  anl_api_url: https://your-anl-argo-endpoint/api/llm
  anl_user: your_anl_username
  anl_model: llama-2-70b
```

Supported models: `llama-2-70b`, `mixtral-8x7b` (check with your ANL Argo administrator for available models)

**Note**: For standard providers, copy `.env.example` to `.env` and fill in your API key. The provider and model can be configured via the Configuration tab in the web interface or by editing `config.yaml`.

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
- **Resource Definitions**: Beamlines, software packages, and organizations defined in `config.yaml` are embedded as searchable documents

Default configuration includes the 2-BM tomography beamline documentation. The ingestion process expects a Sphinx documentation structure with a `docs/` directory.

### Image Handling in Frontend

The Gradio frontend has custom logic to:
- Parse image paths from agent responses
- Resolve relative paths to absolute paths in the project documentation directory
- Serve images through Gradio's `allowed_paths` mechanism

### Path Resolution

All paths in the codebase are computed from `config.yaml`:
- `config.get_data_dir()` → `.bait-{project.name}/`
- `config.get_db_path()` → `.bait-{project.name}/chroma_db`
- `config.get_conversations_dir()` → `.bait-{project.name}/conversations`
- `config.get_docs_output_dir()` → `.bait-{project.name}/documentation`

This ensures no hardcoded paths exist outside of config.yaml.
