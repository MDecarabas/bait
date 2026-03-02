# TomoBait 🔬

A RAG (Retrieval-Augmented Generation) system for tomography beamline documentation at the Advanced Photon Source (APS). TomoBait ingests Sphinx documentation, indexes it in a vector database, and provides an AI-powered conversational interface for querying beamline documentation.

## 🌟 Features

- **Multi-Source Documentation**: Index from Git repositories and local folders
- **AI-Powered Q&A**: Ask questions in natural language and get accurate answers from documentation
- **Multiple LLM Providers**: Support for Gemini, OpenAI, Anthropic, Azure, and ANL Argo
- **Hot-Reload Configuration**: Update settings without restarting the application

- **Conversation History**: Save and resume conversations
- **Web Interface**: Modern Gradio-based UI with chat, history, configuration, and setup tabs

## 📦 Package Structure

```mermaid
graph TB
    subgraph "TomoBait Architecture"
        UI[Frontend Layer<br/>Gradio Web Interface]
        API[Backend Layer<br/>FastAPI Server]
        DB[(Vector Database<br/>ChromaDB)]
        DOCS[Documentation Sources<br/>Git Repos & Local Folders]

        UI -->|HTTP /chat| API
        API -->|Query| DB
        DOCS -->|Ingest| DB

        subgraph "Frontend Components"
            CHAT[Chat Tab]
            HIST[History Tab]
            CFG[Configuration Tab]
            SETUP[Setup Tab]
        end

        subgraph "Backend Components"
            AGENTS[Autogen Agents]
            TOOL[Query Tool]
            RET[Retriever]

            AGENTS -->|Execute| TOOL
            TOOL -->|Retrieve| RET
            RET -->|Query| DB
        end

        subgraph "Data Pipeline"
            CLONE[Clone Repos]
            BUILD[Build Sphinx]
            CHUNK[Chunk Text]
            EMBED[Embed Chunks]

            CLONE --> BUILD
            BUILD --> CHUNK
            CHUNK --> EMBED
            EMBED --> DB
        end
    end

    style UI fill:#e1f5ff
    style API fill:#fff3e0
    style DB fill:#f3e5f5
    style DOCS fill:#e8f5e9
```

## 🏗️ Module Structure

```mermaid
graph LR
    subgraph "src/tomobait"
        APP[app.py<br/>FastAPI Backend]
        FRONT[frontend.py<br/>Gradio UI]
        ING[data_ingestion.py<br/>Doc Processor]
        RET[retriever.py<br/>Vector Search]
        CFG[config.py<br/>Config Models]
        CFGWATCH[config_watcher.py<br/>Hot Reload]
        STORE[storage.py<br/>Conversation DB]
        CLI[cli.py<br/>CLI Interface]
    end

    FRONT -->|HTTP| APP
    APP --> RET
    APP --> CFG
    APP --> CFGWATCH
    ING --> RET
    ING --> CFG
    FRONT --> STORE
    CLI --> APP

    style APP fill:#4caf50
    style FRONT fill:#2196f3
    style ING fill:#ff9800
    style RET fill:#9c27b0
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd tomo-bait
```

2. **Create a virtual environment and install dependencies**
```bash
uv venv
uv pip install -e .
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API key (GEMINI_API_KEY, OPENAI_API_KEY, etc.)
```

4. **Ingest documentation** (first time only)
```bash
uv run python -m tomobait.data_ingestion
```

5. **Start the application**
```bash
# Terminal 1: Start backend
uv run start-backend

# Terminal 2: Start frontend
uv run start-frontend
```

6. **Access the UI**

Open your browser to `http://localhost:8000`

## 🔧 Configuration

TomoBait uses a centralized `config.yaml` configuration file with hot-reload support.

### Configuration Sections

```yaml
project:
  name:                  # Project identifier (used in directory naming)
  data_dir:              # Base directory for all project data (e.g., .bait-tomo)

storage:
  conversations_dir:     # Directory for conversation storage (defaults to {data_dir}/conversations)

documentation:
  git_repos:             # List of Git repository URLs
  local_folders:         # List of local folder paths
  docs_output_dir:       # Where to store documentation (defaults to {data_dir}/documentation)
  resources:             # Reference resources (beamlines, software, organizations, etc.)

retriever:
  db_path:               # ChromaDB storage path (defaults to {data_dir}/chroma_db)
  embedding_model:       # HuggingFace model name
  k:                     # Number of documents to retrieve
  search_type:           # similarity, mmr, or similarity_score_threshold

llm:
provider:          # Environment variable name for API key
  model:                # Model name (gemini-2.5-flash, gpt-4, etc.)
  api_type:             # google, openai, azure, anthropic
  system_message:       # System prompt for the agent

text_processing:
  chunk_size:           # Text chunk size (100-5000)
  chunk_overlap:        # Overlap between chunks (0-1000)

server:
  backend_host:         # Backend server host
  backend_port:         # Backend server port
  frontend_host:        # Frontend server host
  frontend_port:        # Frontend server port
```

### Switching LLM Providers

You can switch between LLM providers in the Configuration tab or by editing `config.yaml`:

**Gemini (Default)**
```yaml
llm:
provider: GEMINI_API_KEY
  model: gemini-2.5-flash
  api_type: google
```

**OpenAI**
```yaml
llm:
provider: OPENAI_API_KEY
  model: gpt-4
  api_type: openai
```

**Anthropic (Claude)**
```yaml
llm:
provider: ANTHROPIC_API_KEY
  model: claude-3-opus
  api_type: anthropic
```

**ANL Argo** (Internal LLM service)
```yaml
llm:
  api_type: anl_argo
  anl_api_url: https://your-anl-argo-endpoint/api/llm
  anl_user: your_anl_username
  anl_model: llama-2-70b
```

## 📋 Available Commands

```bash
# Development
uv run start-backend   # Start FastAPI backend
uv run start-frontend  # Start Gradio frontend
uv run python -m tomobait.cli "query" # Run CLI interface

# Data Management
uv run python -m tomobait.data_ingestion         # Ingest documentation into vector DB

# Code Quality
ruff check .           # Check code style
ruff format .         # Format code with ruff
```

## 🏛️ System Architecture

### Three-Layer Design

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Agent
    participant Tool
    participant ChromaDB

    User->>Frontend: Ask question
    Frontend->>Backend: POST /chat
    Backend->>Agent: Initialize chat
    Agent->>Tool: Call query_documentation
    Tool->>ChromaDB: Retrieve k docs
    ChromaDB-->>Tool: Return chunks
    Tool-->>Agent: Return context
    Agent->>Agent: Generate answer
    Agent-->>Backend: Final response
    Backend-->>Frontend: JSON response
    Frontend-->>User: Display answer
```

### 1. Data Ingestion Layer

- Clones Git repositories or reads local folders
- Builds Sphinx documentation to HTML
- Chunks documents using `RecursiveCharacterTextSplitter`
- Embeds chunks using HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Stores in ChromaDB vector database

### 2. Backend/Agent Layer

- FastAPI server with REST API
- Two-agent system using Autogen (AG2):
  - `doc_expert`: LLM-powered agent that answers questions
  - `tool_worker`: Executes the `query_documentation` tool
- Agent workflow:
  1. User question → doc_expert
  2. doc_expert calls query_documentation tool
  3. tool_worker retrieves relevant docs from ChromaDB
  4. doc_expert synthesizes answer from context

### 3. Frontend Layer

- Gradio-based web interface
- Four tabs:
  - **Chat**: Conversational interface
  - **History**: View and resume past conversations
  - **Configuration**: Edit all settings with hot-reload
- Auto-save conversations
- Image rendering from documentation

## 🔬 Tomography Resources

TomoBait is configured with comprehensive resources for APS tomography beamlines:

- **Beamlines**: 2-BM, 32-ID (TXM)
- **Reconstruction Tools**: TomoPy, ASTRA Toolbox, scikit-image
- **Beamline Control**: tomoscan, dmagic, pyEPICS
- **Data Processing**: tomopy-cli, Xi-CAM, dxfile
- **Visualization**: Tomviz, napari, ParaView
- **File Formats**: h5py, nexusformat, tifffile

See `config.yaml` for the complete list with links.

## 🤝 Contributing

### Code Style

- Ruff for linting and formatting
- Line length: 88 characters
- Double quotes, space indentation

### Development Workflow

1. Make changes
2. Run `ruff format .` to format code
3. Run `ruff check .` to check style
4. Test changes locally
5. Commit and push

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Advanced Photon Source (APS) at Argonne National Laboratory
- 2-BM and 32-ID beamline teams
- TomoPy and related open-source projects

## 📚 Documentation

For detailed development guidance, see [CLAUDE.md](CLAUDE.md).

## 🐛 Troubleshooting

### Backend won't start
- Check that `GEMINI_API_KEY` (or your chosen provider's key) is set in `.env`
- Verify ChromaDB exists: run `uv run python -m tomobait.data_ingestion` if needed

### Frontend can't connect
- Ensure backend is running on port 8001
- Check firewall settings

### No documents retrieved
- Verify project data directory exists (e.g., `.bait-tomo/`)
- Check ChromaDB path in config.yaml (defaults to `.bait-tomo/chroma_db`)
- Re-run ingestion: `uv run python -m tomobait.data_ingestion`
- Check that embedding model matches between ingestion and retrieval

## 📞 Support

For issues and questions, please open an issue on GitHub.
