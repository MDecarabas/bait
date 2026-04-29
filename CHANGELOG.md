# Changelog

All notable changes to TomoBait will be documented in this file.

## [0.1.0] - 2026-04-01

### Changed
- Migrated agent orchestration from AG2/Autogen to LangGraph with a router-based architecture
- Replaced monolithic agent with a multi-agent system: router classifies questions, dispatches to doc_agent or bits_agent
- Simplified frontend from ~1400 lines to a clean Gradio chat interface with sidebar history
- Centralized configuration via pydantic-settings loaded from config.yaml
- Switched to ANL Argo as the LLM and embedding provider
- Restructured data ingestion to support multiple git repos and local folders

### Added
- LangGraph router that classifies questions as "documentation" or "device" and dispatches to the appropriate agent
- BITS device agent that loads device_skills.md from the BITS installation and answers device-related questions
- Per-agent LLM configuration overrides (model, api_key, base_url) in config.yaml
- Chat history CRUD endpoints (save, load, list, delete) with auto-generated titles
- Path traversal protection on all chat history file operations
- Foundational test suite (16 tests covering config, utils, and app endpoints)
- BITS integration plan for ophyd device control via ophyd-websocket
- TODOS.md tracking safety model for device write operations

### Removed
- AG2/Autogen dependency and agent architecture
- Old ANL LLM wrapper module (anl_llm.py)
- Config generator and config watcher modules
- Storage module (replaced by simpler file-based chat history)
- CLI interface (replaced by Gradio frontend)
- .env.example and GEMINI.md (configuration now centralized in config.yaml)
