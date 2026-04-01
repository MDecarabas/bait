# TomoBait Multi-Agent Enhancement Plan

**Status**: Planning
**Created**: 2025-11-20
**Target Completion**: 7 weeks
**Scope**: Full multi-agent system with specialized agents, advanced RAG, and orchestration

---

## Executive Summary

Transform TomoBait from a simple 2-agent Q&A system into a sophisticated multi-agent documentation assistant. This plan adds specialized agents (Query Analyzer, Citation Specialist, Beamline Expert), implements advanced RAG techniques (hybrid search, reranking, query reformulation), and creates a robust orchestration layer for complex multi-agent workflows.

**Key Metrics**:
- Current: 2 agents, 1 tool, simple vector search
- Target: 5+ specialized agents, 10+ tools, hybrid search with reranking
- Expected: 2-3x better retrieval quality, 10x more sophisticated question handling

---

## Current System Analysis

### Existing Architecture (Baseline)

**Location**: `/home/raf/workspace_corgi/tomo-bait/src/tomobait/app.py`

**Current Agents**:
1. `doc_expert` (AssistantAgent) - LLM-powered documentation expert
2. `tool_worker` (UserProxyAgent) - Tool execution proxy (no LLM)

**Current Tools**:
1. `query_documentation` - Retrieves top k=3 documents from ChromaDB using semantic search

**Current Workflow**:
```
User Question
    ↓
tool_worker.initiate_chat(doc_expert)
    ↓
doc_expert generates tool call → query_documentation
    ↓
tool_worker executes tool → retrieves from ChromaDB (k=3)
    ↓
tool_worker returns context to doc_expert
    ↓
doc_expert generates final answer
    ↓
Response returned to user
```

**Current Retrieval**:
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Chunking: RecursiveCharacterTextSplitter (size=1000, overlap=200)
- Search: Standard vector similarity via LangChain Chroma
- Results: Top k=3 documents

**Current LLM Support**:
- Gemini (default)
- OpenAI
- Anthropic
- Azure OpenAI
- ANL Argo (custom internal service)

**Existing Dependencies** (from `pyproject.toml`):
- `ag2[gemini,mcp]` - Autogen with Gemini and MCP support
- `langchain-community`
- `chromadb`
- `sentence-transformers`
- `gradio`
- `pydantic`
- `watchdog` (for config hot-reload)

### Identified Limitations

1. **Simple Agent Architecture**: Only 2 agents with minimal interaction
2. **Limited Tool Ecosystem**: Single retrieval tool, no tool chaining
3. **No MCP Integration**: Package includes MCP support but unused
4. **Basic RAG Pipeline**: Simple vector search, no reranking or hybrid search
5. **No Agent Specialization**: Single general-purpose agent
6. **No Orchestration**: Linear workflow only, no parallel or hierarchical patterns
7. **No Agent Memory**: No context beyond current conversation
8. **No Performance Monitoring**: No metrics or evaluation framework

---

## Phase 1: Core Infrastructure (Week 1)

### 1.1 Agent Framework

**Goal**: Create extensible agent management system

**New Files**:
- `src/tomobait/agents/__init__.py`
- `src/tomobait/agents/base.py`
- `src/tomobait/agents/registry.py`

**Implementation**:

```python
# src/tomobait/agents/base.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from autogen import AssistantAgent, UserProxyAgent
from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Configuration for a single agent"""
    name: str
    type: str  # 'assistant' or 'user_proxy'
    system_message: str
    llm_config: Optional[Dict[str, Any]] = None
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"
    code_execution_config: bool = False
    description: str = ""
    enabled: bool = True

class BaseAgentFactory(ABC):
    """Base class for agent factories"""

    @abstractmethod
    def create_agent(self, config: AgentConfig, llm_config: Dict[str, Any]) -> Any:
        """Create an agent instance"""
        pass

    @abstractmethod
    def get_default_config(self) -> AgentConfig:
        """Get default configuration for this agent type"""
        pass

class AssistantAgentFactory(BaseAgentFactory):
    """Factory for creating AssistantAgent instances"""

    def create_agent(self, config: AgentConfig, llm_config: Dict[str, Any]) -> AssistantAgent:
        return AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=llm_config,
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
        )

class ToolWorkerFactory(BaseAgentFactory):
    """Factory for creating UserProxyAgent tool workers"""

    def create_agent(self, config: AgentConfig, llm_config: Dict[str, Any]) -> UserProxyAgent:
        return UserProxyAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=False,
            human_input_mode=config.human_input_mode,
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            code_execution_config=config.code_execution_config,
        )
```

```python
# src/tomobait/agents/registry.py
from typing import Dict, Type, Optional, List
from .base import BaseAgentFactory, AgentConfig

class AgentRegistry:
    """Registry for managing agent factories and instances"""

    def __init__(self):
        self._factories: Dict[str, BaseAgentFactory] = {}
        self._configs: Dict[str, AgentConfig] = {}
        self._instances: Dict[str, Any] = {}

    def register_factory(self, agent_type: str, factory: BaseAgentFactory):
        """Register an agent factory"""
        self._factories[agent_type] = factory

    def register_config(self, agent_name: str, config: AgentConfig):
        """Register agent configuration"""
        self._configs[agent_name] = config

    def create_agent(self, agent_name: str, llm_config: Dict[str, Any]) -> Any:
        """Create an agent instance"""
        if agent_name in self._instances:
            return self._instances[agent_name]

        config = self._configs.get(agent_name)
        if not config or not config.enabled:
            raise ValueError(f"No enabled config found for agent: {agent_name}")

        factory = self._factories.get(config.type)
        if not factory:
            raise ValueError(f"No factory registered for agent type: {config.type}")

        agent = factory.create_agent(config, llm_config)
        self._instances[agent_name] = agent
        return agent

    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return [name for name, config in self._configs.items() if config.enabled]

    def get_agent(self, agent_name: str) -> Optional[Any]:
        """Get an existing agent instance"""
        return self._instances.get(agent_name)

    def clear_instances(self):
        """Clear all agent instances (useful for config reload)"""
        self._instances.clear()

# Global registry instance
agent_registry = AgentRegistry()
```

**Configuration Schema Addition** (`src/tomobait/config.py`):

```python
class AgentConfigModel(BaseModel):
    """Configuration for a single agent"""
    name: str
    type: str = "assistant"  # 'assistant' or 'user_proxy'
    system_message: str
    description: str = ""
    enabled: bool = True
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"
    code_execution_config: bool = False

class AgentsConfig(BaseModel):
    """Configuration for all agents"""
    agents: List[AgentConfigModel] = Field(default_factory=list)
    default_workflow: str = "sequential"  # 'sequential', 'parallel', 'hierarchical'

class Config(BaseModel):
    # ... existing fields ...
    agents: Optional[AgentsConfig] = None
```

**config.yaml Update**:

```yaml
# Agent configurations
agents:
  default_workflow: sequential
  agents:
    - name: query_analyzer
      type: assistant
      enabled: true
      description: "Analyzes user questions and determines optimal retrieval strategy"
      system_message: |
        You are a Query Analyzer expert. Your role is to:
        1. Analyze user questions to understand intent
        2. Classify question type (factual, how-to, troubleshooting, comparison, code-related)
        3. Extract key entities (beamline names, software packages, technical terms)
        4. Determine the optimal retrieval strategy
        5. Route to appropriate specialist agents

        For each question, provide:
        - Question type classification
        - Extracted entities
        - Recommended retrieval strategy (semantic, keyword, hybrid)
        - Suggested specialist agents to consult
        - Query refinements or expansions

        Be concise and structured in your analysis.

    - name: doc_expert
      type: assistant
      enabled: true
      description: "Main documentation expert for answering questions"
      system_message: |
        You are a helpful documentation assistant for the 2-BM tomography beamline at Argonne National Laboratory.
        Your role is to answer questions accurately based on the retrieved documentation.

        Guidelines:
        - Always cite your sources with specific documentation sections
        - If information is not in the retrieved documents, say so clearly
        - Provide code examples when relevant
        - Use technical terminology appropriately
        - Format responses in clear markdown
        - Include links to relevant documentation pages

    - name: citation_specialist
      type: assistant
      enabled: true
      description: "Extracts precise citations and formats references"
      system_message: |
        You are a Citation Specialist. Your role is to:
        1. Extract precise citations from documentation
        2. Generate proper markdown references with section names
        3. Link answers to specific documentation sections
        4. Create formatted bibliographies
        5. Track documentation versions when available

        For each citation:
        - Include document title and section
        - Provide full URL or file path
        - Add relevant line numbers or page numbers
        - Format consistently in markdown

        Example format:
        [1] Section Title, Document Name, URL#section

    - name: beamline_expert
      type: assistant
      enabled: true
      description: "Specialized knowledge about APS beamlines"
      system_message: |
        You are a Beamline Expert specializing in Advanced Photon Source (APS) beamlines.
        You have deep knowledge of:
        - 2-BM tomography beamline specifications and capabilities
        - Other APS beamlines for comparison
        - Experimental techniques and best practices
        - Beamline equipment and configurations

        Your expertise includes:
        - Recommending appropriate beamlines for user needs
        - Comparing different beamline capabilities
        - Explaining technical specifications
        - Troubleshooting beamline-related issues

        Always provide accurate technical details and cite beamline documentation.

    - name: tool_worker
      type: user_proxy
      enabled: true
      description: "Executes tools for document retrieval"
      system_message: "You are a tool execution agent. Execute the requested tools and return results."
      human_input_mode: "NEVER"
      code_execution_config: false
```

### 1.2 Enhanced Tool System

**Goal**: Create flexible tool registration and discovery system

**New Files**:
- `src/tomobait/tools/__init__.py`
- `src/tomobait/tools/base.py`
- `src/tomobait/tools/search.py`
- `src/tomobait/tools/document.py`
- `src/tomobait/tools/registry.py`

**Implementation**:

```python
# src/tomobait/tools/base.py
from typing import Callable, Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ToolMetadata(BaseModel):
    """Metadata for a tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str = "general"  # search, document, code, comparison
    enabled: bool = True

class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}

    def register(self, metadata: ToolMetadata, func: Callable):
        """Register a tool"""
        self._tools[metadata.name] = func
        self._metadata[metadata.name] = metadata

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata"""
        return self._metadata.get(name)

    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """List available tools"""
        tools = []
        for name, metadata in self._metadata.items():
            if enabled_only and not metadata.enabled:
                continue
            if category and metadata.category != category:
                continue
            tools.append(name)
        return tools

    def get_tool_schemas(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get Autogen-compatible tool schemas"""
        schemas = []
        for name, metadata in self._metadata.items():
            if enabled_only and not metadata.enabled:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": metadata.parameters
                }
            })
        return schemas

# Global tool registry
tool_registry = ToolRegistry()
```

```python
# src/tomobait/tools/search.py
from typing import List, Dict, Any, Optional
from ..retriever import query_vector_db
from .base import tool_registry, ToolMetadata
import logging

logger = logging.getLogger(__name__)

def semantic_search(query: str, k: int = 3) -> str:
    """
    Perform semantic vector search on documentation.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)

    Returns:
        Formatted string with search results and metadata
    """
    try:
        results = query_vector_db(query, k=k)

        if not results or len(results) == 0:
            return "No relevant documentation found for this query."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            metadata = doc.metadata

            # Extract source information
            source = metadata.get('source', 'Unknown source')

            # Filter out internal config sources for display
            if 'config_resource' in source:
                continue

            # Extract URLs from metadata
            urls = []
            for url_key in ['documentation', 'official_page', 'github', 'pypi', 'source']:
                if url_key in metadata and metadata[url_key]:
                    urls.append(f"{url_key}: {metadata[url_key]}")

            url_section = f"\nURLs: {', '.join(urls)}" if urls else ""

            formatted_results.append(
                f"Result {i}:\n"
                f"Source: {source}{url_section}\n"
                f"Content:\n{content}\n"
            )

        if not formatted_results:
            return "No relevant documentation found for this query."

        return "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error in semantic_search: {e}")
        return f"Error performing search: {str(e)}"

def keyword_search(query: str, k: int = 3) -> str:
    """
    Perform keyword-based search on documentation using BM25.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)

    Returns:
        Formatted string with search results
    """
    # TODO: Implement BM25 keyword search
    # For now, fallback to semantic search
    return semantic_search(query, k)

def hybrid_search(query: str, k: int = 3, semantic_weight: float = 0.7) -> str:
    """
    Perform hybrid search combining semantic and keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)
        semantic_weight: Weight for semantic results (0-1, default: 0.7)

    Returns:
        Formatted string with merged search results
    """
    # TODO: Implement true hybrid search with RRF
    # For now, use semantic search with higher k
    return semantic_search(query, k=k * 2)

# Register search tools
tool_registry.register(
    ToolMetadata(
        name="semantic_search",
        description="Search documentation using semantic vector similarity. Best for conceptual questions and finding related topics.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        },
        category="search"
    ),
    semantic_search
)

tool_registry.register(
    ToolMetadata(
        name="keyword_search",
        description="Search documentation using keyword matching (BM25). Best for exact terms and specific technical keywords.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query with keywords"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        },
        category="search"
    ),
    keyword_search
)

tool_registry.register(
    ToolMetadata(
        name="hybrid_search",
        description="Search documentation using both semantic and keyword methods, merging results. Best for comprehensive search coverage.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                },
                "semantic_weight": {
                    "type": "number",
                    "description": "Weight for semantic results (0-1, default: 0.7)",
                    "default": 0.7
                }
            },
            "required": ["query"]
        },
        category="search"
    ),
    hybrid_search
)
```

```python
# src/tomobait/tools/document.py
from typing import Optional, List
from .base import tool_registry, ToolMetadata

def get_document_outline(doc_path: str) -> str:
    """
    Get the table of contents / outline for a documentation file.

    Args:
        doc_path: Path to the documentation file

    Returns:
        Formatted outline with sections and subsections
    """
    # TODO: Implement document outline extraction
    return f"Outline for {doc_path}: (Not yet implemented)"

def get_section(doc_path: str, section_name: str) -> str:
    """
    Retrieve a specific section from a documentation file.

    Args:
        doc_path: Path to the documentation file
        section_name: Name or heading of the section to retrieve

    Returns:
        Content of the specified section
    """
    # TODO: Implement section extraction
    return f"Section '{section_name}' from {doc_path}: (Not yet implemented)"

# Register document tools
tool_registry.register(
    ToolMetadata(
        name="get_document_outline",
        description="Get the table of contents or outline for a documentation file. Useful for understanding document structure.",
        parameters={
            "type": "object",
            "properties": {
                "doc_path": {
                    "type": "string",
                    "description": "Path to the documentation file"
                }
            },
            "required": ["doc_path"]
        },
        category="document"
    ),
    get_document_outline
)

tool_registry.register(
    ToolMetadata(
        name="get_section",
        description="Retrieve a specific section from a documentation file by section name or heading.",
        parameters={
            "type": "object",
            "properties": {
                "doc_path": {
                    "type": "string",
                    "description": "Path to the documentation file"
                },
                "section_name": {
                    "type": "string",
                    "description": "Name or heading of the section"
                }
            },
            "required": ["doc_path", "section_name"]
        },
        category="document"
    ),
    get_section
)
```

### 1.3 Orchestration Layer

**Goal**: Create flexible orchestration for multi-agent workflows

**New Files**:
- `src/tomobait/orchestration/__init__.py`
- `src/tomobait/orchestration/orchestrator.py`
- `src/tomobait/orchestration/workflows.py`

**Implementation**:

```python
# src/tomobait/orchestration/orchestrator.py
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class WorkflowType(str, Enum):
    """Types of agent workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ITERATIVE = "iterative"

class AgentOrchestrator:
    """Orchestrates multi-agent workflows"""

    def __init__(self, agent_registry, tool_registry):
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.workflows: Dict[str, Callable] = {}

    def register_workflow(self, name: str, workflow_func: Callable):
        """Register a workflow function"""
        self.workflows[name] = workflow_func

    async def execute_workflow(
        self,
        workflow_name: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a named workflow.

        Args:
            workflow_name: Name of the workflow to execute
            user_message: User's question or request
            context: Optional context dictionary

        Returns:
            Dictionary with workflow results
        """
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        try:
            result = await workflow(
                user_message=user_message,
                agent_registry=self.agent_registry,
                tool_registry=self.tool_registry,
                context=context or {}
            )
            return result
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_name}: {e}")
            raise

    def sequential_workflow(
        self,
        agent_sequence: List[str],
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute agents in sequence, passing results along the chain.

        Args:
            agent_sequence: List of agent names in execution order
            user_message: User's question
            context: Optional context

        Returns:
            Results from the final agent
        """
        current_message = user_message
        results = {}

        for agent_name in agent_sequence:
            agent = self.agent_registry.get_agent(agent_name)
            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue

            # Execute agent with current message
            # TODO: Implement actual agent execution logic
            logger.info(f"Executing agent: {agent_name}")

            # Store results
            results[agent_name] = {"message": current_message}

        return results

    def parallel_workflow(
        self,
        agent_list: List[str],
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multiple agents in parallel and merge results.

        Args:
            agent_list: List of agent names to execute in parallel
            user_message: User's question
            context: Optional context

        Returns:
            Merged results from all agents
        """
        results = {}

        # TODO: Implement actual parallel execution
        # For now, execute sequentially
        for agent_name in agent_list:
            agent = self.agent_registry.get_agent(agent_name)
            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue

            logger.info(f"Executing agent: {agent_name}")
            results[agent_name] = {"message": user_message}

        return results
```

```python
# src/tomobait/orchestration/workflows.py
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def standard_qa_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standard Q&A workflow:
    1. Query Analyzer analyzes the question
    2. Tool Worker retrieves relevant docs
    3. Doc Expert generates answer
    4. Citation Specialist adds citations

    Args:
        user_message: User's question
        agent_registry: Agent registry instance
        tool_registry: Tool registry instance
        context: Workflow context

    Returns:
        Final answer with citations
    """
    logger.info("Executing standard Q&A workflow")

    # Step 1: Analyze query
    query_analyzer = agent_registry.get_agent("query_analyzer")
    if query_analyzer:
        logger.info("Step 1: Analyzing query")
        # TODO: Execute query analyzer
        analysis = {
            "question_type": "factual",
            "entities": [],
            "retrieval_strategy": "semantic"
        }
    else:
        analysis = {"retrieval_strategy": "semantic"}

    # Step 2: Retrieve documents
    tool_worker = agent_registry.get_agent("tool_worker")
    if tool_worker:
        logger.info("Step 2: Retrieving documents")
        # TODO: Execute tool worker with appropriate search tool
        documents = "Retrieved documents..."
    else:
        documents = ""

    # Step 3: Generate answer
    doc_expert = agent_registry.get_agent("doc_expert")
    if doc_expert:
        logger.info("Step 3: Generating answer")
        # TODO: Execute doc expert
        answer = "Generated answer..."
    else:
        answer = "Unable to generate answer"

    # Step 4: Add citations
    citation_specialist = agent_registry.get_agent("citation_specialist")
    if citation_specialist:
        logger.info("Step 4: Adding citations")
        # TODO: Execute citation specialist
        final_answer = f"{answer}\n\n[Citations added]"
    else:
        final_answer = answer

    return {
        "answer": final_answer,
        "analysis": analysis,
        "documents": documents,
        "workflow": "standard_qa"
    }

async def beamline_expert_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Beamline-specific workflow:
    1. Query Analyzer identifies beamline entities
    2. Beamline Expert provides specialized knowledge
    3. Tool Worker retrieves supporting docs
    4. Citation Specialist adds references

    Args:
        user_message: User's question
        agent_registry: Agent registry instance
        tool_registry: Tool registry instance
        context: Workflow context

    Returns:
        Expert answer with citations
    """
    logger.info("Executing beamline expert workflow")

    # TODO: Implement beamline expert workflow

    return {
        "answer": "Beamline expert answer...",
        "workflow": "beamline_expert"
    }

async def iterative_refinement_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iterative refinement workflow:
    1. Query Analyzer analyzes question
    2. Tool Worker retrieves docs
    3. If results insufficient, Query Analyzer refines query
    4. Repeat steps 2-3 up to max_iterations
    5. Doc Expert generates final answer

    Args:
        user_message: User's question
        agent_registry: Agent registry instance
        tool_registry: Tool registry instance
        context: Workflow context

    Returns:
        Refined answer with iteration history
    """
    logger.info("Executing iterative refinement workflow")

    max_iterations = context.get("max_iterations", 3)
    iterations = []

    # TODO: Implement iterative refinement logic

    return {
        "answer": "Refined answer...",
        "iterations": iterations,
        "workflow": "iterative_refinement"
    }
```

---

## Phase 2: Specialized Agents (Week 2)

### 2.1 Query Analyzer Agent

**Goal**: Analyze questions and route to appropriate specialists

**New File**: `src/tomobait/agents/query_analyzer.py`

**Implementation**:

```python
# src/tomobait/agents/query_analyzer.py
from typing import Dict, Any, List
from autogen import AssistantAgent
from ..tools.base import tool_registry
import logging

logger = logging.getLogger(__name__)

QUERY_ANALYZER_SYSTEM_MESSAGE = """You are a Query Analyzer expert for a tomography beamline documentation system.

Your role is to analyze user questions and determine the optimal retrieval and response strategy.

For each question, provide a structured analysis:

1. **Question Type**: Classify as one of:
   - factual: Simple fact lookup
   - how-to: Step-by-step instructions
   - troubleshooting: Problem diagnosis and solutions
   - comparison: Comparing options or approaches
   - code-related: Code examples or API usage
   - beamline-specific: Questions about specific beamlines

2. **Key Entities**: Extract important entities:
   - Beamline names (e.g., "2-BM", "32-ID")
   - Software packages (e.g., "TomoPy", "Astra Toolbox")
   - Technical terms (e.g., "reconstruction", "ring artifacts")
   - File formats (e.g., "HDF5", "TIFF")

3. **Retrieval Strategy**: Recommend:
   - semantic: For conceptual questions
   - keyword: For exact technical terms
   - hybrid: For comprehensive coverage
   - multi-hop: For questions requiring multiple doc sections

4. **Recommended Agents**: Suggest which specialist agents to consult:
   - beamline_expert: For beamline-specific questions
   - doc_expert: For general documentation questions
   - citation_specialist: When precise citations are needed

5. **Query Refinements**: Suggest:
   - Expanded queries with acronyms spelled out
   - Alternative phrasings
   - Additional context to add

Output your analysis in this JSON format:
{
  "question_type": "factual|how-to|troubleshooting|comparison|code-related|beamline-specific",
  "entities": {
    "beamlines": ["2-BM"],
    "software": ["TomoPy"],
    "terms": ["reconstruction"]
  },
  "retrieval_strategy": "semantic|keyword|hybrid|multi-hop",
  "recommended_agents": ["beamline_expert", "doc_expert"],
  "query_refinements": [
    "original query with expanded acronyms",
    "alternative phrasing"
  ],
  "reasoning": "Brief explanation of your analysis"
}

Be concise and focus on actionable recommendations.
"""

class QueryAnalyzerAgent:
    """Factory for creating query analyzer agent"""

    @staticmethod
    def create(llm_config: Dict[str, Any]) -> AssistantAgent:
        """Create query analyzer agent instance"""

        return AssistantAgent(
            name="query_analyzer",
            system_message=QUERY_ANALYZER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            max_consecutive_auto_reply=3,
        )

    @staticmethod
    def parse_analysis(response: str) -> Dict[str, Any]:
        """Parse query analyzer response into structured data"""
        import json

        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse query analysis: {e}")

        # Return default analysis on parse failure
        return {
            "question_type": "factual",
            "entities": {},
            "retrieval_strategy": "semantic",
            "recommended_agents": ["doc_expert"],
            "query_refinements": [],
            "reasoning": "Parse failed, using defaults"
        }
```

### 2.2 Citation Specialist Agent

**New File**: `src/tomobait/agents/citation_specialist.py`

**Implementation**:

```python
# src/tomobait/agents/citation_specialist.py
from typing import Dict, Any
from autogen import AssistantAgent

CITATION_SPECIALIST_SYSTEM_MESSAGE = """You are a Citation Specialist for academic and technical documentation.

Your role is to extract precise citations from documentation and format them properly.

Guidelines:
1. **Citation Format**: Use this markdown format:
   [N] Section Title, Document Name, URL#section

   Example:
   [1] Installation Guide, TomoPy Documentation, https://tomopy.readthedocs.io/en/latest/install.html#installation

2. **Source Attribution**: For each fact or claim:
   - Identify the source document
   - Extract the specific section or heading
   - Include line numbers or page numbers if available
   - Provide the full URL or file path

3. **Bibliography**: Create a references section at the end:
   ## References
   [1] Source 1 details
   [2] Source 2 details

4. **Inline Citations**: Integrate citations naturally:
   - "According to the installation guide [1], TomoPy requires..."
   - "The 2-BM beamline specifications [2] indicate..."

5. **Version Tracking**: When available, note documentation version:
   - Git commit hash
   - Documentation build date
   - Software version

6. **Consistency**: Use consistent citation style throughout

7. **Verification**: Only cite sources that were actually provided in the context

Be precise and ensure every claim has a proper citation.
"""

class CitationSpecialistAgent:
    """Factory for creating citation specialist agent"""

    @staticmethod
    def create(llm_config: Dict[str, Any]) -> AssistantAgent:
        """Create citation specialist agent instance"""

        return AssistantAgent(
            name="citation_specialist",
            system_message=CITATION_SPECIALIST_SYSTEM_MESSAGE,
            llm_config=llm_config,
            max_consecutive_auto_reply=3,
        )
```

### 2.3 Beamline Expert Agent

**New File**: `src/tomobait/agents/beamline_expert.py`

**Implementation**:

```python
# src/tomobait/agents/beamline_expert.py
from typing import Dict, Any
from autogen import AssistantAgent

BEAMLINE_EXPERT_SYSTEM_MESSAGE = """You are a Beamline Expert specializing in Advanced Photon Source (APS) beamlines.

Your expertise covers:

**2-BM Tomography Beamline**:
- X-ray tomography and imaging
- High-resolution 3D imaging
- Sample preparation and mounting
- Data acquisition workflows
- Reconstruction techniques

**General APS Knowledge**:
- Beamline comparison and selection
- Experimental capabilities
- Technical specifications
- User support and training

**Key Responsibilities**:
1. Answer beamline-specific questions with technical accuracy
2. Compare different beamlines when asked
3. Recommend appropriate beamlines for user needs
4. Explain experimental techniques and best practices
5. Troubleshoot beamline-related issues
6. Provide guidance on sample preparation and data collection

**Response Guidelines**:
- Use precise technical terminology
- Cite beamline specifications and capabilities
- Provide practical experimental advice
- Include relevant links to beamline documentation
- Mention any limitations or constraints
- Suggest contact information for beamline staff when appropriate

**Example Topics**:
- "What is the spatial resolution of the 2-BM beamline?"
- "How do I prepare samples for tomography?"
- "Which beamline is best for studying metal samples?"
- "What reconstruction algorithms does 2-BM support?"

Be helpful, accurate, and technical in your responses.
"""

class BeamlineExpertAgent:
    """Factory for creating beamline expert agent"""

    @staticmethod
    def create(llm_config: Dict[str, Any]) -> AssistantAgent:
        """Create beamline expert agent instance"""

        return AssistantAgent(
            name="beamline_expert",
            system_message=BEAMLINE_EXPERT_SYSTEM_MESSAGE,
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
        )
```

### 2.4 Integration into app.py

**Modify**: `src/tomobait/app.py`

Add imports and initialize new agents:

```python
from .agents.registry import agent_registry, AgentRegistry
from .agents.query_analyzer import QueryAnalyzerAgent
from .agents.citation_specialist import CitationSpecialistAgent
from .agents.beamline_expert import BeamlineExpertAgent
from .tools.registry import tool_registry
from .tools import search, document  # This will register all tools
from .orchestration.orchestrator import AgentOrchestrator
from .orchestration.workflows import (
    standard_qa_workflow,
    beamline_expert_workflow,
    iterative_refinement_workflow
)

# Initialize orchestrator
orchestrator = AgentOrchestrator(agent_registry, tool_registry)

# Register workflows
orchestrator.register_workflow("standard_qa", standard_qa_workflow)
orchestrator.register_workflow("beamline_expert", beamline_expert_workflow)
orchestrator.register_workflow("iterative_refinement", iterative_refinement_workflow)

def initialize_agents(config: Config) -> AgentRegistry:
    """Initialize all agents from configuration"""

    llm_config = get_llm_config(config)

    # Create agents based on config
    if config.agents:
        for agent_config in config.agents.agents:
            if not agent_config.enabled:
                continue

            # Create agent based on type
            if agent_config.name == "query_analyzer":
                agent = QueryAnalyzerAgent.create(llm_config)
            elif agent_config.name == "citation_specialist":
                agent = CitationSpecialistAgent.create(llm_config)
            elif agent_config.name == "beamline_expert":
                agent = BeamlineExpertAgent.create(llm_config)
            # ... other agents

            # Register in registry
            agent_registry._instances[agent_config.name] = agent

    return agent_registry
```

---

## Phase 3: Advanced RAG Pipeline (Week 3)

### 3.1 Hybrid Search Implementation

**Goal**: Combine semantic and keyword search for better retrieval

**New File**: `src/tomobait/retrieval/hybrid_search.py`

**Add Dependencies** to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "rank-bm25>=0.2.2",  # BM25 keyword search
]
```

**Implementation**:

```python
# src/tomobait/retrieval/hybrid_search.py
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines semantic and keyword search using RRF (Reciprocal Rank Fusion)"""

    def __init__(self, vector_store, documents: List[Document], k: int = 3):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store
            documents: List of all documents for BM25 indexing
            k: Number of results to return
        """
        self.vector_store = vector_store
        self.k = k

        # Build BM25 index
        self._build_bm25_index(documents)

    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from documents"""
        logger.info(f"Building BM25 index from {len(documents)} documents")

        # Tokenize documents
        self.documents = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        logger.info("BM25 index built successfully")

    def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform semantic vector search"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        # Convert ChromaDB scores to similarity scores (higher is better)
        return [(doc, 1.0 / (1.0 + score)) for doc, score in results]

    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Return documents with scores
        return [(self.documents[i], scores[i]) for i in top_indices]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int = 60
    ) -> List[Document]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: RRF(d) = Σ 1 / (k + rank(d))
        where k is typically 60 (constant to reduce impact of high ranks)

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            k: RRF constant (default: 60)

        Returns:
            Merged and ranked documents
        """
        # Build RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Add semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

        # Add keyword results
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_id = doc.page_content[:100]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top k documents
        return [doc_map[doc_id] for doc_id, score in sorted_docs[:self.k]]

    def search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        k: int = None
    ) -> List[Document]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            semantic_weight: Weight for semantic search (0-1)
            k: Number of results (uses self.k if None)

        Returns:
            List of documents ranked by RRF
        """
        k = k or self.k

        # Retrieve more candidates for fusion
        candidate_k = k * 3

        # Perform both searches
        semantic_results = self._semantic_search(query, candidate_k)
        keyword_results = self._keyword_search(query, candidate_k)

        # Apply weights
        semantic_results = [(doc, score * semantic_weight) for doc, score in semantic_results]
        keyword_results = [(doc, score * (1 - semantic_weight)) for doc, score in keyword_results]

        # Merge using RRF
        merged_results = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        return merged_results[:k]
```

### 3.2 Reranking Pipeline

**Goal**: Improve result quality with cross-encoder reranking

**Add Dependencies** to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "sentence-transformers>=2.2.0",  # For cross-encoder models
]
```

**New File**: `src/tomobait/retrieval/reranker.py`

**Implementation**:

```python
# src/tomobait/retrieval/reranker.py
from typing import List, Tuple, Optional
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class DocumentReranker:
    """Reranks documents using a cross-encoder model"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        score_threshold: Optional[float] = None
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            score_threshold: Minimum score to keep (0-1)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.score_threshold = score_threshold
        logger.info("Cross-encoder model loaded")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Return top k results (returns all if None)

        Returns:
            List of (document, score) tuples sorted by score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Filter by threshold
        if self.score_threshold is not None:
            doc_scores = [
                (doc, score) for doc, score in doc_scores
                if score >= self.score_threshold
            ]

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        if top_k:
            return doc_scores[:top_k]
        return doc_scores

    def rerank_and_format(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> str:
        """
        Rerank documents and format as string.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of results to return

        Returns:
            Formatted string with reranked results
        """
        reranked = self.rerank(query, documents, top_k)

        if not reranked:
            return "No relevant documents found."

        formatted_results = []
        for i, (doc, score) in enumerate(reranked, 1):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown source')

            formatted_results.append(
                f"Result {i} (relevance: {score:.3f}):\n"
                f"Source: {source}\n"
                f"Content:\n{content}\n"
            )

        return "\n---\n".join(formatted_results)
```

### 3.3 Query Reformulation

**New File**: `src/tomobait/retrieval/query_reformulation.py`

**Implementation**:

```python
# src/tomobait/retrieval/query_reformulation.py
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)

# Common acronyms and expansions in tomography/beamline domain
ACRONYM_MAP = {
    "2-BM": "2-BM tomography beamline",
    "APS": "Advanced Photon Source",
    "XRF": "X-ray fluorescence",
    "CT": "computed tomography",
    "HDF5": "Hierarchical Data Format 5",
    "TIFF": "Tagged Image File Format",
    # Add more as needed
}

class QueryReformulator:
    """Reformulates queries for better retrieval"""

    def __init__(self, acronym_map: Dict[str, str] = None):
        """
        Initialize query reformulator.

        Args:
            acronym_map: Custom acronym expansion map
        """
        self.acronym_map = acronym_map or ACRONYM_MAP

    def expand_acronyms(self, query: str) -> str:
        """Expand known acronyms in query"""
        expanded = query
        for acronym, expansion in self.acronym_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(acronym) + r'\b'
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
        return expanded

    def generate_variants(self, query: str) -> List[str]:
        """
        Generate query variants for better coverage.

        Returns:
            List of query variants including original
        """
        variants = [query]

        # Add acronym-expanded version
        expanded = self.expand_acronyms(query)
        if expanded != query:
            variants.append(expanded)

        # Add question reformulations
        if query.endswith('?'):
            # Convert question to statement
            statement = query.rstrip('?')
            variants.append(statement)

        # Add domain context
        if any(term in query.lower() for term in ['beamline', 'tomography', 'reconstruction']):
            # Already has domain context
            pass
        else:
            # Add beamline context
            variants.append(f"{query} tomography beamline")

        return variants

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'is', 'are', 'the', 'a', 'an'}

        # Tokenize and filter
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords
```

---

## Phase 4: Agent Orchestration Patterns (Week 4)

### 4.1 Workflow Router

**Goal**: Automatically select appropriate workflow based on question type

**New File**: `src/tomobait/orchestration/router.py`

**Implementation**:

```python
# src/tomobait/orchestration/router.py
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class WorkflowRoute(str, Enum):
    """Available workflow routes"""
    STANDARD_QA = "standard_qa"
    BEAMLINE_EXPERT = "beamline_expert"
    CODE_GENERATION = "code_generation"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    ITERATIVE_REFINEMENT = "iterative_refinement"

class WorkflowRouter:
    """Routes questions to appropriate workflows based on analysis"""

    def __init__(self, orchestrator):
        """
        Initialize router.

        Args:
            orchestrator: AgentOrchestrator instance
        """
        self.orchestrator = orchestrator

        # Define routing rules
        self.routing_rules = {
            "beamline-specific": WorkflowRoute.BEAMLINE_EXPERT,
            "code-related": WorkflowRoute.CODE_GENERATION,
            "troubleshooting": WorkflowRoute.TROUBLESHOOTING,
            "comparison": WorkflowRoute.COMPARISON,
            "factual": WorkflowRoute.STANDARD_QA,
            "how-to": WorkflowRoute.STANDARD_QA,
        }

    def route(
        self,
        user_message: str,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine which workflow to use.

        Args:
            user_message: User's question
            query_analysis: Optional analysis from Query Analyzer

        Returns:
            Workflow name to execute
        """
        # If no analysis, use default
        if not query_analysis:
            logger.info("No query analysis available, using standard QA workflow")
            return WorkflowRoute.STANDARD_QA

        question_type = query_analysis.get("question_type", "factual")

        # Apply routing rules
        workflow = self.routing_rules.get(question_type, WorkflowRoute.STANDARD_QA)

        logger.info(f"Routing question type '{question_type}' to workflow '{workflow}'")

        return workflow

    async def route_and_execute(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route to appropriate workflow and execute.

        Args:
            user_message: User's question
            context: Optional context

        Returns:
            Workflow execution results
        """
        # First, run query analyzer if available
        query_analysis = None
        # TODO: Execute query analyzer

        # Route to workflow
        workflow_name = self.route(user_message, query_analysis)

        # Execute workflow
        result = await self.orchestrator.execute_workflow(
            workflow_name=workflow_name,
            user_message=user_message,
            context=context or {}
        )

        # Add routing metadata
        result["workflow_used"] = workflow_name
        result["query_analysis"] = query_analysis

        return result
```

### 4.2 Multi-Agent Collaboration

**Modify**: `src/tomobait/orchestration/workflows.py`

Add advanced collaboration patterns:

```python
async def parallel_retrieval_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parallel retrieval workflow:
    1. Query Analyzer analyzes question
    2. Multiple agents retrieve in parallel:
       - Beamline Expert (from resource knowledge)
       - Tool Worker (from vector DB)
       - (Future: External sources via MCP)
    3. Summary Agent synthesizes all inputs
    4. Citation Specialist adds references
    """
    logger.info("Executing parallel retrieval workflow")

    # Step 1: Analyze query
    analysis = {}  # TODO: Execute query analyzer

    # Step 2: Parallel retrieval
    # TODO: Execute multiple agents in parallel using asyncio.gather()

    # Step 3: Synthesize
    # TODO: Combine results from multiple agents

    # Step 4: Add citations
    # TODO: Execute citation specialist

    return {
        "answer": "Synthesized answer from multiple sources...",
        "workflow": "parallel_retrieval"
    }

async def consensus_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Consensus workflow for complex questions:
    1. Multiple expert agents answer independently
    2. Consensus agent identifies agreements and conflicts
    3. Summary agent presents unified answer with caveats
    """
    logger.info("Executing consensus workflow")

    # TODO: Implement consensus-based multi-agent collaboration

    return {
        "answer": "Consensus answer...",
        "workflow": "consensus"
    }
```

---

## Phase 5: Memory & Context Management (Week 5)

### 5.1 Agent Memory System

**Goal**: Enable agents to share knowledge and remember context

**New Files**:
- `src/tomobait/memory/__init__.py`
- `src/tomobait/memory/agent_memory.py`
- `src/tomobait/memory/conversation_memory.py`

**Implementation**:

```python
# src/tomobait/memory/agent_memory.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class MemoryEntry(BaseModel):
    """Single memory entry"""
    timestamp: datetime
    agent_name: str
    content: str
    metadata: Dict[str, Any] = {}
    importance: float = 0.5  # 0-1 scale

class AgentMemory:
    """Shared memory system for agents"""

    def __init__(self, max_entries: int = 100):
        """
        Initialize agent memory.

        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.knowledge_graph: Dict[str, Any] = {}

    def add_entry(
        self,
        agent_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ):
        """Add a memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now(),
            agent_name=agent_name,
            content=content,
            metadata=metadata or {},
            importance=importance
        )

        self.entries.append(entry)

        # Prune old entries if exceeding max
        if len(self.entries) > self.max_entries:
            # Keep most important entries
            self.entries.sort(key=lambda e: e.importance, reverse=True)
            self.entries = self.entries[:self.max_entries]
            # Re-sort by timestamp
            self.entries.sort(key=lambda e: e.timestamp)

    def get_recent(self, n: int = 10, agent_name: Optional[str] = None) -> List[MemoryEntry]:
        """Get recent memory entries"""
        entries = self.entries

        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]

        return entries[-n:]

    def search(self, query: str, n: int = 5) -> List[MemoryEntry]:
        """Search memory entries"""
        # Simple keyword search for now
        # TODO: Implement semantic search over memory
        query_lower = query.lower()

        matching = [
            e for e in self.entries
            if query_lower in e.content.lower()
        ]

        # Sort by importance
        matching.sort(key=lambda e: e.importance, reverse=True)

        return matching[:n]

    def update_knowledge_graph(self, entity: str, properties: Dict[str, Any]):
        """Update knowledge graph with entity information"""
        if entity not in self.knowledge_graph:
            self.knowledge_graph[entity] = {}

        self.knowledge_graph[entity].update(properties)

    def get_entity_info(self, entity: str) -> Optional[Dict[str, Any]]:
        """Get information about an entity"""
        return self.knowledge_graph.get(entity)
```

```python
# src/tomobait/memory/conversation_memory.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    role: str  # 'user' or 'assistant' or agent name
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class ConversationMemory:
    """Manages conversation context and summarization"""

    def __init__(self, max_turns: int = 50):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum conversation turns to keep in memory
        """
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.summary: Optional[str] = None

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn"""
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.turns.append(turn)

        # Prune old turns if exceeding max
        if len(self.turns) > self.max_turns:
            # Keep recent turns and update summary
            old_turns = self.turns[:-self.max_turns]
            self.turns = self.turns[-self.max_turns:]

            # TODO: Generate summary of old turns using LLM

    def get_recent_context(self, n: int = 10) -> str:
        """Get recent conversation context as formatted string"""
        recent = self.turns[-n:]

        formatted = []
        for turn in recent:
            formatted.append(f"{turn.role}: {turn.content}")

        return "\n".join(formatted)

    def get_full_context(self) -> str:
        """Get full conversation context including summary"""
        parts = []

        if self.summary:
            parts.append(f"Previous conversation summary:\n{self.summary}\n")

        parts.append(self.get_recent_context(len(self.turns)))

        return "\n".join(parts)
```

### 5.2 Context Window Management

**New File**: `src/tomobait/memory/context_manager.py`

**Implementation**:

```python
# src/tomobait/memory/context_manager.py
from typing import List, Dict, Any, Optional
import tiktoken
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context window for LLM calls"""

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_tokens: int = 8000,
        reserve_tokens: int = 2000
    ):
        """
        Initialize context manager.

        Args:
            model_name: Model name for tokenization
            max_tokens: Maximum context tokens
            reserve_tokens: Tokens to reserve for response
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to cl100k_base encoding
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))

    def fit_context(
        self,
        system_message: str,
        documents: List[str],
        conversation_history: List[Dict[str, str]],
        user_message: str
    ) -> Dict[str, Any]:
        """
        Fit context within token budget.

        Args:
            system_message: System message
            documents: Retrieved documents
            conversation_history: Previous messages
            user_message: Current user message

        Returns:
            Dictionary with fitted context
        """
        # Count tokens for each component
        system_tokens = self.count_tokens(system_message)
        user_tokens = self.count_tokens(user_message)

        # Budget remaining for docs and history
        remaining = self.available_tokens - system_tokens - user_tokens

        # Allocate tokens (70% docs, 30% history as default)
        doc_budget = int(remaining * 0.7)
        history_budget = remaining - doc_budget

        # Fit documents
        fitted_docs = self._fit_documents(documents, doc_budget)

        # Fit conversation history
        fitted_history = self._fit_history(conversation_history, history_budget)

        return {
            "system_message": system_message,
            "documents": fitted_docs,
            "conversation_history": fitted_history,
            "user_message": user_message,
            "total_tokens": system_tokens + user_tokens +
                          self.count_tokens(" ".join(fitted_docs)) +
                          sum(self.count_tokens(m["content"]) for m in fitted_history)
        }

    def _fit_documents(self, documents: List[str], budget: int) -> List[str]:
        """Fit documents within token budget"""
        fitted = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc)

            if used_tokens + doc_tokens <= budget:
                fitted.append(doc)
                used_tokens += doc_tokens
            else:
                # Try to fit truncated version
                remaining = budget - used_tokens
                if remaining > 100:  # Only if meaningful space left
                    truncated = self._truncate_to_budget(doc, remaining)
                    fitted.append(truncated)
                break

        return fitted

    def _fit_history(
        self,
        history: List[Dict[str, str]],
        budget: int
    ) -> List[Dict[str, str]]:
        """Fit conversation history within budget, keeping most recent"""
        fitted = []
        used_tokens = 0

        # Process in reverse to keep most recent
        for message in reversed(history):
            msg_tokens = self.count_tokens(message["content"])

            if used_tokens + msg_tokens <= budget:
                fitted.insert(0, message)
                used_tokens += msg_tokens
            else:
                break

        return fitted

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within token budget"""
        tokens = self.encoder.encode(text)

        if len(tokens) <= budget:
            return text

        # Truncate and add ellipsis
        truncated_tokens = tokens[:budget - 3]  # Reserve for "..."
        truncated_text = self.encoder.decode(truncated_tokens)

        return truncated_text + "..."
```

---

## Phase 6: Monitoring & Evaluation (Week 6)

### 6.1 Performance Tracking

**New Files**:
- `src/tomobait/evaluation/__init__.py`
- `src/tomobait/evaluation/metrics.py`
- `src/tomobait/evaluation/logger.py`

**Implementation**:

```python
# src/tomobait/evaluation/metrics.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import statistics
import logging

logger = logging.getLogger(__name__)

class AgentMetrics(BaseModel):
    """Metrics for a single agent invocation"""
    agent_name: str
    timestamp: datetime
    latency_ms: float
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

class RetrievalMetrics(BaseModel):
    """Metrics for document retrieval"""
    query: str
    timestamp: datetime
    num_results: int
    latency_ms: float
    retrieval_strategy: str
    relevance_scores: Optional[List[float]] = None

class WorkflowMetrics(BaseModel):
    """Metrics for complete workflow"""
    workflow_name: str
    timestamp: datetime
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    agents_used: List[str]
    success: bool
    user_rating: Optional[int] = None  # 1-5 stars

class MetricsCollector:
    """Collects and aggregates performance metrics"""

    def __init__(self):
        self.agent_metrics: List[AgentMetrics] = []
        self.retrieval_metrics: List[RetrievalMetrics] = []
        self.workflow_metrics: List[WorkflowMetrics] = []

    def log_agent_call(
        self,
        agent_name: str,
        latency_ms: float,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Log agent invocation metrics"""
        metric = AgentMetrics(
            agent_name=agent_name,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            success=success,
            error=error
        )
        self.agent_metrics.append(metric)
        logger.info(f"Agent {agent_name}: {latency_ms:.2f}ms, {tokens_used} tokens")

    def log_retrieval(
        self,
        query: str,
        num_results: int,
        latency_ms: float,
        retrieval_strategy: str,
        relevance_scores: Optional[List[float]] = None
    ):
        """Log retrieval metrics"""
        metric = RetrievalMetrics(
            query=query,
            timestamp=datetime.now(),
            num_results=num_results,
            latency_ms=latency_ms,
            retrieval_strategy=retrieval_strategy,
            relevance_scores=relevance_scores
        )
        self.retrieval_metrics.append(metric)
        logger.info(f"Retrieval: {retrieval_strategy}, {num_results} results, {latency_ms:.2f}ms")

    def log_workflow(
        self,
        workflow_name: str,
        total_latency_ms: float,
        total_tokens: int,
        total_cost_usd: float,
        agents_used: List[str],
        success: bool,
        user_rating: Optional[int] = None
    ):
        """Log workflow metrics"""
        metric = WorkflowMetrics(
            workflow_name=workflow_name,
            timestamp=datetime.now(),
            total_latency_ms=total_latency_ms,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            agents_used=agents_used,
            success=success,
            user_rating=user_rating
        )
        self.workflow_metrics.append(metric)
        logger.info(
            f"Workflow {workflow_name}: {total_latency_ms:.2f}ms, "
            f"{total_tokens} tokens, ${total_cost_usd:.4f}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.workflow_metrics:
            return {"error": "No metrics collected"}

        return {
            "total_workflows": len(self.workflow_metrics),
            "avg_latency_ms": statistics.mean(m.total_latency_ms for m in self.workflow_metrics),
            "total_tokens": sum(m.total_tokens for m in self.workflow_metrics),
            "total_cost_usd": sum(m.total_cost_usd for m in self.workflow_metrics),
            "success_rate": sum(1 for m in self.workflow_metrics if m.success) / len(self.workflow_metrics),
            "avg_rating": statistics.mean(
                m.user_rating for m in self.workflow_metrics if m.user_rating
            ) if any(m.user_rating for m in self.workflow_metrics) else None
        }

# Global metrics collector
metrics_collector = MetricsCollector()
```

### 6.2 Evaluation Framework

**New File**: `src/tomobait/evaluation/evaluator.py`

**Implementation**:

```python
# src/tomobait/evaluation/evaluator.py
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluates retrieval quality"""

    @staticmethod
    def precision_at_k(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = 3
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Cutoff for evaluation

        Returns:
            Precision@K score (0-1)
        """
        if not retrieved_docs or k == 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(
            1 for doc in top_k
            if doc.metadata.get('id') in relevant_doc_ids
        )

        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = 3
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Cutoff for evaluation

        Returns:
            Recall@K score (0-1)
        """
        if not relevant_doc_ids:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(
            1 for doc in top_k
            if doc.metadata.get('id') in relevant_doc_ids
        )

        return relevant_in_top_k / len(relevant_doc_ids)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of relevant document IDs

        Returns:
            MRR score (0-1)
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc.metadata.get('id') in relevant_doc_ids:
                return 1.0 / i

        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[Document],
        relevance_scores: List[float],
        k: int = 3
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            retrieved_docs: Retrieved documents
            relevance_scores: Relevance score for each retrieved doc
            k: Cutoff for evaluation

        Returns:
            NDCG@K score (0-1)
        """
        import math

        if not retrieved_docs or not relevance_scores:
            return 0.0

        # DCG@K
        dcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )

        # IDCG@K (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal_scores[:k])
        )

        return dcg / idcg if idcg > 0 else 0.0

class ResponseEvaluator:
    """Evaluates response quality using LLM"""

    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize response evaluator.

        Args:
            llm_config: LLM configuration for evaluation
        """
        self.llm_config = llm_config

    async def evaluate_relevance(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Evaluate answer relevance to question.

        Args:
            question: User's question
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Evaluation results
        """
        # TODO: Use LLM to evaluate answer quality
        # Criteria: relevance, accuracy, completeness, clarity

        return {
            "relevance_score": 0.8,  # 0-1
            "accuracy_score": 0.9,
            "completeness_score": 0.7,
            "clarity_score": 0.85,
            "overall_score": 0.81
        }

    async def evaluate_citations(
        self,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Evaluate citation quality.

        Args:
            answer: Generated answer with citations
            retrieved_docs: Retrieved documents

        Returns:
            Citation evaluation results
        """
        # TODO: Evaluate citation quality
        # Criteria: presence, accuracy, formatting, attribution

        return {
            "has_citations": True,
            "citation_accuracy": 0.9,
            "proper_formatting": True,
            "all_claims_cited": False
        }
```

---

## Phase 7: UI/UX Enhancements (Week 7)

### 7.1 Frontend Multi-Agent Visualization

**Modify**: `src/tomobait/frontend.py`

Add new tab for agent visualization:

```python
# Add to frontend.py

def create_agent_metrics_tab():
    """Create tab for agent performance metrics"""
    with gr.Tab("Agent Metrics"):
        gr.Markdown("## Agent Performance Metrics")

        metrics_display = gr.JSON(label="Current Session Metrics")
        refresh_btn = gr.Button("Refresh Metrics")

        # Agent timeline visualization
        gr.Markdown("### Agent Execution Timeline")
        timeline_plot = gr.Plot(label="Agent Timeline")

        # Cost breakdown
        gr.Markdown("### Cost Breakdown")
        cost_plot = gr.Plot(label="Cost by Agent")

        def refresh_metrics():
            from .evaluation.metrics import metrics_collector
            summary = metrics_collector.get_summary()
            return summary

        refresh_btn.click(
            fn=refresh_metrics,
            outputs=[metrics_display]
        )

        return metrics_display

def create_agent_config_tab():
    """Create tab for configuring agents"""
    with gr.Tab("Agent Configuration"):
        gr.Markdown("## Agent Configuration")

        # Enable/disable agents
        agent_toggles = {}
        for agent_name in ["query_analyzer", "doc_expert", "citation_specialist", "beamline_expert"]:
            with gr.Row():
                toggle = gr.Checkbox(label=f"Enable {agent_name}", value=True)
                agent_toggles[agent_name] = toggle

        # Workflow selection
        workflow_dropdown = gr.Dropdown(
            choices=["standard_qa", "beamline_expert", "iterative_refinement", "auto"],
            value="auto",
            label="Default Workflow"
        )

        save_config_btn = gr.Button("Save Configuration")

        def save_agent_config(**kwargs):
            # TODO: Save agent configuration
            return "Configuration saved!"

        save_config_btn.click(
            fn=save_agent_config,
            inputs=list(agent_toggles.values()) + [workflow_dropdown],
            outputs=[gr.Textbox(label="Status")]
        )
```

### 7.2 Agent Decision Flow Display

Add visual feedback for agent activity:

```python
def chat_with_agent_feedback(message, history):
    """Enhanced chat function with agent activity feedback"""

    # Show which agents are active
    status_updates = []

    try:
        # Step 1: Query Analysis
        status_updates.append("🔍 Query Analyzer: Analyzing your question...")
        yield history, "\n".join(status_updates)

        # TODO: Execute query analyzer

        # Step 2: Retrieval
        status_updates.append("📚 Retrieving relevant documentation...")
        yield history, "\n".join(status_updates)

        # TODO: Execute retrieval

        # Step 3: Answer Generation
        status_updates.append("💡 Generating answer...")
        yield history, "\n".join(status_updates)

        # TODO: Execute answer generation

        # Step 4: Citation
        status_updates.append("📝 Adding citations...")
        yield history, "\n".join(status_updates)

        # TODO: Execute citation specialist

        # Final result
        final_answer = "Answer with citations..."
        history.append((message, final_answer))

        yield history, "✅ Complete!"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        yield history, error_msg
```

---

## Testing & Validation

### Unit Tests

**New File**: `tests/test_agents.py`

```python
import pytest
from src.tomobait.agents.query_analyzer import QueryAnalyzerAgent
from src.tomobait.agents.citation_specialist import CitationSpecialistAgent
from src.tomobait.agents.beamline_expert import BeamlineExpertAgent

def test_query_analyzer_creation():
    """Test query analyzer agent creation"""
    llm_config = {"config_list": [{"model": "gpt-4", "api_key": "test"}]}
    agent = QueryAnalyzerAgent.create(llm_config)
    assert agent.name == "query_analyzer"

def test_citation_specialist_creation():
    """Test citation specialist agent creation"""
    llm_config = {"config_list": [{"model": "gpt-4", "api_key": "test"}]}
    agent = CitationSpecialistAgent.create(llm_config)
    assert agent.name == "citation_specialist"

def test_beamline_expert_creation():
    """Test beamline expert agent creation"""
    llm_config = {"config_list": [{"model": "gpt-4", "api_key": "test"}]}
    agent = BeamlineExpertAgent.create(llm_config)
    assert agent.name == "beamline_expert"
```

**New File**: `tests/test_retrieval.py`

```python
import pytest
from src.tomobait.retrieval.hybrid_search import HybridRetriever
from src.tomobait.retrieval.reranker import DocumentReranker
from src.tomobait.retrieval.query_reformulation import QueryReformulator

def test_query_reformulator():
    """Test query reformulation"""
    reformulator = QueryReformulator()

    # Test acronym expansion
    expanded = reformulator.expand_acronyms("What is 2-BM?")
    assert "tomography beamline" in expanded.lower()

    # Test variant generation
    variants = reformulator.generate_variants("What is APS?")
    assert len(variants) > 1

def test_document_reranker():
    """Test document reranking"""
    # TODO: Add test with mock documents
    pass

def test_hybrid_retriever():
    """Test hybrid search"""
    # TODO: Add test with mock vector store
    pass
```

### Integration Tests

**New File**: `tests/test_workflows.py`

```python
import pytest
from src.tomobait.orchestration.orchestrator import AgentOrchestrator
from src.tomobait.orchestration.workflows import standard_qa_workflow

@pytest.mark.asyncio
async def test_standard_qa_workflow():
    """Test standard Q&A workflow"""
    # TODO: Test complete workflow execution
    pass

@pytest.mark.asyncio
async def test_workflow_routing():
    """Test workflow routing logic"""
    # TODO: Test router selects correct workflow
    pass
```

---

## Documentation

### User Documentation

**New File**: `docs/multi_agent_system.md`

```markdown
# Multi-Agent System Guide

## Overview

TomoBait uses a sophisticated multi-agent system to answer your questions about tomography beamline documentation.

## Agents

### Query Analyzer
- **Role**: Analyzes your question to determine the best way to answer it
- **When it's used**: Every question
- **What it does**: Classifies question type, extracts key entities, recommends retrieval strategy

### Documentation Expert
- **Role**: Main agent for answering questions based on documentation
- **When it's used**: Most questions
- **What it does**: Synthesizes information from retrieved documents into clear answers

### Citation Specialist
- **Role**: Adds precise citations and references
- **When it's used**: When you need specific sources
- **What it does**: Extracts citations, formats references, creates bibliographies

### Beamline Expert
- **Role**: Specialized knowledge about APS beamlines
- **When it's used**: Beamline-specific questions
- **What it does**: Provides expert advice on beamline selection, capabilities, and usage

## Workflows

### Standard Q&A
Best for: General questions

Flow: Query Analyzer → Retrieval → Documentation Expert → Citation Specialist

### Beamline Expert
Best for: Beamline-specific questions

Flow: Query Analyzer → Beamline Expert + Documentation Expert → Citation Specialist

### Iterative Refinement
Best for: Complex questions requiring multiple searches

Flow: Query Analyzer ↔ Retrieval (loop) → Documentation Expert → Citation Specialist

## Configuration

You can customize the multi-agent system in `config.yaml`:

```yaml
agents:
  default_workflow: auto  # or specify: standard_qa, beamline_expert, etc.
  agents:
    - name: query_analyzer
      enabled: true
    - name: doc_expert
      enabled: true
    - name: citation_specialist
      enabled: false  # Disable if you don't need citations
```

## Monitoring

View agent performance in the "Agent Metrics" tab:
- Response latency per agent
- Token usage and costs
- Success rates
- Agent activity timeline
```

### Developer Documentation

**New File**: `docs/agent_development.md`

```markdown
# Agent Development Guide

## Creating a New Agent

### 1. Define Agent Factory

Create a new file in `src/tomobait/agents/`:

```python
# src/tomobait/agents/my_agent.py
from typing import Dict, Any
from autogen import AssistantAgent

SYSTEM_MESSAGE = """
You are a specialized agent for [specific task].
Your role is to...
"""

class MyAgent:
    @staticmethod
    def create(llm_config: Dict[str, Any]) -> AssistantAgent:
        return AssistantAgent(
            name="my_agent",
            system_message=SYSTEM_MESSAGE,
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
        )
```

### 2. Register in Configuration

Add to `config.yaml`:

```yaml
agents:
  agents:
    - name: my_agent
      type: assistant
      enabled: true
      description: "Brief description"
      system_message: |
        System message here...
```

### 3. Create Workflow

Add workflow in `src/tomobait/orchestration/workflows.py`:

```python
async def my_workflow(
    user_message: str,
    agent_registry,
    tool_registry,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    # Implement workflow logic
    pass
```

### 4. Register Workflow

In `app.py`:

```python
from .orchestration.workflows import my_workflow

orchestrator.register_workflow("my_workflow", my_workflow)
```

### 5. Add Tests

Create tests in `tests/test_my_agent.py`:

```python
def test_my_agent():
    # Test agent creation and behavior
    pass
```

## Creating Custom Tools

### 1. Define Tool Function

In `src/tomobait/tools/`:

```python
def my_tool(param1: str, param2: int) -> str:
    """Tool description"""
    # Implementation
    return result
```

### 2. Register Tool

```python
from .base import tool_registry, ToolMetadata

tool_registry.register(
    ToolMetadata(
        name="my_tool",
        description="What this tool does",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."}
            },
            "required": ["param1"]
        },
        category="custom"
    ),
    my_tool
)
```

### 3. Use in Agent

The tool is now available to all agents via the tool registry.

## Best Practices

1. **Agent Specialization**: Keep agents focused on specific tasks
2. **Clear System Messages**: Provide detailed instructions and examples
3. **Error Handling**: Always handle errors gracefully
4. **Logging**: Use logging for debugging and monitoring
5. **Testing**: Write comprehensive unit and integration tests
6. **Documentation**: Document agent behavior and use cases
7. **Performance**: Monitor token usage and latency
```

---

## Migration Guide

### From Current System to Multi-Agent

**Step 1**: Backup current configuration
```bash
cp config.yaml config.yaml.backup
```

**Step 2**: Install new dependencies
```bash
uv sync
```

**Step 3**: Update configuration
Add agents section to `config.yaml` (see Phase 1.1)

**Step 4**: Test with simple agent
Start with just Query Analyzer enabled

**Step 5**: Gradually enable agents
Enable one agent at a time and test

**Step 6**: Monitor performance
Use metrics dashboard to track improvements

**Rollback**: If issues occur, restore `config.yaml.backup` and restart

---

## Expected Outcomes

### Quantitative Improvements

1. **Retrieval Quality**:
   - Current: ~60% precision@3 (estimated)
   - Target: ~85% precision@3 with hybrid search + reranking
   - Improvement: +40% relative improvement

2. **Response Latency**:
   - Current: ~2-5 seconds
   - Target: ~3-7 seconds (acceptable trade-off for quality)
   - Note: Parallel execution can reduce this

3. **Citation Accuracy**:
   - Current: Generic source references
   - Target: Precise section-level citations with URLs
   - Improvement: 100% (new capability)

4. **Cost Efficiency**:
   - Monitor cost per query
   - Optimize by using cheaper models for simple agents
   - Target: <$0.05 per query with GPT-4

### Qualitative Improvements

1. **Answer Quality**:
   - More comprehensive answers from multiple sources
   - Better handling of complex questions
   - Reduced hallucination through verification

2. **User Experience**:
   - Transparent agent activity
   - Clear source attribution
   - Adaptive workflow based on question type

3. **Maintainability**:
   - Modular agent design
   - Easy to add new agents
   - Clear separation of concerns

4. **Flexibility**:
   - Configurable workflows
   - Enable/disable agents as needed
   - A/B testing different configurations

---

## Risk Mitigation

### Potential Risks

1. **Increased Complexity**: Multiple agents harder to debug
   - Mitigation: Comprehensive logging and metrics

2. **Higher Latency**: More agent calls = slower responses
   - Mitigation: Parallel execution, caching, faster models for simple tasks

3. **Higher Costs**: More API calls = higher costs
   - Mitigation: Usage monitoring, budget alerts, model optimization

4. **Configuration Complexity**: More settings to manage
   - Mitigation: Good defaults, clear documentation, UI for configuration

5. **Breaking Changes**: Migration issues
   - Mitigation: Backward compatibility, gradual rollout, rollback plan

### Rollback Plan

1. Restore `config.yaml.backup`
2. Revert code to previous commit
3. Restart services
4. Verify system works with 2-agent setup

---

## Success Criteria

### Phase 1 (Week 1)
- ✅ Agent registry implemented
- ✅ Tool registry functional
- ✅ Basic orchestration working
- ✅ Config schema updated

### Phase 2 (Week 2)
- ✅ Query Analyzer agent operational
- ✅ Citation Specialist agent working
- ✅ Beamline Expert agent deployed
- ✅ All agents integrated in app.py

### Phase 3 (Week 3)
- ✅ Hybrid search functional
- ✅ Reranking improves results
- ✅ Query reformulation working
- ✅ Retrieval quality improved by 30%+

### Phase 4 (Week 4)
- ✅ Workflow router implemented
- ✅ 3+ workflows operational
- ✅ Multi-agent collaboration working
- ✅ Parallel execution functional

### Phase 5 (Week 5)
- ✅ Agent memory system working
- ✅ Conversation memory functional
- ✅ Context manager prevents overflow
- ✅ Multi-turn conversations improved

### Phase 6 (Week 6)
- ✅ Metrics collection working
- ✅ Performance dashboard functional
- ✅ Evaluation framework operational
- ✅ Cost tracking implemented

### Phase 7 (Week 7)
- ✅ UI shows agent activity
- ✅ Agent configuration UI working
- ✅ Metrics visualization functional
- ✅ User documentation complete

---

## Future Enhancements (Beyond 7 Weeks)

### MCP Integration
- GitHub MCP server for code search
- Filesystem MCP for local docs
- Web search MCP for external resources
- Custom MCP servers for beamline databases

### Advanced Features
- Code generation agent for runnable examples
- Troubleshooting agent with diagnostic workflows
- Comparison agent for side-by-side analysis
- Summary agent for multi-document synthesis

### AI Improvements
- Fine-tuned models for beamline domain
- Custom embeddings for better retrieval
- Agent learning from user feedback
- Automated prompt optimization

### Scale & Performance
- Agent result caching
- Distributed agent execution
- Load balancing across LLM providers
- Cost optimization strategies

---

## Appendix

### Dependencies Added

```toml
# pyproject.toml additions
dependencies = [
    # ... existing ...
    "rank-bm25>=0.2.2",           # BM25 keyword search
    "sentence-transformers>=2.2.0", # Cross-encoder reranking
    "tiktoken>=0.5.0",            # Token counting
]
```

### File Structure

```
src/tomobait/
├── agents/
│   ├── __init__.py
│   ├── base.py                 # Base agent classes
│   ├── registry.py             # Agent registry
│   ├── query_analyzer.py       # Query Analyzer agent
│   ├── citation_specialist.py  # Citation Specialist agent
│   └── beamline_expert.py      # Beamline Expert agent
├── tools/
│   ├── __init__.py
│   ├── base.py                 # Tool registry
│   ├── search.py               # Search tools
│   └── document.py             # Document tools
├── orchestration/
│   ├── __init__.py
│   ├── orchestrator.py         # Agent orchestrator
│   ├── workflows.py            # Workflow definitions
│   └── router.py               # Workflow router
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_search.py        # Hybrid retrieval
│   ├── reranker.py             # Document reranking
│   └── query_reformulation.py  # Query reformulation
├── memory/
│   ├── __init__.py
│   ├── agent_memory.py         # Agent memory system
│   ├── conversation_memory.py  # Conversation memory
│   └── context_manager.py      # Context window management
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Performance metrics
│   ├── logger.py               # Metrics logging
│   └── evaluator.py            # Evaluation framework
└── app.py                      # Modified main application

tests/
├── test_agents.py
├── test_tools.py
├── test_orchestration.py
├── test_retrieval.py
├── test_memory.py
└── test_workflows.py

docs/
├── multi_agent_system.md       # User guide
└── agent_development.md        # Developer guide
```

### Configuration Reference

**Complete `config.yaml` with all agent settings**:

See Phase 1.1 for full configuration example.

---

## Contact & Support

For questions about this enhancement plan:
- Review the developer documentation in `docs/agent_development.md`
- Check existing issues in the repository
- Consult the Autogen (AG2) documentation: https://ag2.ai/

---

**End of Enhancement Plan**
