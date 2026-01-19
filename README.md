# GenAI Agent Library - MiniMinds

A modular Python library for building LLM-powered agents with tool execution, web browsing capabilities, and observability. This library provides a flexible framework for creating autonomous agents that can interact with files, execute code, browse the web, and perform complex reasoning tasks.

## Overview

MiniMinds is a lightweight framework that abstracts the complexity of building generative AI agents. It provides:

- **Pluggable LLM Support**: Abstract interface for multiple LLM providers (Groq, OpenAI, Gemini)
- **Tool System**: Decorator-based tool registration with automatic schema generation
- **Agent Framework**: Base classes for implementing stateful, iterative agent workflows
- **Web Automation**: Browser control via Playwright for web-based tasks
- **Observability**: Built-in Langfuse integration for tracing and debugging
- **Context Management**: Session-based resource management for multi-agent scenarios

The library demonstrates practical agent patterns including automated unit test generation, web exploration, and file manipulation.

## Key Features

- **Multi-Provider LLM Support**: Abstracted client interface supporting Groq, with extensibility for OpenAI and Gemini
- **Declarative Tool System**: Define tools using Python decorators; automatic conversion to LLM-compatible schemas
- **Built-in Tool Suites**:
  - File operations (read, write, list, create/remove directories)
  - Code execution (run Python files, execute pytest)
  - Web automation (navigate, click, fill forms, screenshot)
  - Utility tools (JSON validation, string manipulation, math operations)
- **Agent Lifecycle Management**: Base agent class with configurable iteration limits and state tracking
- **Persistent Sessions**: Context managers for managing browser instances and agent state
- **Context Optimization**: Scratchpad pattern for pruning conversation history to reduce token usage
- **Tracing & Observability**: Langfuse decorators for LLM call and tool execution monitoring

## System Architecture

### Agent Execution Flow

```
User Query
    ↓
[Agent State Initialization]
    ↓
┌─────────────────────────────────┐
│  Iterative Reasoning Loop       │
│  ┌─────────────────────────┐   │
│  │ 1. LLM Generate         │   │
│  │    (with tool schemas)  │   │
│  └───────────┬─────────────┘   │
│              ↓                  │
│  ┌─────────────────────────┐   │
│  │ 2. Parse Response       │   │
│  │    (content + tool calls)│  │
│  └───────────┬─────────────┘   │
│              ↓                  │
│  ┌─────────────────────────┐   │
│  │ 3. Execute Tools        │   │
│  │    (via registry)       │   │
│  └───────────┬─────────────┘   │
│              ↓                  │
│  ┌─────────────────────────┐   │
│  │ 4. Update State         │   │
│  │    (messages + status)  │   │
│  └───────────┬─────────────┘   │
│              ↓                  │
│      [Check Stop Condition]    │
│              ↓                  │
└──────────────┬─────────────────┘
               ↓
          Final Result
```

### Tool Registration & Execution

1. **Tool Definition**: Functions decorated with `@tool()` are converted to `Tool` objects
2. **Schema Generation**: Tools are serialized to OpenAI/Gemini-compatible function schemas
3. **Registry Management**: `ToolRegistry` aggregates tools and manages session injection
4. **LLM Integration**: Tool schemas passed to LLM; tool calls parsed from responses
5. **Execution**: Tools invoked via registry with automatic session ID injection

### Context Management Strategies

The library implements two agent patterns for handling conversation context:

- **Simple Agent** (`v1_simple.py`): Retains full message history; suitable for short tasks
- **Scratchpad Agent** (`v2_scratchpad.py`): Prunes tool outputs after each iteration, maintaining only system/user messages and latest assistant state

## Technologies & Models Used

### Core Dependencies

- **Python**: 3.11+
- **Pydantic**: Data validation and settings management
- **Loguru**: Structured logging

### LLM Integration

- **Groq**: Primary LLM provider (llama-3.3-70b-versatile, llama-3.1-70b-versatile)
- **OpenAI**: Supported via abstract interface
- **Gemini**: Supported via abstract interface

### Tools & Automation

- **Playwright**: Headless browser automation for web interaction
- **Pytest**: Test execution and validation

### Observability

- **Langfuse**: Distributed tracing for LLM calls and tool executions

### Package Management

- **UV**: Fast Python package installer and environment manager

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- UV package manager (recommended)

### Step 1: Install UV

```bash
pip install uv
```

### Step 2: Clone Repository

```bash
git clone https://github.com/kariem-magdy/GenAI-Agent-Lab-Library-MiniMinds.git
cd GenAI-Agent-Lab-Library-MiniMinds
```

### Step 3: Install Dependencies

```bash
uv sync
```

### Step 4: Activate Virtual Environment

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### Step 5: Configure API Keys

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here

# Optional: for observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

Get your Groq API key from [console.groq.com](https://console.groq.com)

### Step 6: Install Playwright Browsers (for web tools)

```bash
playwright install chromium
```

## Usage Instructions

### Basic Tool Usage

```python
from tools.toolkit.builtin.math_tools import add, multiply

# Tools are callable
result = add(5, 3)  # Returns 8

# Get tool description for LLM
print(add.to_string())
# Output: Tool Name: add, Description: Adding Numbers, Arguments: a: int|float, b: int|float, Outputs: int|float
```

### Creating Custom Tools

```python
from tools.decorator import tool

@tool()
def fetch_weather(city: str) -> dict:
    """Fetch weather information for a given city."""
    # Implementation here
    return {"city": city, "temp": 22}

# Tool is now registered and has schema generation
print(fetch_weather.to_openai_format())
```

### Building a Simple Agent

```python
from llm.groq_client import GroqClient, LLMConfig
from tools.registry import ToolRegistry
from tools.toolkit.builtin import math_tools, string_tools

# Configure LLM
config = LLMConfig(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=2048
)
client = GroqClient(config)

# Register tools
registry = ToolRegistry()
registry.register_from_module(math_tools)
registry.register_from_module(string_tools)

# Create messages
messages = [
    {"role": "system", "content": f"You are a helpful assistant. Available tools:\n{registry.to_string()}"},
    {"role": "user", "content": "Calculate 15 * 7 and convert the result to uppercase string"}
]

# Generate with tools
response = client.generate(messages, tools=registry.to_client_tools(config.provider))
print(response)
```

### Using the Unit Tester Agent

The library includes a practical example agent that generates and executes unit tests:

```python
from agent.unit_tester.v2_scratchpad import ScratchpadUnitTesterAgent
from llm.groq_client import GroqClient, LLMConfig

config = LLMConfig(
    model_name="llama-3.3-70b-versatile",
    temperature=1.0,
    max_tokens=5000
)
client = GroqClient(config)

agent = ScratchpadUnitTesterAgent(client, max_iterations=20)

user_query = """
Write unit tests for tools/toolkit/web_explorer.py and run them.
Output test results in tools/llm_tests/ directory.
"""

state = agent.iterate(user_query=user_query)
print(state.messages[-1])  # Final report
```

Run the example:

```bash
python -m agent.examples.03_use_v2_agent
```

### Web Automation with Browser Tools

```python
from session import Session
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_tools

with Session("web-session") as session:
    registry = ToolRegistry(session.session_id)
    registry.register_from_module(web_tools)
    
    # Navigate to a URL
    status = registry.get("goto_url")("https://example.com")
    print(status)
    
    # Extract page content
    content = registry.get("get_page_content")(mode="text")
    print(content)
    
    # Take screenshot
    screenshot_data = registry.get("screenshot")(full_page=True)
    
    # Clean up
    registry.get("end_browsing_page")()
```

## Example Workflow

### Automated Unit Test Generation

**Scenario**: Generate pytest tests for a Python module, execute them, and report results.

**Agent Workflow**:

1. **File Discovery**: Agent lists project files using `list_directory_files` tool
2. **Code Reading**: Reads target module using `read_file` tool
3. **Test Generation**: Uses LLM to generate pytest test cases
4. **Test Writing**: Writes tests to file using `write_file` tool
5. **Execution**: Runs tests using `run_pytest_tests` tool
6. **Analysis**: Parses test results (pass/fail counts)
7. **Iteration**: If failures detected, reads error output and regenerates tests
8. **Reporting**: Returns structured JSON report with test summary

**Example Output**:

```json
{
  "finished": true,
  "message": "10 tests passed, 2 failed in 1.86s",
  "scratchpad": "Generated tests for web_explorer.py. All core functions covered. Two edge case failures require mock adjustment."
}
```

**Run Example**:

```bash
python -m agent.examples.00_raw_unit_tester
```

With tracing:

```bash
python -m agent.examples.01_raw_traced_unit_tester
```

## Project Structure

```
GenAI-Agent-Lab-Library-MiniMinds/
├── agent/                          # Agent framework
│   ├── base.py                     # Base agent class with iteration logic
│   ├── examples/                   # Example agent implementations
│   │   ├── 00_raw_unit_tester.py  # Raw agent without framework
│   │   ├── 01_raw_traced_unit_tester.py  # With Langfuse tracing
│   │   ├── 02_use_v1_agent.py     # Using simple agent class
│   │   └── 03_use_v2_agent.py     # Using scratchpad agent
│   └── unit_tester/                # Unit tester agent implementations
│       ├── v1_simple.py            # Full history retention
│       └── v2_scratchpad.py        # Context pruning strategy
├── llm/                            # LLM client abstractions
│   ├── base.py                     # Abstract LLM client interface
│   ├── config.py                   # Configuration with Pydantic
│   └── groq_client.py              # Groq implementation
├── tools/                          # Tool system
│   ├── base.py                     # Tool class definition
│   ├── decorator.py                # @tool() decorator
│   ├── registry.py                 # Tool registry and management
│   ├── main.py                     # Tool testing script
│   └── toolkit/                    # Tool collections
│       ├── web_explorer.py         # Browser automation tools
│       └── builtin/                # Built-in utility tools
│           ├── code_tools.py       # Python/pytest execution
│           ├── file_tools.py       # File system operations
│           ├── json_tools.py       # JSON validation
│           ├── math_tools.py       # Mathematical operations
│           └── string_tools.py     # String manipulation
├── prompts/                        # System prompts
│   ├── unit_tester_v1.txt          # Prompt for simple agent
│   └── unit_tester_v2.txt          # Prompt for scratchpad agent
├── browser_manager.py              # Playwright browser lifecycle
├── session.py                      # Session context manager
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## Limitations & Future Improvements

### Current Limitations

- **LLM Provider Support**: Only Groq is fully implemented; OpenAI/Gemini require client implementation
- **Error Recovery**: Limited retry logic for tool execution failures
- **Parallel Execution**: Tools execute sequentially; no concurrent tool calls
- **Memory Management**: Context pruning is manual; no automatic summarization
- **Tool Validation**: No runtime validation of tool outputs against schemas
- **Browser Isolation**: Single browser instance per session; no headful mode option

### Planned Improvements

- Implement OpenAI and Gemini client adapters
- Add automatic context summarization using LLM
- Support streaming responses for real-time agent output
- Implement tool output validation layer
- Add multi-agent coordination primitives
- Develop memory module for long-term agent state persistence
- Create tool marketplace for community-contributed tools
- Add structured logging for tool execution timeline
- Implement automatic test generation for custom tools

## Contributing Guidelines

Contributions are welcome. Please follow these guidelines:

1. **Code Style**: Follow PEP 8; use Pydantic for configuration models
2. **Type Hints**: All functions must have type annotations
3. **Documentation**: Docstrings required for all public methods
4. **Testing**: Add tests for new tools in `tools/llm_tests/`
5. **Logging**: Use Loguru for structured logging; avoid print statements
6. **Tracing**: Decorate new LLM/tool wrappers with `@observe`

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes with clear messages
4. Add tests and ensure existing tests pass
5. Submit a pull request with description of changes

### Tool Development

When creating new tools:

- Use the `@tool()` decorator
- Provide clear docstrings (used as LLM descriptions)
- Return JSON-serializable types (str, dict, list, int, bool)
- Handle exceptions gracefully; return error dictionaries
- Support `session_id` parameter if the tool requires state

## References

### Learning Resources

- [HuggingFace Agent Course](https://huggingface.co/learn/agents-course/en/unit0/introduction) - Introduction to AI agents
- [Context Engineering by Langchain](https://blog.langchain.com/context-engineering-for-agents/) - Managing agent context
- [Multi-Agent Architectures](https://www.youtube.com/watch?v=4nZl32FwU-o) - Conceptual overview
- [Advanced Context Engineering](https://www.youtube.com/watch?v=IS_y40zY-hc) - Context optimization techniques

### Research & Papers

- [How Long Contexts Fail](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) - Understanding context window limitations

## License

This project is part of a GenAI educational lab. License information not specified.
