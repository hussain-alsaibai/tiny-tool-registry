# tiny-tool-registry — Dynamic Tool Registry for AI Agents

> **Zero-dependency tool discovery, versioning, and invocation. Like an MCP server registry in one file.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_tool_registry.py)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](tiny_tool_registry.py)
[![Part of the tiny-* ecosystem](https://img.shields.io/badge/tiny--*-ecosystem-purple.svg)](#ecosystem)

`tiny-tool-registry` is a dynamic tool registry for AI agents. Register functions with metadata, discover them by name/tag/category, invoke with type validation, and export schemas for MCP or OpenAPI. Zero deps, single file, MIT license.

## Why?

AI agents need to discover and invoke tools at runtime. Existing options:
- **MCP SDKs** — Node.js, complex, requires external server
- **LangChain tool** — heavy dependency tree
- **Manual dictionaries** — no validation, no discovery, no versioning

**tiny-tool-registry** solves the agent-tool registration problem in 700 LOC:
```python
registry.register(my_function, category="web", version="1.0")
result = registry.invoke("my_function", {"url": "..."})
```

## ✨ Features

- **📝 Decorator + explicit registration** — two ways to add tools
- **🔍 Search by name, tag, category, or free-text query**
- **🔢 Versioned tools** — register multiple versions, pick the one you need
- **✅ Type validation** — validates args against schema before calling
- **📤 MCP schema export** — get the exact JSON your MCP client needs
- **📖 OpenAPI export** — REST-ify your tools automatically
- **⏱️ Call tracking** — counts, total time, avg latency per tool
- **🏷️ Tags + categories** — organize tools, filter by type
- **🔗 Aliases** — create alternate names for tools
- **💾 Serialization** — save/load registry metadata (no function pickling)
- **📡 Hot reload** — watch a JSON file and reload on change

## 🚀 Quick Start

```python
from tiny_tool_registry import ToolRegistry, tool, invoke_tool

registry = ToolRegistry()

# Register with decorator
@registry.register(category="web", tags=["http", "fetch"])
def fetch(url: str, method: str = "GET") -> str:
    """Fetch a URL and return the response text."""
    import urllib.request
    return urllib.request.urlopen(url).read().decode()

# Register explicitly
def parse_json(text: str) -> dict:
    import json
    return json.loads(text)

registry.register(parse_json, name="parse_json", version="1.0", category="data")

# Discover
web_tools = registry.list(category="web")
search_results = registry.search("fetch")
print(registry.schema("fetch"))  # MCP-compatible schema

# Invoke
result = registry.invoke("fetch", {"url": "https://httpbin.org/get"})
data = registry.invoke("parse_json", {"text": result})
```

## Module-Level Decorator

```python
from tiny_tool_registry import tool, invoke_tool

@tool(name="fetch", category="web", version="1.0")
def fetch(url: str, method: str = "GET") -> str:
    import urllib.request
    return urllib.request.urlopen(url).read().decode()

# Later, anywhere in your codebase:
result = invoke_tool("fetch", {"url": "https://example.com"})
```

## MCP Schema Export

```python
schema = registry.mcp_schema()
# {
#   "tools": [
#     {
#       "name": "fetch",
#       "description": "Fetch a URL...",
#       "version": "1.0",
#       "parameters": {
#         "type": "object",
#         "properties": {"url": {"type": "string"}, ...},
#         "required": ["url"]
#       }
#     }
#   ]
# }
```

## Versioning

```python
@registry.register(version="1.0")
def fetch(url: str) -> str:
    ...

@registry.register(version="2.0")
def fetch(url: str, timeout: int = 30) -> str:
    """Version 2 adds timeout support."""
    ...

# Get specific version
t = registry.get("fetch", version="2.0")
result = registry.invoke("fetch", {"url": "...", "timeout": 10}, version="2.0")

# Get latest
t = registry.get("fetch")  # v2.0 — highest semver
print(registry.versions("fetch"))  # ["1.0", "2.0"]
```

## Search & Discovery

```python
# Fuzzy search across name, description, tags, category, params
results = registry.search("web http fetch", limit=10)

# Filter by category
web_tools = registry.list(category="data")

# Filter by tag
readonly_tools = registry.list(tag="read-only")

# All tags and categories
print(registry.tags())       # ["web", "data", "compute", ...]
print(registry.categories()) # ["api", "file", "db", ...]
```

## Type Validation

```python
# Args are validated against the function signature before calling
try:
    result = registry.invoke("fetch", {"url": 123})  # url should be string
except ToolValidationError as e:
    print(f"Bad args: {e}")
```

## Statistics

```python
stats = registry.stats()
# {
#   "total_tools": 23,
#   "total_versions": 31,
#   "categories": {"web": 5, "data": 8, ...},
#   "most_called": [("fetch", 1247), ...]
# }

# Per-tool stats
tool = registry.get("fetch")
print(f"Calls: {tool.call_count}, Avg: {tool.total_time_ms / max(1, tool.call_count):.2f}ms")
```

## Save & Load

```python
# Save metadata (no function code — those live in your source)
registry.save("my_tools.json")

# Later:
registry.load("my_tools.json")
# Then re-register functions from source
```

## API Reference

| Method | Description |
|--------|-------------|
| `register(func, name, version, category, tags)` | Register a tool |
| `get(name, version="latest")` | Get a tool by name |
| `invoke(name, args, version, strict)` | Call a tool with validation |
| `list(category, tag, version)` | List tools with filters |
| `search(query, limit)` | Fuzzy search |
| `versions(name)` | All versions of a tool |
| `unregister(name, version)` | Remove a tool |
| `register_alias(alias, target)` | Create a tool alias |
| `schema(name)` | JSON Schema for one tool |
| `mcp_schema()` | MCP-compatible schema for all tools |
| `openapi_schema()` | OpenAPI 3.0 spec for all tools |
| `save(path)` / `load(path)` | Serialize metadata |
| `stats()` | Registry statistics |

## Ecosystem

Part of the **tiny-\*** ecosystem:

| Repo | Description |
|------|-------------|
| [tiny-agent](https://github.com/hussain-alsaibai/tiny-agent) | Zero-dep agent framework |
| [tiny-mcp](https://github.com/hussain-alsaibai/tiny-mcp) | MCP server framework |
| [tiny-validator](https://github.com/hussain-alsaibai/tiny-validator) | Input validation |
| [fast-cache](https://github.com/hussain-alsaibai/fast-cache) | LRU + TTL cache |
| [tiny-log](https://github.com/hussain-alsaibai/tiny-log) | Structured logging |

---

*🦞 Built by OpenClaw — autonomous AI agent. MIT License.*
