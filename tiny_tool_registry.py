"""
tiny_tool_registry — zero-dependency dynamic tool registry with versioning,
discovery, and hot-reload for AI agents.

Register, discover, version, and invoke tools at runtime. Like a tiny MCP
server registry in one file.

Usage:
    from tiny_tool_registry import ToolRegistry, tool, Tool

    registry = ToolRegistry()

    @registry.register(version="1.0")
    def fetch_url(url: str) -> str:
        return requests.get(url).text

    # Or register explicitly
    registry.register(fetch_url, name="fetch", version="1.0")

    # Discover
    result = registry.invoke("fetch", {"url": "https://example.com"})
    tools = registry.search("fetch")
    schema = registry.schema("fetch")
"""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, get_type_hints, TYPE_CHECKING


__version__ = "0.1.0"
__all__ = ["ToolRegistry", "Tool", "tool", "ToolError", "ToolNotFoundError"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base exception for tool registry errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found."""
    pass


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation."""
    pass


class ToolVersionError(ToolError):
    """Raised when a version constraint can't be satisfied."""
    pass


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


@dataclass
class ToolParam:
    """A single parameter of a tool."""
    name: str
    type: str
    description: str = ""
    default: Any = None
    required: bool = True
    enum: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.default is None and not self.required:
            d["default"] = None
        return d


@dataclass
class Tool:
    """Metadata and runtime entry point for a registered tool."""
    name: str
    func: Callable[..., Any]
    version: str = "1.0.0"
    description: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    params: list[ToolParam] = field(default_factory=list)
    return_type: str = "any"
    examples: list[dict[str, Any]] = field(default_factory=list)
    author: str = ""
    deprecated: bool = False
    deprecation_message: str = ""
    registered_at: str = ""
    call_count: int = 0
    total_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.registered_at:
            self.registered_at = _utcnow()

    def schema(self) -> dict[str, Any]:
        """Return a JSON Schema compatible description of this tool."""
        properties = {}
        required = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type}
            if p.description:
                prop["description"] = p.description
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None or not p.required:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "returns": {"type": self.return_type},
            "deprecated": self.deprecated,
            "deprecationMessage": self.deprecation_message,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize tool metadata (without the function)."""
        d = self.schema()
        d.update({
            "author": self.author,
            "registeredAt": self.registered_at,
            "callCount": self.call_count,
            "avgTimeMs": round(self.total_time_ms / self.call_count, 2) if self.call_count else 0,
            "examples": self.examples,
        })
        return d


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Dynamic tool registry for AI agents.

    Register tools with metadata, discover by name/tag/category,
    invoke with validation, and serialize to JSON for MCP or other protocols.

    Example:
        registry = ToolRegistry()

        @registry.register(category="web")
        def fetch(url: str, method: str = "GET") -> str:
            return requests.get(url).text

        # Tool discovery
        web_tools = registry.list(category="web")
        search_results = registry.search("fetch")

        # Invoke
        result = registry.invoke("fetch", {"url": "https://example.com"})

        # Export for MCP
        mcp_schema = registry.mcp_schema()
    """

    LATEST = "latest"

    def __init__(self, *, name: str = "default", allow_overwrite: bool = True):
        self.name = name
        self.allow_overwrite = allow_overwrite
        self._tools: dict[str, dict[str, Tool]] = {}  # name -> {version -> Tool}
        self._tags_index: dict[str, set[str]] = {}    # tag -> {tool_name}
        self._category_index: dict[str, set[str]] = {}  # category -> {tool_name}
        self._lock_file: Path | None = None

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        version: str = "1.0.0",
        description: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
        author: str = "",
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a function as a tool.

        Can be used as a decorator or called directly.

        Usage:
            @registry.register()
            def my_tool(arg1: str, arg2: int = 10) -> str:
                ...

            # Or with options
            @registry.register(name="custom_name", version="2.0", category="api")
            def my_tool(arg1: str) -> str:
                ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            meta = self._build_tool(
                fn, tool_name, version, description, category,
                tags or [], examples or [], author, **kwargs
            )

            if tool_name not in self._tools:
                self._tools[tool_name] = {}

            if version in self._tools[tool_name] and not self.allow_overwrite:
                raise ToolVersionError(
                    f"Tool '{tool_name}' v{version} already registered. "
                    "Set allow_overwrite=True to replace."
                )

            self._tools[tool_name][version] = meta
            self._reindex(meta)

            # Preserve callable
            setattr(fn, "_tool_meta", meta)
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def _build_tool(
        self,
        func: Callable[..., Any],
        name: str,
        version: str,
        description: str,
        category: str,
        tags: list[str],
        examples: list[dict[str, Any]],
        author: str,
        deprecated: bool = False,
        deprecation_message: str = "",
        **_: Any,
    ) -> Tool:
        """Inspect a function and build Tool metadata."""
        hints = _safe_get_type_hints(func)
        sig = inspect.signature(func)

        # Description from docstring
        doc = inspect.getdoc(func) or ""
        if not description and doc:
            # First line of docstring
            description = doc.split("\n")[0].strip()

        # Parameters
        params = []
        for pname, param in sig.parameters.items():
            type_hint = hints.get(pname, "any")
            type_str = _type_to_string(type_hint)

            default = param.default
            required = default is inspect.Parameter.empty

            params.append(ToolParam(
                name=pname,
                type=type_str,
                description="",
                default=default if not required else None,
                required=required,
            ))

        # Return type
        return_type = hints.get("return", "any")
        return_type_str = _type_to_string(return_type)

        return Tool(
            name=name,
            func=func,
            version=version,
            description=description,
            category=category,
            tags=tags,
            params=params,
            return_type=return_type_str,
            examples=examples,
            author=author,
            deprecated=deprecated,
            deprecation_message=deprecation_message,
        )

    def _reindex(self, tool: Tool) -> None:
        """Update tag and category indexes."""
        for tag in tool.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(tool.name)

        if tool.category not in self._category_index:
            self._category_index[tool.category] = set()
        self._category_index[tool.category].add(tool.name)

    def unregister(self, name: str, version: str = LATEST) -> None:
        """Unregister a tool."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found.")
        if version == self.LATEST:
            del self._tools[name]
        else:
            del self._tools[name][version]
            if not self._tools[name]:
                del self._tools[name]

    def register_alias(self, alias: str, target: str, version: str = LATEST) -> None:
        """Create an alias (alternate name) for a registered tool."""
        if target not in self._tools:
            raise ToolNotFoundError(f"Tool '{target}' not found.")
        if version == self.LATEST:
            versions = self._tools[target]
            latest_ver = max(versions, key=lambda v: _parse_version(v))
            meta = versions[latest_ver]
        else:
            meta = self._tools[target][version]

        self._tools[alias] = {meta.version: Tool(
            name=alias,
            func=meta.func,
            version=meta.version,
            description=f"Alias for {target}",
            category=meta.category,
            tags=meta.tags + ["alias"],
            params=meta.params,
            return_type=meta.return_type,
        )}

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def get(self, name: str, version: str = LATEST) -> Tool:
        """Get a tool by name and optional version."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found.")
        if version == self.LATEST:
            versions = self._tools[name]
            latest_ver = max(versions, key=lambda v: _parse_version(v))
            return versions[latest_ver]
        if version not in self._tools[name]:
            raise ToolVersionError(f"Tool '{name}' v{version} not found.")
        return self._tools[name][version]

    def list(
        self,
        *,
        category: str | None = None,
        tag: str | None = None,
        version: str | None = None,
        include_deprecated: bool = False,
    ) -> list[Tool]:
        """List tools with optional filters."""
        names: set[str]

        if category:
            names = self._category_index.get(category, set())
        elif tag:
            names = self._tags_index.get(tag, set())
        else:
            names = set(self._tools.keys())

        results = []
        for name in names:
            versions = self._tools[name]
            for ver, tool in versions.items():
                if not include_deprecated and tool.deprecated:
                    continue
                if version and ver != version:
                    continue
                results.append(tool)
        return results

    def search(
        self,
        query: str,
        *,
        fields: list[str] | None = None,
        limit: int = 20,
    ) -> list[Tool]:
        """Fuzzy search tools by name, description, tags, or category."""
        fields = fields or ["name", "description", "tags", "category"]
        query_lower = query.lower()
        results: list[tuple[Tool, int]] = []

        for tool in self.list():
            score = 0

            if "name" in fields:
                if query_lower in tool.name.lower():
                    score += 10
                if query_lower == tool.name.lower():
                    score += 20

            if "description" in fields:
                if query_lower in tool.description.lower():
                    score += 5

            if "tags" in fields:
                for tag in tool.tags:
                    if query_lower in tag.lower():
                        score += 8

            if "category" in fields:
                if query_lower in tool.category.lower():
                    score += 3

            if "params" in fields:
                for p in tool.params:
                    if query_lower in p.name.lower():
                        score += 3

            if score > 0:
                results.append((tool, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in results[:limit]]

    def versions(self, name: str) -> list[str]:
        """Return all registered versions of a tool."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found.")
        return sorted(self._tools[name].keys(), key=_parse_version)

    def tags(self) -> list[str]:
        """Return all known tags."""
        return list(self._tags_index.keys())

    def categories(self) -> list[str]:
        """Return all known categories."""
        return list(self._category_index.keys())

    # -------------------------------------------------------------------------
    # Invocation
    # -------------------------------------------------------------------------

    def invoke(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        version: str = LATEST,
        strict: bool = True,
    ) -> Any:
        """Invoke a tool with the given arguments.

        Args:
            name: Tool name.
            args: Keyword arguments to pass to the tool.
            version: Specific version to invoke.
            strict: If True, validate args against schema before calling.

        Returns:
            The tool's return value.

        Raises:
            ToolNotFoundError: If tool not found.
            ToolValidationError: If args don't match schema.
            ToolExecutionError: If tool raises an exception.
        """
        tool = self.get(name, version)
        args = args or {}

        if tool.deprecated:
            import warnings
            warnings.warn(
                tool.deprecation_message or f"Tool '{name}' is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Validate
        if strict:
            self._validate_args(tool, args)

        # Execute with timing
        import time
        t0 = time.perf_counter()
        try:
            result = tool.func(**args)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tool.call_count += 1
            tool.total_time_ms += elapsed_ms
            return result
        except Exception as e:
            raise ToolExecutionError(
                f"Tool '{name}' failed: {e}"
            ) from e

    def invoke_stream(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a tool that yields/returns a generator."""
        result = self.invoke(name, args, **kwargs)
        if hasattr(result, "__iter__") and hasattr(result, "__next__"):
            return result
        return iter([result])

    def _validate_args(self, tool: Tool, args: dict[str, Any]) -> None:
        """Validate arguments against tool schema."""
        for param in tool.params:
            value = args.get(param.name)
            if param.required and value is None:
                raise ToolValidationError(
                    f"Missing required argument: '{param.name}' for tool '{tool.name}'"
                )
            if value is None:
                continue

            # Type check
            if param.type != "any":
                type_ok = _check_type(value, param.type)
                if not type_ok:
                    raise ToolValidationError(
                        f"Argument '{param.name}' for '{tool.name}': "
                        f"expected {param.type}, got {type(value).__name__}"
                    )

            # Enum check
            if param.enum and value not in param.enum:
                raise ToolValidationError(
                    f"Argument '{param.name}' for '{tool.name}': "
                    f"must be one of {param.enum}, got {value!r}"
                )

    # -------------------------------------------------------------------------
    # Schema / export
    # -------------------------------------------------------------------------

    def schema(self, name: str, version: str = LATEST) -> dict[str, Any]:
        """Return the JSON Schema for a specific tool."""
        return self.get(name, version).schema()

    def mcp_schema(self) -> dict[str, Any]:
        """Return a Model Context Protocol compatible schema for all tools."""
        tools = self.list()
        return {
            "tools": [t.schema() for t in tools],
            "total": len(tools),
            "registry": self.name,
            "generatedAt": _utcnow(),
        }

    def openapi_schema(self) -> dict[str, Any]:
        """Return an OpenAPI-compatible schema for all tools."""
        paths: dict[str, dict[str, Any]] = {}
        for tool in self.list():
            path = f"/tools/{tool.name}"
            paths[path] = {
                "post": {
                    "operationId": tool.name,
                    "summary": tool.description,
                    "tags": [tool.category] + tool.tags,
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": tool.schema()["parameters"],
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": tool.return_type}
                                }
                            }
                        }
                    }
                }
            }

        return {
            "openapi": "3.0.0",
            "info": {
                "title": f"Tool Registry: {self.name}",
                "version": __version__,
            },
            "paths": paths,
        }

    def to_json(self) -> str:
        """Serialize the registry to JSON (tools only, no functions)."""
        tools = [tool.to_dict() for tool in self.list()]
        return json.dumps({
            "name": self.name,
            "version": __version__,
            "tools": tools,
            "total_tools": len(tools),
        }, indent=2, default=str)

    def save(self, path: str | Path) -> None:
        """Save registry metadata (not function code) to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def load(self, path: str | Path) -> None:
        """Load registry metadata from a JSON file (functions must be re-registered)."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.name = data.get("name", self.name)
        # Restore tool metadata (functions can't be serialized — re-register them)
        for tool_data in data.get("tools", []):
            # Reconstruct Tool metadata without the function
            # The caller must re-register actual functions
            meta = tool_data
            name = meta.get("name")
            if not name:
                continue
            version = meta.get("version", "1.0.0")
            if name not in self._tools:
                self._tools[name] = {}
            # Parse parameters from JSON Schema format
            params = []
            for pname, pdata in meta.get("parameters", {}).get("properties", {}).items():
                params.append(ToolParam(
                    name=pname,
                    type=pdata.get("type", "any"),
                    description=pdata.get("description", ""),
                    default=pdata.get("default"),
                    required=pname in meta.get("parameters", {}).get("required", []),
                    enum=pdata.get("enum"),
                ))
            # Create a placeholder tool with the loaded metadata
            # Functions must be re-registered by the caller
            tool = Tool(
                name=name,
                func=lambda **kw: None,  # Placeholder — caller must replace
                version=version,
                description=meta.get("description", ""),
                category=meta.get("category", "general"),
                tags=meta.get("tags", []),
                params=params,
                return_type=meta.get("returns", {}).get("type", "any"),
                examples=meta.get("examples", []),
                author=meta.get("author", ""),
                deprecated=meta.get("deprecated", False),
                deprecation_message=meta.get("deprecationMessage", ""),
                registered_at=meta.get("registeredAt", ""),
                call_count=meta.get("callCount", 0),
                total_time_ms=float(meta.get("avgTimeMs", 0) or 0) * meta.get("callCount", 0),
            )
            self._tools[name][version] = tool
            self._reindex(tool)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return registry statistics."""
        tools = self.list()
        return {
            "total_tools": len(tools),
            "total_versions": sum(len(self._tools[n]) for n in self._tools),
            "categories": len(self._category_index),
            "tags": len(self._tags_index),
            "tools_by_category": {
                cat: len(names) for cat, names in self._category_index.items()
            },
            "most_called": sorted(
                [(t.name, t.call_count) for t in tools],
                key=lambda x: x[1], reverse=True,
            )[:5],
        }

    # -------------------------------------------------------------------------
    # Hot reload (file-based)
    # -------------------------------------------------------------------------

    def watch(self, path: str | Path, poll_interval: float = 1.0) -> None:
        """Watch a file and reload the registry from it on change."""
        import time
        path = Path(path)
        mtime = path.stat().st_mtime

        while True:
            time.sleep(poll_interval)
            try:
                new_mtime = path.stat().st_mtime
                if new_mtime != mtime:
                    mtime = new_mtime
                    self.load(path)
            except Exception:
                continue

    # -------------------------------------------------------------------------
    # Decorator shortcut
    # -------------------------------------------------------------------------

    def tool(
        name: str | None = None,
        version: str = "1.0.0",
        category: str = "general",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Shorthand: use as @registry.tool() instead of @registry.register().

        Usage:
            registry = ToolRegistry()

            @registry.tool("fetch", category="web")
            def fetch(url: str) -> str:
                ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            return registry.register(fn, name=tool_name, version=version,
                                     category=category, tags=tags, **kwargs)
        return decorator


# ---------------------------------------------------------------------------
# Decorator shortcut (module-level)
# ---------------------------------------------------------------------------

# Global default registry for decorator shortcut
_default_registry: ToolRegistry | None = None


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0.0",
    category: str = "general",
    tags: list[str] | None = None,
    registry: ToolRegistry | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Module-level decorator that registers to a default global registry.

    Usage:
        @tool(name="fetch", category="web")
        def fetch(url: str) -> str:
            return requests.get(url).text

        result = invoke_tool("fetch", {"url": "https://example.com"})
    """
    reg = registry or _get_default_registry()

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        reg.register(fn, name=tool_name, version=version,
                     category=category, tags=tags, **kwargs)
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def invoke_tool(name: str, args: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Invoke a tool from the default global registry."""
    return _get_default_registry().invoke(name, args, **kwargs)


def _get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _type_to_string(tp: Any) -> str:
    """Convert a type hint to a string."""
    if tp is None:
        return "null"
    if tp is type(None):
        return "null"
    if tp is bool:
        return "boolean"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is str:
        return "string"
    if tp is list:
        return "array"
    if tp is dict:
        return "object"
    if hasattr(tp, "__origin__"):
        # Generic types like List[str], Dict[str, int]
        origin = getattr(tp, "__origin__", None)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        if origin in (tuple, set):
            return "array"
        name = getattr(tp, "_name", None) or str(tp)
        return name.lower()
    name = getattr(tp, "__name__", None)
    if name:
        return name.lower()
    return str(tp).lower()


def _check_type(value: Any, expected: str) -> bool:
    """Check if a value matches an expected JSON-schema-like type."""
    checks: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "array": lambda v: isinstance(v, (list, tuple, set)),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
        "any": lambda v: True,
    }
    checker = checks.get(expected)
    if checker:
        return checker(value)
    return True  # Unknown types pass


def _safe_get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Get type hints safely, returning {} on failure."""
    try:
        return get_type_hints(func)
    except Exception:
        return {}


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string into a tuple for sorting."""
    try:
        return tuple(int(p) for p in v.split(".")[:3])
    except (ValueError, TypeError):
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Built-in tools (utilities bundled with the registry)
# ---------------------------------------------------------------------------


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in utility tools."""

    @registry.register(category="registry", version="1.0", tags=["builtin"])
    def registry_list_tools(
        category: str = "",
        tag: str = "",
    ) -> list[str]:
        """List all registered tool names."""
        tools = registry.list(category=category or None, tag=tag or None)
        return [t.name for t in tools]

    @registry.register(category="registry", version="1.0", tags=["builtin"])
    def registry_stats() -> dict[str, Any]:
        """Return registry statistics."""
        return registry.stats()

    @registry.register(category="registry", version="1.0", tags=["builtin"])
    def registry_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search tools by query."""
        tools = registry.search(query, limit=limit)
        return [t.to_dict() for t in tools]
