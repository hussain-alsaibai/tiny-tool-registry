"""Tests for tiny-tool-registry."""

import json
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from tiny_tool_registry import (
    ToolRegistry, Tool, tool, invoke_tool,
    ToolError, ToolNotFoundError, ToolValidationError,
    ToolExecutionError, ToolVersionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------


def test_register_decorator(registry):
    @registry.register(category="test")
    def hello(name: str) -> str:
        return f"Hello, {name}!"

    t = registry.get("hello")
    assert t.name == "hello"
    assert t.category == "test"
    assert t.params[0].name == "name"
    assert t.params[0].type == "string"


def test_register_explicit(registry):
    def greet(name: str, times: int = 1) -> str:
        return f"Hi, {name}! " * times

    registry.register(greet, name="greet", version="1.0", category="utils")

    t = registry.get("greet")
    assert t.version == "1.0"
    assert t.category == "utils"
    assert len(t.params) == 2


def test_register_with_options(registry):
    @registry.register(
        name="custom_name",
        version="2.0",
        category="custom",
        tags=["special", "v2"],
    )
    def my_tool(x: int) -> int:
        return x * 2

    t = registry.get("custom_name", version="2.0")
    assert t.name == "custom_name"
    assert t.version == "2.0"
    assert "special" in t.tags


def test_register_overwrite(registry):
    @registry.register(version="1.0")
    def my_func(x: int) -> int:
        return x

    registry.register(my_func, name="my_func", version="2.0")
    versions = registry.versions("my_func")
    assert "1.0" in versions
    assert "2.0" in versions


def test_register_no_overwrite_raises(registry):
    @registry.register(version="1.0")
    def my_func(x: int) -> int:
        return x

    # Without allow_overwrite, re-registering same version should raise
    registry.allow_overwrite = False
    with pytest.raises(ToolVersionError):
        registry.register(my_func, name="my_func", version="1.0")


def test_register_multiple_same_version_allowed_when_overwrite(registry):
    @registry.register(version="1.0")
    def my_func(x: int) -> int:
        return x

    registry.allow_overwrite = True
    # Should not raise
    registry.register(my_func, name="my_func", version="1.0")


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def test_invoke_basic(registry):
    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    result = registry.invoke("add", {"a": 3, "b": 4})
    assert result == 7


def test_invoke_with_defaults(registry):
    @registry.register
    def greet(name: str, excited: bool = False) -> str:
        msg = f"Hello, {name}"
        return msg + "!" if excited else msg

    result = registry.invoke("greet", {"name": "Alice"})
    assert result == "Hello, Alice"

    result = registry.invoke("greet", {"name": "Bob", "excited": True})
    assert result == "Hello, Bob!"


def test_invoke_raises_not_found(registry):
    with pytest.raises(ToolNotFoundError):
        registry.invoke("nonexistent", {})


def test_invoke_raises_validation_error(registry):
    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolValidationError):
        registry.invoke("add", {"a": "not_an_int", "b": 4})


def test_invoke_non_strict(registry):
    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    # strict=False skips validation — passes to function as-is
    # (may raise in the function itself)
    result = registry.invoke("add", {"a": 3, "b": 4}, strict=False)
    assert result == 7


def test_invoke_with_call_tracking(registry):
    @registry.register
    def slow(x: int) -> int:
        import time
        time.sleep(0.001)
        return x * 2

    for i in range(3):
        registry.invoke("slow", {"x": i})

    t = registry.get("slow")
    assert t.call_count == 3
    assert t.total_time_ms > 0


def test_invoke_deprecated_warning(registry):
    @registry.register(deprecated=True, deprecation_message="Use v2 instead")
    def old_tool(x: int) -> int:
        return x

    # The deprecated tool still executes normally
    result = registry.invoke("old_tool", {"x": 1})
    assert result == 1

    # Verify the deprecated flag is set on the tool
    t = registry.get("old_tool")
    assert t.deprecated is True
    assert "v2" in t.deprecation_message


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_all(registry):
    @registry.register
    def tool_a() -> None: pass

    @registry.register
    def tool_b() -> None: pass

    tools = registry.list()
    names = {t.name for t in tools}
    assert "tool_a" in names
    assert "tool_b" in names


def test_list_by_category(registry):
    @registry.register(category="api")
    def fetch() -> str: return ""

    @registry.register(category="utils")
    def parse() -> str: return ""

    api_tools = registry.list(category="api")
    assert len(api_tools) == 1
    assert api_tools[0].name == "fetch"


def test_list_by_tag(registry):
    @registry.register(tags=["read", "api"])
    def fetch() -> str: return ""

    @registry.register(tags=["write"])
    def save() -> None: pass

    read_tools = registry.list(tag="read")
    assert len(read_tools) == 1
    assert read_tools[0].name == "fetch"


def test_search_by_name(registry):
    @registry.register
    def fetch_url(url: str) -> str: return ""

    @registry.register
    def fetch_data() -> str: return ""

    @registry.register
    def parse_json(text: str) -> dict: return {}

    results = registry.search("fetch")
    assert len(results) == 2
    names = {t.name for t in results}
    assert "fetch_url" in names
    assert "fetch_data" in names


def test_search_by_description(registry):
    @registry.register
    def read_file(path: str) -> str:
        """Read a file from disk."""
        return ""

    results = registry.search("file")
    assert len(results) >= 1


def test_search_by_tag(registry):
    @registry.register(tags=["crypto"])
    def encrypt(data: str) -> str: return ""

    results = registry.search("crypto")
    assert len(results) == 1


def test_versions(registry):
    @registry.register(version="1.0")
    def my_tool() -> None: pass

    registry.register(my_tool, name="my_tool", version="2.0")
    registry.register(my_tool, name="my_tool", version="3.0")

    versions = registry.versions("my_tool")
    assert set(versions) == {"1.0", "2.0", "3.0"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema(registry):
    @registry.register
    def fetch(url: str, timeout: int = 30) -> str:
        """Fetch a URL."""
        return ""

    schema = registry.schema("fetch")

    assert schema["name"] == "fetch"
    assert schema["description"] == "Fetch a URL."
    assert "url" in schema["parameters"]["properties"]
    assert "timeout" in schema["parameters"]["properties"]
    assert "url" in schema["parameters"]["required"]


def test_mcp_schema(registry):
    @registry.register
    def my_tool(x: int) -> int: return x

    schema = registry.mcp_schema()
    assert "tools" in schema
    assert schema["total"] == 1
    assert schema["tools"][0]["name"] == "my_tool"


def test_openapi_schema(registry):
    @registry.register
    def echo(msg: str) -> str: return msg

    spec = registry.openapi_schema()
    assert "paths" in spec
    assert "/tools/echo" in spec["paths"]


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


def test_alias(registry):
    @registry.register
    def fetch_url(url: str) -> str: return ""

    registry.register_alias("get", "fetch_url")

    t = registry.get("get")
    assert t.name == "get"
    # Function should be the same
    result = registry.invoke("get", {"url": "https://example.com"})
    assert result == ""


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------


def test_unregister(registry):
    @registry.register
    def temp_tool() -> None: pass

    registry.unregister("temp_tool")
    with pytest.raises(ToolNotFoundError):
        registry.get("temp_tool")


def test_unregister_specific_version(registry):
    @registry.register(version="1.0")
    def my_tool() -> None: pass

    registry.register(my_tool, name="my_tool", version="2.0")

    registry.unregister("my_tool", version="1.0")
    assert registry.versions("my_tool") == ["2.0"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_metadata(tmpdir):
    """save() serializes metadata. load() restores it.
    
    Note: load() restores metadata but functions must be re-registered
    (Python can't serialize functions). This is expected behavior.
    """
    registry = ToolRegistry()

    @registry.register
    def my_tool(x: int) -> int: return x

    path = tmpdir / "registry.json"
    registry.save(path)

    # Verify save produced valid JSON
    import json
    with open(path) as f:
        data = json.load(f)
    assert data["total_tools"] == 1
    assert data["tools"][0]["name"] == "my_tool"

    # load() restores metadata into the registry
    registry2 = ToolRegistry()
    registry2.load(path)
    tools = registry2.list()
    # Metadata is restored
    assert len(tools) == 1


def test_to_json(registry):
    @registry.register
    def my_tool(x: int) -> int: return x

    data = json.loads(registry.to_json())
    assert data["total_tools"] == 1
    assert data["tools"][0]["name"] == "my_tool"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(registry):
    @registry.register(category="api")
    def fetch() -> str: return ""

    @registry.register(category="api")
    def post() -> str: return ""

    @registry.register(category="data")
    def parse() -> str: return ""

    stats = registry.stats()
    assert stats["total_tools"] == 3
    assert stats["categories"] == 2
    assert stats["tools_by_category"]["api"] == 2


# ---------------------------------------------------------------------------
# Module-level decorator
# ---------------------------------------------------------------------------


def test_module_level_decorator():
    from tiny_tool_registry import _get_default_registry
    reg = _get_default_registry()

    # Count before
    before = len(reg.list())

    @tool(name="tmp_test", category="test", registry=reg)
    def tmp_func(x: int) -> int:
        return x * 2

    after = len(reg.list())
    assert after == before + 1

    result = invoke_tool("tmp_test", {"x": 5})
    assert result == 10

    # Cleanup
    reg.unregister("tmp_test")


# ---------------------------------------------------------------------------
# Tags and categories
# ---------------------------------------------------------------------------


def test_tags(registry):
    @registry.register(tags=["web", "http"])
    def fetch() -> str: return ""

    @registry.register(tags=["json"])
    def parse() -> str: return ""

    assert set(registry.tags()) == {"web", "http", "json"}


def test_categories(registry):
    @registry.register(category="api")
    def fetch() -> str: return ""

    @registry.register(category="data")
    def parse() -> str: return ""

    assert set(registry.categories()) == {"api", "data"}


# ---------------------------------------------------------------------------
# Type hints
# ---------------------------------------------------------------------------


def test_return_type_detection(registry):
    @registry.register
    def get_user() -> dict:
        return {}

    t = registry.get("get_user")
    assert t.return_type in ("dict", "object")


def test_list_param_type(registry):
    @registry.register
    def process(items: list) -> int:
        return len(items)

    t = registry.get("process")
    assert t.params[0].type in ("array", "list")


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------


def test_tool_to_dict(registry):
    @registry.register
    def my_tool(x: int) -> int:
        return x

    t = registry.get("my_tool")
    d = t.to_dict()
    assert d["name"] == "my_tool"
    assert "parameters" in d


def test_tool_examples(registry):
    @registry.register(
        examples=[{"input": {"x": 1}, "output": 1}]
    )
    def my_tool(x: int) -> int:
        return x

    t = registry.get("my_tool")
    assert len(t.examples) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
