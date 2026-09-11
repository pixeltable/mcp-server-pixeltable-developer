"""MCP 2 discovery, schema, resource, prompt, and error contracts."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from importlib.metadata import version as distribution_version
from typing import Any

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent, TextResourceContents
from packaging.version import Version

from mcp_server_pixeltable_developer.runtime import ServerConfig
from mcp_server_pixeltable_developer.server import create_server

DEFAULT_TOOLS = [
    "pixeltable_list_catalog",
    "pixeltable_describe",
    "pixeltable_rows",
    "pixeltable_get_row",
    "pixeltable_errors",
    "pixeltable_insert_rows",
    "pixeltable_recompute",
    "pixeltable_scaffold_app",
    "pixeltable_schema_check",
    "pixeltable_schema_diff",
    "pixeltable_schema_update",
    "pixeltable_schema_prune",
    "pixeltable_service_check",
    "pixeltable_service_diff",
    "pixeltable_service_update",
    "pixeltable_service_list",
    "pixeltable_service_stop",
    "pixeltable_service_prune",
]

RESOURCE_MIME_TYPES = {
    "pixeltable://status": "application/json",
    "pixeltable://catalog": "application/json",
    "pixeltable://guidance/app": "text/markdown",
    "pixeltable://guidance/cloud": "text/markdown",
}

PROMPT_ARGUMENTS = {
    "pixeltable_build_app": {"goal": "Build a notes API"},
    "pixeltable_build_rag": {
        "source_media": "local PDFs",
        "query_behavior": "return relevant chunks",
    },
    "pixeltable_build_agent": {
        "objective": "answer from a catalog",
        "tools": "search_docs",
    },
    "pixeltable_debug_computation": {"table": "app/docs", "column": "summary"},
}


def _annotation_tuple(tool: Any) -> tuple[bool, bool, bool, bool]:
    annotations = tool.annotations
    assert annotations is not None
    return (
        bool(annotations.read_only_hint),
        bool(annotations.destructive_hint),
        bool(annotations.idempotent_hint),
        bool(annotations.open_world_hint),
    )


@pytest.mark.asyncio
async def test_discovery_has_exact_typed_surface(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        tools_result = await client.list_tools()
        resources_result = await client.list_resources()
        prompts_result = await client.list_prompts()
        server_info = client.server_info

    assert server_info is not None
    assert server_info.name == "pixeltable-developer"
    assert server_info.version == "0.2.0"
    assert [tool.name for tool in tools_result.tools] == DEFAULT_TOOLS
    assert {str(resource.uri): resource.mime_type for resource in resources_result.resources} == RESOURCE_MIME_TYPES
    assert [prompt.name for prompt in prompts_result.prompts] == list(PROMPT_ARGUMENTS)
    assert tools_result.ttl_ms == 300_000
    assert tools_result.cache_scope == "private"
    assert resources_result.ttl_ms == 300_000
    assert prompts_result.ttl_ms == 300_000

    for tool in tools_result.tools:
        assert tool.input_schema.get("type") == "object"
        assert tool.output_schema is not None
        assert "success" not in tool.output_schema.get("properties", {})
        assert tool.annotations is not None

    by_name = {tool.name: tool for tool in tools_result.tools}
    assert by_name["pixeltable_rows"].input_schema["properties"]["limit"]["maximum"] == 100
    assert by_name["pixeltable_insert_rows"].input_schema["properties"]["rows"]["maxItems"] == 1_000
    recompute_properties = by_name["pixeltable_recompute"].input_schema["properties"]
    assert recompute_properties["dry_run"]["default"] is True
    assert recompute_properties["errors_only"]["default"] is True
    assert by_name["pixeltable_schema_prune"].input_schema["properties"]["dry_run"]["default"] is True
    assert by_name["pixeltable_service_prune"].input_schema["properties"]["dry_run"]["default"] is True


@pytest.mark.asyncio
async def test_annotations_match_effects(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    local_reads = {
        "pixeltable_list_catalog",
        "pixeltable_describe",
        "pixeltable_rows",
        "pixeltable_get_row",
        "pixeltable_errors",
        "pixeltable_schema_check",
        "pixeltable_service_check",
    }
    remote_reads = {
        "pixeltable_schema_diff",
        "pixeltable_service_diff",
        "pixeltable_service_list",
    }
    for name in local_reads:
        assert _annotation_tuple(tools[name]) == (True, False, True, False)
    for name in remote_reads:
        assert _annotation_tuple(tools[name]) == (True, False, True, True)
    for name in {"pixeltable_insert_rows", "pixeltable_recompute"}:
        assert _annotation_tuple(tools[name]) == (False, False, False, True)
    assert _annotation_tuple(tools["pixeltable_scaffold_app"]) == (False, False, False, False)
    for name in {
        "pixeltable_schema_update",
        "pixeltable_schema_prune",
        "pixeltable_service_update",
        "pixeltable_service_stop",
        "pixeltable_service_prune",
    }:
        assert _annotation_tuple(tools[name]) == (False, True, True, True)


@pytest.mark.asyncio
async def test_resources_are_typed_redacted_and_current(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        status_result = await client.read_resource("pixeltable://status")
        app_result = await client.read_resource("pixeltable://guidance/app")
        cloud_result = await client.read_resource("pixeltable://guidance/cloud")

    assert status_result.ttl_ms == 0
    status_content = status_result.contents[0]
    assert isinstance(status_content, TextResourceContents)
    assert status_content.mime_type == "application/json"
    status = json.loads(status_content.text)
    installed_mcp = distribution_version("mcp")
    installed_pixeltable = distribution_version("pixeltable")
    assert status == {
        "server_name": "pixeltable-developer",
        "server_version": "0.2.0",
        "mcp_version": installed_mcp,
        "pixeltable_version": installed_pixeltable,
        "project": server_config.project_root.name,
        "transport": "stdio",
        "unsafe_tools_enabled": False,
    }
    # The resource must report the installed versions, and those must stay inside the
    # supported lines pyproject.toml declares. Asserting exact patch versions here made
    # the unlocked CI job fail on every upstream release.
    assert Version("2.2") <= Version(installed_mcp) < Version("3")
    assert Version("0.7.6") <= Version(installed_pixeltable) < Version("0.8")
    assert str(server_config.project_root) not in status_content.text

    app_content = app_result.contents[0]
    cloud_content = cloud_result.contents[0]
    assert isinstance(app_content, TextResourceContents)
    assert isinstance(cloud_content, TextResourceContents)
    app_text = app_content.text
    assert "from `pixeltable.serving`" in app_text
    assert "computed columns use assignments" in app_text
    assert "--allow-destructive` does not make" in app_text
    assert "pxt.Required" in app_text and "Do not use" in app_text
    cloud_text = cloud_content.text
    assert "not live-tested" in cloud_text
    assert "pxt db update" in cloud_text
    assert "pxt schema update" in cloud_text
    assert "pxt service update" in cloud_text


@pytest.mark.asyncio
async def test_all_prompts_render(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        for name, arguments in PROMPT_ARGUMENTS.items():
            prompt = await client.get_prompt(name, arguments)
            assert prompt.messages
            content = prompt.messages[0].content
            assert isinstance(content, TextContent)
            text = content.text
            assert text.strip()
            assert "pixeltable-new" not in text
            assert "pxt.Required" not in text


@pytest.mark.asyncio
async def test_missing_resource_uses_invalid_params_error(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        with pytest.raises(MCPError) as error:
            await client.read_resource("pixeltable://missing")

    assert error.value.code == -32602
    assert "unknown resource" in error.value.message.lower()


@pytest.mark.asyncio
async def test_recoverable_tool_failure_is_an_mcp_error(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=False) as client:
        result = await client.call_tool("pixeltable_rows", {"path": "../outside"})

    assert result.is_error is True
    assert result.structured_content is None
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "Invalid Pixeltable catalog path" in content.text


@pytest.mark.asyncio
async def test_unexpected_tool_failure_is_sanitized(
    server_config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def crash(*args: object, **kwargs: object) -> object:
        raise RuntimeError("secret-client-must-not-see")

    monkeypatch.setattr("mcp_server_pixeltable_developer.runtime.CommandRunner.pxt", crash)
    with caplog.at_level(logging.ERROR, logger="mcp.server.mcpserver.server"):
        async with Client(create_server(server_config), raise_exceptions=False) as client:
            result = await client.call_tool("pixeltable_list_catalog")

    assert result.is_error is True
    assert result.structured_content is None
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "secret-client-must-not-see" not in content.text
    assert content.text == "Error executing tool pixeltable_list_catalog"
    assert any("unexpected exception" in record.getMessage() for record in caplog.records)


def test_contract_constants_have_no_duplicates() -> None:
    assert len(DEFAULT_TOOLS) == len(set(DEFAULT_TOOLS)) == 18
    assert isinstance(RESOURCE_MIME_TYPES, Mapping)
