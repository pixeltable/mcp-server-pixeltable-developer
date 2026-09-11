"""Diagnostic fixtures used only by the Streamable HTTP conformance entrypoint."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.prompts.base import Prompt
from mcp.shared.exceptions import MCPError
from mcp.types import MISSING_REQUIRED_CLIENT_CAPABILITY


def register_conformance_diagnostics(server: MCPServer[Any]) -> None:
    """Register harness probes without changing the production stdio surface."""

    @server.tool(name="test_missing_capability")
    async def missing_capability(ctx: Context) -> str:
        capabilities = ctx.session.client_capabilities
        if capabilities is None or capabilities.sampling is None:
            raise MCPError(
                code=MISSING_REQUIRED_CLIENT_CAPABILITY,
                message="This diagnostic requires the sampling capability",
                data={"requiredCapabilities": {"sampling": {}}},
            )
        return "sampling capability declared"

    def dynamic_tool() -> str:
        return "dynamic"

    def dynamic_prompt() -> str:
        return "dynamic"

    @server.tool(name="test_trigger_tool_change")
    async def trigger_tool_change(ctx: Context) -> str:
        server.add_tool(dynamic_tool, name="test_dynamic_tool")
        server.remove_tool("test_dynamic_tool")
        await ctx.notify_tools_changed()
        return "tool list changed"

    @server.tool(name="test_trigger_prompt_change")
    async def trigger_prompt_change(ctx: Context) -> str:
        server.add_prompt(Prompt.from_function(dynamic_prompt, name="test_dynamic_prompt", description="dynamic"))
        server.remove_prompt("test_dynamic_prompt")
        await ctx.notify_prompts_changed()
        return "prompt list changed"
