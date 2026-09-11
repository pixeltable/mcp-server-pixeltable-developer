"""MCP 2.2 server construction for Pixeltable developer workflows."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer

from . import __version__
from .prompts import register_prompts
from .resources import register_resources
from .runtime import CommandRunner, ServerConfig
from .tools import register_default_tools

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """Pixeltable developer server for source-controlled multimodal applications.
Read pixeltable://guidance/app before scaffolding or changing an application. Prefer read-only describe,
check, and diff calls before updates. Schema and service updates may affect hosted targets; prune tools
default to dry runs. Keep tables, computed columns, and HTTP routes in the application file. Unsafe local
Python, package-install, and display tools are absent unless PIXELTABLE_MCP_ENABLE_UNSAFE=1 over stdio."""


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Dependencies fixed for one MCP server lifetime."""

    config: ServerConfig
    runner: CommandRunner


def create_server(config: ServerConfig | None = None) -> MCPServer[RuntimeState]:
    """Build a fully registered MCP server without starting a transport."""

    config = config or ServerConfig.from_env(transport="stdio")
    if config.unsafe_enabled and os.environ.get("PIXELTABLE_MCP_ENABLE_UNSAFE") != "1":
        config = replace(config, unsafe_enabled=False)
    config.apply_environment()
    runner = CommandRunner(config)
    state = RuntimeState(config=config, runner=runner)
    unsafe_runtime: list[Any] = []

    @asynccontextmanager
    async def lifespan(_: MCPServer[RuntimeState]) -> AsyncIterator[RuntimeState]:
        config.apply_environment()
        try:
            yield state
        finally:
            for runtime in reversed(unsafe_runtime):
                close = getattr(runtime, "close", None)
                if close is None:
                    continue
                result = close()
                if inspect.isawaitable(result):
                    await result

    server: MCPServer[RuntimeState] = MCPServer(
        "pixeltable-developer",
        title="Pixeltable Developer",
        description="Local developer tools for Pixeltable application and service workflows.",
        instructions=SERVER_INSTRUCTIONS,
        website_url="https://github.com/pixeltable/mcp-server-pixeltable-developer",
        version=__version__,
        lifespan=lifespan,
        cache_hints={
            "server/discover": CacheHint(ttl_ms=300_000, scope="private"),
            "tools/list": CacheHint(ttl_ms=300_000, scope="private"),
            "resources/list": CacheHint(ttl_ms=300_000, scope="private"),
            "prompts/list": CacheHint(ttl_ms=300_000, scope="private"),
            "resources/read": CacheHint(ttl_ms=0, scope="private"),
        },
    )
    register_default_tools(server, config, runner)
    register_resources(server, config, runner)
    register_prompts(server)

    if config.unsafe_enabled:
        from .unsafe import register_unsafe_tools

        unsafe_runtime.append(register_unsafe_tools(server, config))

    return server


def create_conformance_app(config: ServerConfig | None = None) -> Any:
    """Create a stateless Streamable HTTP app for local conformance tests only."""

    if config is None:
        env_config = ServerConfig.from_env(transport="streamable-http")
        config = ServerConfig(
            project_root=env_config.project_root,
            pixeltable_home=env_config.pixeltable_home,
            transport="streamable-http",
            unsafe_enabled=False,
            pxt_executable=env_config.pxt_executable,
            command_timeout_seconds=env_config.command_timeout_seconds,
            max_output_bytes=env_config.max_output_bytes,
        )
    elif config.transport != "streamable-http":
        raise ValueError("Conformance app requires transport='streamable-http'")
    server = create_server(config)
    from .conformance import register_conformance_diagnostics

    register_conformance_diagnostics(server)
    return server.streamable_http_app(stateless_http=True, json_response=True, host="127.0.0.1")


mcp = create_server()


def main() -> None:
    """Run the production stdio server and propagate startup failures."""

    logger.info("Starting Pixeltable Developer MCP %s", __version__)
    mcp.run("stdio")


__all__ = ["RuntimeState", "create_conformance_app", "create_server", "main", "mcp"]
