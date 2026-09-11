#!/usr/bin/env python3
"""Print the public MCP inventory through the SDK client."""

from __future__ import annotations

import asyncio

from mcp import Client

from mcp_server_pixeltable_developer.server import mcp


async def list_all() -> None:
    async with Client(mcp, raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools
        resources = (await client.list_resources()).resources
        prompts = (await client.list_prompts()).prompts

    print(f"Tools ({len(tools)})")
    for tool in tools:
        print(f"  {tool.name}: {tool.description or ''}")
    print(f"Resources ({len(resources)})")
    for resource in resources:
        print(f"  {resource.uri}: {resource.description or ''}")
    print(f"Prompts ({len(prompts)})")
    for prompt in prompts:
        print(f"  {prompt.name}: {prompt.description or ''}")


if __name__ == "__main__":
    asyncio.run(list_all())
