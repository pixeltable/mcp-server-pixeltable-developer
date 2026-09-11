"""Real subprocess tests for the production stdio transport."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from mcp_server_pixeltable_developer.runtime import ServerConfig


def _stdio_environment(config: ServerConfig) -> dict[str, str]:
    return {
        **os.environ,
        "PIXELTABLE_MCP_PROJECT_ROOT": str(config.project_root),
        "PIXELTABLE_HOME": str(config.pixeltable_home),
        "PIXELTABLE_DISABLE_STDOUT": "1",
    }


@pytest.mark.asyncio
async def test_real_stdio_client_supports_concurrent_calls_and_clean_eof(server_config: ServerConfig) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server_pixeltable_developer"],
        env=_stdio_environment(server_config),
        cwd=server_config.project_root,
    )
    async with asyncio.timeout(10):
        async with Client(parameters, raise_exceptions=True) as client:
            tools, resources, prompt = await asyncio.gather(
                client.list_tools(),
                client.list_resources(),
                client.get_prompt("pixeltable_build_app", {"goal": "make a local API"}),
            )
            assert len(tools.tools) == 18
            assert len(resources.resources) == 4
            assert prompt.messages


@pytest.mark.asyncio
async def test_raw_stdio_stdout_contains_only_protocol_frames_and_eof_exits(
    server_config: ServerConfig,
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "mcp_server_pixeltable_developer",
        cwd=server_config.project_root,
        env=_stdio_environment(server_config),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "stdio-test", "version": "1"},
        },
    }
    process.stdin.write((json.dumps(initialize) + "\n").encode())
    await process.stdin.drain()
    first_line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
    first = json.loads(first_line)
    assert first["id"] == 1
    # MCP 2.2 uses 2026-07-28 for stateless per-request HTTP. The newest
    # initialize-handshake revision used by stdio remains 2025-11-25.
    assert first["result"]["protocolVersion"] == "2025-11-25"
    assert first["result"]["serverInfo"]["name"] == "pixeltable-developer"

    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    list_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    process.stdin.write((json.dumps(initialized) + "\n" + json.dumps(list_tools) + "\n").encode())
    await process.stdin.drain()
    second_line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
    second = json.loads(second_line)
    assert second["id"] == 2
    assert len(second["result"]["tools"]) == 18

    process.stdin.close()
    await process.stdin.wait_closed()
    assert await asyncio.wait_for(process.wait(), timeout=5) == 0
    assert await process.stdout.read() == b""


@pytest.mark.asyncio
async def test_startup_configuration_error_exits_nonzero_without_protocol_output(tmp_path: Path) -> None:
    missing_project = tmp_path / "missing"
    environment = {
        **os.environ,
        "PIXELTABLE_MCP_PROJECT_ROOT": str(missing_project),
        "PIXELTABLE_HOME": str(tmp_path / "catalog"),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "mcp_server_pixeltable_developer",
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
    assert process.returncode not in (None, 0)
    assert stdout == b""
    assert b"Project root is not a directory" in stderr


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness assertion")
async def test_real_stdio_cancellation_terminates_pxt_child(
    server_config: ServerConfig,
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pxt-child.pid"
    fake_pxt = tmp_path / "pxt"
    fake_pxt.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['PIXELTABLE_MCP_TEST_PID_FILE']).write_text(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    fake_pxt.chmod(0o755)
    environment = {
        **_stdio_environment(server_config),
        "PIXELTABLE_MCP_PXT": str(fake_pxt),
        "PIXELTABLE_MCP_TEST_PID_FILE": str(pid_file),
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server_pixeltable_developer"],
        env=environment,
        cwd=server_config.project_root,
    )
    async with Client(parameters, raise_exceptions=True) as client:
        call = asyncio.create_task(client.call_tool("pixeltable_list_catalog"))
        for _ in range(200):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()
        pid = int(pid_file.read_text())
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        for _ in range(200):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail(f"cancelled pxt child process {pid} is still alive")
