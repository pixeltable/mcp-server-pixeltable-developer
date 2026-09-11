"""Startup configuration, containment, subprocess, and redaction tests."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server_pixeltable_developer.runtime import CommandRunner, ServerConfig, find_pxt_executable, redact_secrets


def test_environment_is_fixed_and_unsafe_requires_exact_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    catalog = tmp_path / "catalog"
    project.mkdir()
    catalog.mkdir()
    monkeypatch.setenv("PIXELTABLE_MCP_PROJECT_ROOT", str(project))
    monkeypatch.setenv("PIXELTABLE_HOME", str(catalog))
    monkeypatch.setenv("PIXELTABLE_MCP_ENABLE_UNSAFE", "true")
    assert ServerConfig.from_env().unsafe_enabled is False
    monkeypatch.setenv("PIXELTABLE_MCP_ENABLE_UNSAFE", "1")
    config = ServerConfig.from_env()
    assert config.project_root == project.resolve()
    assert config.pixeltable_home == catalog.resolve()
    assert config.unsafe_enabled is True

    config.apply_environment()
    monkeypatch.setenv("PIXELTABLE_HOME", str(tmp_path / "changed"))
    assert config.command_environment()["PIXELTABLE_HOME"] == str(catalog.resolve())


def test_pxt_discovery_prefers_running_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PIXELTABLE_MCP_PXT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_pxt_executable() == str(Path(sys.executable).with_name("pxt"))

    override = tmp_path / "custom-pxt"
    monkeypatch.setenv("PIXELTABLE_MCP_PXT", str(override))
    assert find_pxt_executable() == str(override)


def test_http_transport_rejects_unsafe_mode(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="only be enabled over stdio"):
        ServerConfig(
            project_root=project,
            pixeltable_home=tmp_path / "catalog",
            transport="streamable-http",
            unsafe_enabled=True,
        )


def test_project_path_and_symlink_containment(server_config: ServerConfig, tmp_path: Path) -> None:
    inside = server_config.project_root / "app.py"
    inside.write_text("# app\n")
    assert server_config.resolve_project_file("app.py") == inside
    assert server_config.resolve_project_file(str(inside)) == inside

    with pytest.raises(ToolError, match="inside the configured project root"):
        server_config.resolve_project_file("../escape.py", must_exist=False)

    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n")
    link = server_config.project_root / "linked.py"
    link.symlink_to(outside)
    with pytest.raises(ToolError, match="inside the configured project root"):
        server_config.resolve_project_file("linked.py")

    with pytest.raises(ToolError, match="relative"):
        server_config.resolve_scaffold_output(str(server_config.project_root / "new.py"))
    with pytest.raises(ToolError, match="overwrite"):
        server_config.resolve_scaffold_output("app.py")


@pytest.mark.parametrize("value", ["../x", "bad-name", "x//y", "x/2bad", "pxt://bad"])
def test_catalog_path_validation_rejects_ambiguous_values(value: str) -> None:
    with pytest.raises(ToolError):
        ServerConfig.validate_target(value)


def test_catalog_and_hosted_target_validation() -> None:
    assert ServerConfig.validate_catalog_path("app/docs") == "app/docs"
    assert ServerConfig.validate_target("pxt://acme:main/docs") == "pxt://acme:main/docs"
    assert ServerConfig.validate_target("app/docs/") == "app/docs"


def test_redact_secrets() -> None:
    raw = "api_key=abc123 token:xyz789 https://user:password@example.com sk-abcdefghijk"
    redacted = redact_secrets(raw)
    assert "abc123" not in redacted
    assert "xyz789" not in redacted
    assert "password" not in redacted
    assert "sk-abcdefghijk" not in redacted


@pytest.mark.asyncio
async def test_runner_parses_json_and_accepts_pending_exit(
    server_config: ServerConfig,
    tmp_path: Path,
) -> None:
    script = tmp_path / "pending.py"
    script.write_text("import json; print(json.dumps({'pending': True})); raise SystemExit(2)\n")
    result = await CommandRunner(server_config).run(
        [sys.executable, str(script)],
        allowed_exit_codes=frozenset({0, 2}),
        parse_json=True,
    )
    assert result.exit_code == 2
    assert result.pending is True
    assert result.data == {"pending": True}


@pytest.mark.asyncio
async def test_runner_turns_cli_failure_into_redacted_tool_error(
    server_config: ServerConfig,
    tmp_path: Path,
) -> None:
    script = tmp_path / "failure.py"
    script.write_text("import sys; print('token=super-secret-value', file=sys.stderr); raise SystemExit(3)\n")
    with pytest.raises(ToolError) as error:
        await CommandRunner(server_config).run(
            [sys.executable, str(script)],
            allowed_exit_codes=frozenset({0}),
            parse_json=False,
        )
    assert "super-secret-value" not in str(error.value)
    assert "token=***" in str(error.value)


@pytest.mark.asyncio
async def test_runner_rejects_invalid_or_oversized_output(
    server_config: ServerConfig,
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.py"
    invalid.write_text("print('not-json')\n")
    with pytest.raises(ToolError, match="invalid JSON"):
        await CommandRunner(server_config).run(
            [sys.executable, str(invalid)],
            allowed_exit_codes=frozenset({0}),
            parse_json=True,
        )

    large = tmp_path / "large.py"
    large.write_text("print('x' * 200)\n")
    small_config = replace(server_config, max_output_bytes=100)
    with pytest.raises(ToolError, match="output exceeded"):
        await CommandRunner(small_config).run(
            [sys.executable, str(large)],
            allowed_exit_codes=frozenset({0}),
            parse_json=False,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness assertion")
async def test_cancellation_terminates_child_command(
    server_config: ServerConfig,
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pid"
    script = tmp_path / "wait.py"
    script.write_text(
        f"import os, pathlib, time\npathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(120)\n"
    )
    task = asyncio.create_task(
        CommandRunner(server_config).run(
            [sys.executable, str(script)],
            allowed_exit_codes=frozenset({0}),
            parse_json=False,
        )
    )
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(f"cancelled child process {pid} is still alive")
