"""Opt-in execution, installation, canvas, and unsafe registration tests."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import get_args
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextResourceContents

from mcp_server_pixeltable_developer.canvas import CanvasController, CanvasError, transform_local_media
from mcp_server_pixeltable_developer.runtime import ServerConfig
from mcp_server_pixeltable_developer.server import create_server
from mcp_server_pixeltable_developer.unsafe import (
    CanvasContentType,
    UnsafeRuntime,
    _ProcessResult,
    _validate_requirement,
)


@pytest.mark.asyncio
async def test_unsafe_tools_are_absent_by_default(server_config: ServerConfig) -> None:
    async with Client(create_server(server_config), raise_exceptions=True) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert all(not name.startswith("pixeltable_unsafe_") for name in names)


@pytest.mark.asyncio
async def test_programmatic_flag_cannot_bypass_exact_env_opt_in(
    server_config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIXELTABLE_MCP_ENABLE_UNSAFE", raising=False)
    config = replace(server_config, unsafe_enabled=True)
    async with Client(create_server(config), raise_exceptions=True) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        status = await client.read_resource("pixeltable://status")

    assert all(not name.startswith("pixeltable_unsafe_") for name in names)
    status_content = status.contents[0]
    assert isinstance(status_content, TextResourceContents)
    assert json.loads(status_content.text)["unsafe_tools_enabled"] is False


@pytest.mark.asyncio
async def test_exact_env_opt_in_registers_only_three_unsafe_tools(
    server_config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIXELTABLE_MCP_ENABLE_UNSAFE", "1")
    config = replace(server_config, unsafe_enabled=True)
    async with Client(create_server(config), raise_exceptions=True) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        status = await client.read_resource("pixeltable://status")

    assert names - set(names for names in names if not names.startswith("pixeltable_unsafe_")) == {
        "pixeltable_unsafe_execute_python",
        "pixeltable_unsafe_install_package",
        "pixeltable_unsafe_display",
    }
    status_content = status.contents[0]
    assert isinstance(status_content, TextResourceContents)
    assert json.loads(status_content.text)["unsafe_tools_enabled"] is True


@pytest.mark.asyncio
async def test_one_shot_python_is_bounded_and_redacts_inherited_secrets(
    server_config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "secret-value-123")
    runtime = UnsafeRuntime(server_config)
    result = await runtime.pixeltable_unsafe_execute_python(
        "import os; print(os.getcwd()); print(os.environ['EXAMPLE_API_KEY'])"
    )
    assert str(server_config.project_root) in result.stdout
    assert "secret-value-123" not in result.stdout
    assert "[REDACTED]" in result.stdout

    truncated = await runtime.pixeltable_unsafe_execute_python("print('x' * 300_000)")
    assert truncated.output_truncated is True
    assert len(truncated.stdout.encode()) <= 256 * 1024


@pytest.mark.asyncio
async def test_one_shot_python_timeout_and_failure_are_tool_errors(server_config: ServerConfig) -> None:
    runtime = UnsafeRuntime(server_config)
    with pytest.raises(ToolError, match="1-second timeout"):
        await runtime.pixeltable_unsafe_execute_python("import time; time.sleep(30)", timeout_seconds=1)
    with pytest.raises(ToolError, match="exit code 7"):
        await runtime.pixeltable_unsafe_execute_python("raise SystemExit(7)")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness assertion")
async def test_one_shot_python_cancellation_stops_process(server_config: ServerConfig) -> None:
    pid_file = server_config.project_root / "unsafe.pid"
    code = f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    task = asyncio.create_task(UnsafeRuntime(server_config).pixeltable_unsafe_execute_python(code))
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
        pytest.fail(f"cancelled unsafe process {pid} is still alive")


@pytest.mark.parametrize(
    "requirement",
    [
        "demo @ https://example.com/demo.whl",
        "demo @ file:///tmp/demo.whl",
        "../demo",
        "/tmp/demo",
        " demo>=1",
        "demo>=1 ",
    ],
)
def test_package_requirement_rejects_urls_and_paths(requirement: str) -> None:
    with pytest.raises(ToolError):
        _validate_requirement(requirement)


def test_package_requirement_accepts_pep508_index_requirement() -> None:
    assert _validate_requirement("httpx[http2]>=0.27,<1; python_version >= '3.11'") == (
        'httpx[http2]<1,>=0.27; python_version >= "3.11"'
    )


@pytest.mark.asyncio
async def test_install_uses_an_argument_array(
    server_config: ServerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_process(command: list[str], **kwargs: object) -> _ProcessResult:
        captured["command"] = command
        captured.update(kwargs)
        return _ProcessResult(
            exit_code=0,
            stdout="installed",
            stderr="",
            output_truncated=False,
            elapsed_ms=1,
        )

    monkeypatch.setattr("mcp_server_pixeltable_developer.unsafe._run_process", fake_run_process)
    result = await UnsafeRuntime(server_config).pixeltable_unsafe_install_package("packaging>=24")
    command = captured["command"]
    assert isinstance(command, list)
    assert command[-2:] == ["--", "packaging>=24"]
    assert result.requirement == "packaging>=24"


def test_canvas_authentication_origin_and_media_containment(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    image = media_root / "fixture.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    canvas = CanvasController([media_root], port=0)
    try:
        parsed = urlsplit(canvas.url)
        token = parse_qs(parsed.fragment)["token"][0]
        assert token and token not in canvas.origin

        page = urllib.request.urlopen(f"{canvas.origin}/canvas", timeout=2)
        assert page.status == 200
        assert token.encode() not in page.read()

        media_path = canvas.register_media(image.as_uri())
        request = urllib.request.Request(
            f"{canvas.origin}{media_path}",
            headers={"Authorization": f"Bearer {token}", "Origin": canvas.origin},
        )
        response = urllib.request.urlopen(request, timeout=2)
        assert response.status == 200
        assert response.headers["Access-Control-Allow-Origin"] == canvas.origin
        assert response.read() == image.read_bytes()

        with pytest.raises(urllib.error.HTTPError) as missing_auth:
            urllib.request.urlopen(f"{canvas.origin}{media_path}", timeout=2)
        assert missing_auth.value.code == 401

        evil_request = urllib.request.Request(
            f"{canvas.origin}{media_path}",
            headers={"Authorization": f"Bearer {token}", "Origin": "https://evil.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as bad_origin:
            urllib.request.urlopen(evil_request, timeout=2)
        assert bad_origin.value.code == 403

        with pytest.raises(CanvasError, match="configured media root"):
            canvas.register_media(outside.as_uri())
        link = media_root / "linked.png"
        link.symlink_to(outside)
        with pytest.raises(CanvasError, match="configured media root"):
            canvas.register_media(link.as_uri())
    finally:
        canvas.close()


def test_canvas_transforms_only_contained_file_urls(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    media = root / "audio.wav"
    media.write_bytes(b"RIFFfixture")
    canvas = CanvasController([root], port=0)
    try:
        transformed = transform_local_media({"items": [media.as_uri(), "https://example.com/a.wav"]}, canvas)
        assert transformed["items"][0].startswith("/media/")
        assert transformed["items"][1] == "https://example.com/a.wav"
    finally:
        canvas.close()


def test_canvas_has_no_raw_html_renderer() -> None:
    assert "html" not in get_args(CanvasContentType)
