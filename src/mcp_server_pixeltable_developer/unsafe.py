"""Explicitly enabled local-development tools with bounded execution."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, Field

from .canvas import CanvasController, CanvasError, transform_local_media

UNSAFE_ENV_VAR = "PIXELTABLE_MCP_ENABLE_UNSAFE"
_MAX_CODE_CHARS = 100_000
_MAX_REQUIREMENT_CHARS = 512
_MAX_CAPTURE_BYTES = 256 * 1024
_MAX_EXECUTION_SECONDS = 120
_DEFAULT_EXECUTION_SECONDS = 30
_INSTALL_TIMEOUT_SECONDS = 120
_REDACTED_URL_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]{0,31}://)(?P<credentials>[^/@\s]{1,512})@",
    flags=re.IGNORECASE,
)
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:API[_-]?KEY|AUTH|CREDENTIAL|PASSWORD|SECRET|TOKEN)",
    flags=re.IGNORECASE,
)

CanvasContentType = Literal[
    "audio",
    "chart",
    "comparison",
    "image",
    "image_grid",
    "json",
    "mermaid",
    "table",
    "text",
    "video",
]


class UnsafeServerConfig(Protocol):
    """Configuration fields consumed by :func:`register_unsafe_tools`."""

    @property
    def project_root(self) -> Path: ...

    @property
    def pixeltable_home(self) -> Path: ...

    @property
    def unsafe_enabled(self) -> bool: ...

    @property
    def transport(self) -> str: ...


class ExecutionResult(BaseModel):
    """Structured result from a one-shot Python process."""

    exit_code: int
    stdout: str
    stderr: str
    output_truncated: bool
    elapsed_ms: int


class InstallResult(BaseModel):
    """Structured result from an explicit package installation."""

    requirement: str
    installer: Literal["uv", "pip"]
    exit_code: int
    stdout: str
    stderr: str
    output_truncated: bool
    elapsed_ms: int


class DisplayResult(BaseModel):
    """Location of the authenticated localhost canvas."""

    canvas_url: str
    content_type: CanvasContentType


@dataclass
class _CapturedStream:
    limit: int
    value: bytearray = field(default_factory=bytearray)
    truncated: bool = False

    async def read(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while chunk := await stream.read(64 * 1024):
            remaining = self.limit - len(self.value)
            if remaining > 0:
                self.value.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.truncated = True

    def text(self) -> str:
        return self.value.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class _ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    output_truncated: bool
    elapsed_ms: int


class UnsafeRuntime:
    """State shared by the three opt-in unsafe tools."""

    def __init__(self, config: UnsafeServerConfig) -> None:
        self.project_root = Path(config.project_root).expanduser().resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError("The configured project root must be a directory.")
        self.pixeltable_home = Path(config.pixeltable_home).expanduser().resolve(strict=False)
        self.canvas_port = int(getattr(config, "canvas_port", 7777))

        configured_roots = getattr(config, "media_roots", None)
        roots = list(configured_roots) if configured_roots is not None else []
        roots.extend((self.project_root, self.pixeltable_home))
        self.media_roots = tuple(Path(root) for root in roots)
        self._canvas: CanvasController | None = None
        self._canvas_lock = threading.Lock()

    async def pixeltable_unsafe_execute_python(
        self,
        code: Annotated[
            str,
            Field(
                min_length=1,
                max_length=_MAX_CODE_CHARS,
                description="Python source to execute once in the configured project directory.",
            ),
        ],
        timeout_seconds: Annotated[
            int,
            Field(
                ge=1,
                le=_MAX_EXECUTION_SECONDS,
                description="Maximum process runtime before it and its process group are stopped.",
            ),
        ] = _DEFAULT_EXECUTION_SECONDS,
    ) -> ExecutionResult:
        """Execute Python once in a bounded subprocess with no persistent session."""
        result = await _run_process(
            [sys.executable, "-c", code],
            cwd=self.project_root,
            env=self._subprocess_environment(),
            timeout_seconds=timeout_seconds,
        )
        if result.exit_code != 0:
            raise ToolError(_process_failure("Python execution", result))
        return ExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            output_truncated=result.output_truncated,
            elapsed_ms=result.elapsed_ms,
        )

    async def pixeltable_unsafe_install_package(
        self,
        requirement: Annotated[
            str,
            Field(
                min_length=1,
                max_length=_MAX_REQUIREMENT_CHARS,
                description="One PEP 508 package requirement from a package index.",
            ),
        ],
    ) -> InstallResult:
        """Install one validated index requirement into the server environment."""
        normalized = _validate_requirement(requirement)
        uv_executable = shutil.which("uv")
        if uv_executable is not None:
            installer: Literal["uv", "pip"] = "uv"
            command = [
                uv_executable,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--",
                normalized,
            ]
        else:
            installer = "pip"
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--",
                normalized,
            ]

        result = await _run_process(
            command,
            cwd=self.project_root,
            env=self._subprocess_environment(),
            timeout_seconds=_INSTALL_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            raise ToolError(_process_failure(f"Installation of {normalized!r}", result))
        return InstallResult(
            requirement=normalized,
            installer=installer,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            output_truncated=result.output_truncated,
            elapsed_ms=result.elapsed_ms,
        )

    async def pixeltable_unsafe_display(
        self,
        content_type: Annotated[
            CanvasContentType,
            Field(description="Canvas renderer to use for the supplied JSON data."),
        ],
        data: Annotated[Any, Field(description="JSON-serializable content to display.")],
        title: Annotated[
            str | None,
            Field(max_length=200, description="Optional heading shown above the content."),
        ] = None,
    ) -> DisplayResult:
        """Queue content for an authenticated, localhost-only browser canvas."""
        try:
            canvas = self._get_canvas()
            processed_data = transform_local_media(data, canvas)
            message: dict[str, Any] = {
                "content_type": content_type,
                "data": processed_data,
            }
            if title is not None:
                message["title"] = title
            canvas.broadcast(message)
        except CanvasError as exc:
            raise ToolError(str(exc)) from exc
        return DisplayResult(canvas_url=canvas.url, content_type=content_type)

    def close(self) -> None:
        """Release optional unsafe-mode resources."""
        with self._canvas_lock:
            if self._canvas is not None:
                self._canvas.close()
                self._canvas = None

    def _get_canvas(self) -> CanvasController:
        with self._canvas_lock:
            if self._canvas is None:
                self._canvas = CanvasController(self.media_roots, port=self.canvas_port)
            return self._canvas

    def _subprocess_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PIXELTABLE_HOME"] = str(self.pixeltable_home)
        environment["PIXELTABLE_DISABLE_STDOUT"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        return environment


def register_unsafe_tools(server: Any, config: UnsafeServerConfig) -> UnsafeRuntime | None:
    """Register unsafe tools only for an explicitly enabled stdio server."""
    if not config.unsafe_enabled or os.environ.get(UNSAFE_ENV_VAR) != "1":
        return None
    if config.transport.lower() != "stdio":
        raise ValueError("Unsafe tools can only be enabled for the stdio transport.")

    runtime = UnsafeRuntime(config)
    annotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=True,
    )
    display_annotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    )
    server.tool(
        name="pixeltable_unsafe_execute_python",
        annotations=annotations,
        structured_output=True,
    )(runtime.pixeltable_unsafe_execute_python)
    server.tool(
        name="pixeltable_unsafe_install_package",
        annotations=annotations,
        structured_output=True,
    )(runtime.pixeltable_unsafe_install_package)
    server.tool(
        name="pixeltable_unsafe_display",
        annotations=display_annotations,
        structured_output=True,
    )(runtime.pixeltable_unsafe_display)
    return runtime


def _validate_requirement(value: str) -> str:
    if value != value.strip():
        raise ToolError("Package requirement cannot start or end with whitespace.")
    try:
        requirement = Requirement(value)
    except InvalidRequirement as exc:
        raise ToolError(f"Invalid PEP 508 package requirement: {exc}") from exc
    if requirement.url is not None:
        raise ToolError("Direct URL and local-path requirements are not allowed.")
    return str(requirement)


async def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> _ProcessResult:
    started = time.perf_counter()
    process_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        process_kwargs["start_new_session"] = True
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_kwargs,
        )
    except OSError as exc:
        raise ToolError(f"Could not start subprocess: {exc}") from exc

    stdout = _CapturedStream(_MAX_CAPTURE_BYTES)
    stderr = _CapturedStream(_MAX_CAPTURE_BYTES)
    readers = [
        asyncio.create_task(stdout.read(process.stdout)),
        asyncio.create_task(stderr.read(process.stderr)),
    ]
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError as exc:
        await _stop_process(process)
        await _finish_readers(readers)
        raise ToolError(f"Subprocess exceeded the {timeout_seconds}-second timeout.") from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        await _finish_readers(readers)
        raise
    await _finish_readers(readers)

    return _ProcessResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=_redact_secrets(stdout.text(), env),
        stderr=_redact_secrets(stderr.text(), env),
        output_truncated=stdout.truncated or stderr.truncated,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


async def _finish_readers(readers: list[asyncio.Task[None]]) -> None:
    try:
        await asyncio.wait_for(asyncio.gather(*readers), timeout=2)
    except TimeoutError:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


def _process_failure(action: str, result: _ProcessResult) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or "No subprocess output."
    detail = _redact_credentials(detail[-2_000:])
    truncation = " Output was truncated." if result.output_truncated else ""
    return f"{action} failed with exit code {result.exit_code}: {detail}{truncation}"


def _redact_credentials(value: str) -> str:
    return _REDACTED_URL_CREDENTIALS.sub(r"\g<scheme>***@", value)


def _redact_secrets(value: str, environment: dict[str, str]) -> str:
    redacted = _redact_credentials(value)
    sensitive_values = {
        secret for name, secret in environment.items() if _SENSITIVE_ENV_NAME.search(name) and len(secret) >= 6
    }
    for secret in sorted(sensitive_values, key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted
