"""Process configuration, path validation, and subprocess execution."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anyio
from anyio.abc import ByteReceiveStream, ByteSendStream, Process
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import JsonValue

Transport = Literal["stdio", "streamable-http"]

_CATALOG_COMPONENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HOSTED_TARGET = re.compile(r"^pxt://[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_SECRET_ASSIGNMENT = re.compile(r"(?i)(api[_-]?key|password|secret|token)(\s*[=:]\s*)([^\s,;]+|\"[^\"]*\"|'[^']*')")
_URL_PASSWORD = re.compile(r"(://[^:/\s]+:)[^@/\s]+(@)")
_OPENAI_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def find_pxt_executable() -> str:
    """Prefer the Pixeltable CLI installed beside the running server."""

    environment_pxt = os.environ.get("PIXELTABLE_MCP_PXT")
    if environment_pxt:
        return environment_pxt
    sibling_pxt = Path(sys.executable).with_name("pxt")
    if sibling_pxt.is_file():
        return str(sibling_pxt)
    return shutil.which("pxt") or "pxt"


def redact_secrets(text: str) -> str:
    """Remove common secret forms from diagnostic text."""

    text = _SECRET_ASSIGNMENT.sub(r"\1\2***", text)
    text = _URL_PASSWORD.sub(r"\1***\2", text)
    return _OPENAI_TOKEN.sub("sk-***", text)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable configuration fixed when the MCP process starts."""

    project_root: Path
    pixeltable_home: Path
    transport: Transport = "stdio"
    unsafe_enabled: bool = False
    pxt_executable: str = "pxt"
    command_timeout_seconds: float = 120.0
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        project_root = self.project_root.expanduser().resolve()
        if not project_root.is_dir():
            raise ValueError(f"Project root is not a directory: {project_root}")
        pixeltable_home = self.pixeltable_home.expanduser().resolve()
        if self.transport != "stdio" and self.unsafe_enabled:
            raise ValueError("Unsafe tools can only be enabled over stdio")
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "pixeltable_home", pixeltable_home)

    @classmethod
    def from_env(cls, *, transport: Transport = "stdio") -> ServerConfig:
        """Load startup-only settings from the environment."""

        project_root = Path(os.environ.get("PIXELTABLE_MCP_PROJECT_ROOT", os.getcwd()))
        pixeltable_home = Path(os.environ.get("PIXELTABLE_HOME", "~/.pixeltable"))
        unsafe_enabled = os.environ.get("PIXELTABLE_MCP_ENABLE_UNSAFE") == "1"
        pxt_executable = find_pxt_executable()
        return cls(
            project_root=project_root,
            pixeltable_home=pixeltable_home,
            transport=transport,
            unsafe_enabled=unsafe_enabled,
            pxt_executable=pxt_executable,
        )

    def apply_environment(self) -> None:
        """Pin Pixeltable to this process's catalog before it is imported."""

        os.environ["PIXELTABLE_HOME"] = str(self.pixeltable_home)
        os.environ.setdefault("PIXELTABLE_DISABLE_STDOUT", "1")

    def command_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PIXELTABLE_HOME"] = str(self.pixeltable_home)
        env.setdefault("PIXELTABLE_DISABLE_STDOUT", "1")
        return env

    def resolve_project_file(self, value: str, *, must_exist: bool = True) -> Path:
        """Resolve an application path and reject project-root escapes."""

        if not value.strip() or "\x00" in value:
            raise ToolError("File path must be a non-empty project path")
        raw = Path(value).expanduser()
        candidate = (raw if raw.is_absolute() else self.project_root / raw).resolve(strict=False)
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise ToolError("File path must stay inside the configured project root") from exc
        if must_exist and not candidate.is_file():
            raise ToolError(f"Project file does not exist: {value}")
        return candidate

    def resolve_scaffold_output(self, value: str) -> Path:
        """Validate a new project-relative scaffold path."""

        raw = Path(value)
        if raw.is_absolute():
            raise ToolError("Scaffold output must be relative to the project root")
        candidate = self.resolve_project_file(value, must_exist=False)
        if candidate.exists():
            raise ToolError(f"Refusing to overwrite existing path: {value}")
        if not candidate.parent.is_dir():
            raise ToolError(f"Scaffold output directory does not exist: {candidate.parent}")
        return candidate

    @staticmethod
    def validate_catalog_path(value: str, *, allow_empty: bool = False) -> str:
        """Validate a local Pixeltable catalog path."""

        value = value.strip().strip("/")
        if not value and allow_empty:
            return ""
        if not value or any(not _CATALOG_COMPONENT.fullmatch(part) for part in value.split("/")):
            raise ToolError(f"Invalid Pixeltable catalog path: {value!r}")
        return value

    @staticmethod
    def validate_target(value: str) -> str:
        """Validate a local catalog path or an explicit hosted database URI."""

        value = value.strip().rstrip("/")
        if value.startswith("pxt://"):
            if not _HOSTED_TARGET.fullmatch(value):
                raise ToolError(f"Invalid hosted Pixeltable target: {value!r}")
            return value
        return ServerConfig.validate_catalog_path(value)

    @staticmethod
    def validate_identifier(value: str, *, label: str = "identifier") -> str:
        value = value.strip()
        if not _CATALOG_COMPONENT.fullmatch(value):
            raise ToolError(f"Invalid {label}: {value!r}")
        return value


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    data: JsonValue

    @property
    def pending(self) -> bool:
        return self.exit_code == 2


class CommandRunner:
    """Run fixed argument arrays with captured, redacted output."""

    def __init__(self, config: ServerConfig):
        self.config = config

    async def pxt(
        self,
        arguments: Sequence[str],
        *,
        allowed_exit_codes: frozenset[int] = frozenset({0}),
        parse_json: bool = True,
    ) -> ProcessResult:
        return await self.run(
            [self.config.pxt_executable, *arguments],
            allowed_exit_codes=allowed_exit_codes,
            parse_json=parse_json,
        )

    async def worker(self, operation: str, payload: object) -> ProcessResult:
        worker_path = Path(__file__).with_name("_worker.py")
        return await self.run(
            [sys.executable, str(worker_path), operation],
            input_bytes=json.dumps(payload).encode(),
            allowed_exit_codes=frozenset({0}),
            parse_json=True,
        )

    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        allowed_exit_codes: frozenset[int],
        parse_json: bool,
    ) -> ProcessResult:
        """Execute a known command without a shell and normalize its result."""

        process: Process | None = None
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        output_limit_exceeded = False

        def terminate_process() -> None:
            if process is None or process.returncode is not None:
                return
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass

        async def drain_stream(stream: ByteReceiveStream, destination: bytearray) -> None:
            nonlocal output_limit_exceeded
            while True:
                try:
                    chunk = await stream.receive()
                except anyio.EndOfStream:
                    return
                remaining = self.config.max_output_bytes + 1 - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
                if len(destination) > self.config.max_output_bytes:
                    output_limit_exceeded = True
                    terminate_process()
                    return

        async def send_input(stream: ByteSendStream, value: bytes) -> None:
            try:
                await stream.send(value)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                if not output_limit_exceeded:
                    raise
            finally:
                await stream.aclose()

        try:
            started_process = await anyio.open_process(
                list(arguments),
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.project_root,
                env=self.config.command_environment(),
                start_new_session=os.name == "posix",
            )
            process = started_process
            with anyio.fail_after(self.config.command_timeout_seconds):
                async with anyio.create_task_group() as task_group:
                    if started_process.stdout is not None:
                        task_group.start_soon(drain_stream, started_process.stdout, stdout_buffer)
                    if started_process.stderr is not None:
                        task_group.start_soon(drain_stream, started_process.stderr, stderr_buffer)
                    if started_process.stdin is not None and input_bytes is not None:
                        task_group.start_soon(send_input, started_process.stdin, input_bytes)
                    await started_process.wait()
        except TimeoutError as exc:
            raise ToolError(
                f"Pixeltable command timed out after {self.config.command_timeout_seconds:g} seconds"
            ) from exc
        except FileNotFoundError as exc:
            raise ToolError(f"Pixeltable executable was not found: {arguments[0]}") from exc
        except OSError as exc:
            raise ToolError(f"Could not start Pixeltable command: {exc}") from exc
        finally:
            if process is not None:
                terminate_process()
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(5):
                        await process.wait()
                        await process.aclose()

        if output_limit_exceeded:
            raise ToolError("Pixeltable command output exceeded the configured limit")
        assert process is not None and process.returncode is not None
        stdout_bytes = bytes(stdout_buffer)
        stderr_bytes = bytes(stderr_buffer)
        stdout = redact_secrets(stdout_bytes.decode(errors="replace").strip())
        stderr = redact_secrets(stderr_bytes.decode(errors="replace").strip())
        if process.returncode not in allowed_exit_codes:
            detail = stderr or stdout or f"exit code {process.returncode}"
            raise ToolError(f"Pixeltable command failed: {detail}")

        data: JsonValue = None
        if parse_json:
            data = self._parse_json(stdout)
        elif stdout:
            data = stdout
        return ProcessResult(exit_code=process.returncode, stdout=stdout, stderr=stderr, data=data)

    @staticmethod
    def _parse_json(stdout: str) -> JsonValue:
        if not stdout:
            return None
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, character in enumerate(stdout):
                if character not in "[{":
                    continue
                try:
                    value, end = decoder.raw_decode(stdout[index:])
                except json.JSONDecodeError:
                    continue
                if not stdout[index + end :].strip():
                    return value
        raise ToolError("Pixeltable command returned invalid JSON")
