"""Shared test configuration."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from mcp_server_pixeltable_developer.runtime import ServerConfig, find_pxt_executable


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests that initialize a real Pixeltable catalog and service.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ServerConfig]:
    project_root = tmp_path / "project"
    pixeltable_home = tmp_path / "catalog"
    project_root.mkdir()
    pixeltable_home.mkdir()
    monkeypatch.delenv("PIXELTABLE_MCP_ENABLE_UNSAFE", raising=False)
    monkeypatch.setenv("PIXELTABLE_MCP_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("PIXELTABLE_HOME", str(pixeltable_home))
    monkeypatch.setenv("PIXELTABLE_DISABLE_STDOUT", "1")
    # One pxt daemon serves one project root per port, and a client from another project restarts it.
    # A private port keeps other pxt users on the machine from killing this test's daemon mid-call.
    monkeypatch.setenv("PXT_PORT", str(_free_port()))
    pxt_executable = find_pxt_executable()
    if not Path(pxt_executable).is_file():
        pytest.fail("pxt executable is unavailable in the test environment")
    config = ServerConfig(
        project_root=project_root,
        pixeltable_home=pixeltable_home,
        pxt_executable=pxt_executable,
        command_timeout_seconds=60,
    )
    yield config
    subprocess.run(
        [pxt_executable, "daemon", "stop"],
        cwd=project_root,
        env=config.command_environment(),
        capture_output=True,
        check=False,
        timeout=30,
    )
    # Postgres outlives `pxt daemon stop`, and each instance holds a SysV shared-memory segment; macOS
    # allows 32 machine-wide, so leaked instances make later initdb calls fail. The first line of
    # postmaster.pid is the postmaster's PID, and SIGINT is Postgres's fast shutdown.
    postmaster = pixeltable_home / "pgdata" / "postmaster.pid"
    with suppress(FileNotFoundError, IndexError, ValueError, ProcessLookupError, PermissionError):
        os.kill(int(postmaster.read_text().split()[0]), signal.SIGINT)
