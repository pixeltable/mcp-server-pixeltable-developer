"""Shared test configuration."""

from __future__ import annotations

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


@pytest.fixture
def server_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ServerConfig:
    project_root = tmp_path / "project"
    pixeltable_home = tmp_path / "catalog"
    project_root.mkdir()
    pixeltable_home.mkdir()
    monkeypatch.delenv("PIXELTABLE_MCP_ENABLE_UNSAFE", raising=False)
    monkeypatch.setenv("PIXELTABLE_MCP_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("PIXELTABLE_HOME", str(pixeltable_home))
    monkeypatch.setenv("PIXELTABLE_DISABLE_STDOUT", "1")
    pxt_executable = find_pxt_executable()
    if not Path(pxt_executable).is_file():
        pytest.fail("pxt executable is unavailable in the test environment")
    return ServerConfig(
        project_root=project_root,
        pixeltable_home=pixeltable_home,
        pxt_executable=pxt_executable,
        command_timeout_seconds=60,
    )
