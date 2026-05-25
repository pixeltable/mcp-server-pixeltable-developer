"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Allow tests to run without installing the package first (e.g. local dev,
# uv run pytest directly against the repo).
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow was passed.

    Aligned with the pixeltable-starter-kit's pattern: fast suite by default,
    opt-in to the smoke test that actually initializes Pixeltable on a temp
    PIXELTABLE_HOME.
    """
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that initialize a real Pixeltable instance.",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def isolated_pixeltable_home(tmp_path, monkeypatch):
    """Point PIXELTABLE_HOME at a temp directory so tests never touch user data."""
    home = tmp_path / "pxt-home"
    home.mkdir()
    monkeypatch.setenv("PIXELTABLE_HOME", str(home))
    monkeypatch.setenv("PIXELTABLE_DISABLE_STDOUT", "1")
    return home
