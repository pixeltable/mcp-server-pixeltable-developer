"""Package metadata, shipped guidance, and deprecated-pattern release gates."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fenced_code(text: str) -> str:
    return "\n".join(re.findall(r"```(?:python|bash|sh|json)?\n(.*?)```", text, flags=re.DOTALL))


def test_release_metadata_and_lock_are_pinned_to_reviewed_lines() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["version"] == "0.2.0"
    assert project["requires-python"] == ">=3.11"
    assert "mcp>=2.2,<3" in project["dependencies"]
    assert "pixeltable[serve]>=0.7.6,<0.8" in project["dependencies"]
    assert all(not dependency.startswith(("requests", "toml", "uvloop")) for dependency in project["dependencies"])
    assert project["scripts"]["mcp-server-pixeltable-developer"] == ("mcp_server_pixeltable_developer.__main__:main")

    lock = (ROOT / "uv.lock").read_text()
    assert 'name = "mcp"\nversion = "2.2.0"' in lock
    assert 'name = "pixeltable"\nversion = "0.7.6"' in lock
    assert "9e6cdbe54f042b31786bede4a4cd4d68b361b31d75f34ea4e9c415146c901114" in lock


def test_public_examples_do_not_use_retired_workflows() -> None:
    public_docs = "\n".join((ROOT / path).read_text() for path in ["README.md", "docs/migration-0.1-to-0.2.md"])
    code = _fenced_code(public_docs)
    banned = {
        "pxt.Required": "non-nullable types and T | None",
        "create_default_idxs": "has_default_idxs",
        "pixeltable-new": "pxt service example",
        "--template": "the single generated application",
        "from pixeltable.iterators": "pixeltable.functions iterators",
        "openai.vision(": "chat_completions or responses",
    }
    for pattern, replacement in banned.items():
        assert pattern not in code, f"replace {pattern!r} with {replacement}"
    assert re.search(r"\.similarity\(\s*['\"]", code) is None


def test_default_source_has_no_execution_or_private_manager_path() -> None:
    safe_files = [
        ROOT / "src/mcp_server_pixeltable_developer/models.py",
        ROOT / "src/mcp_server_pixeltable_developer/prompts.py",
        ROOT / "src/mcp_server_pixeltable_developer/resources.py",
        ROOT / "src/mcp_server_pixeltable_developer/runtime.py",
        ROOT / "src/mcp_server_pixeltable_developer/server.py",
        ROOT / "src/mcp_server_pixeltable_developer/tools.py",
    ]
    safe_source = "\n".join(path.read_text() for path in safe_files)
    assert "eval(" not in safe_source
    assert "exec(" not in safe_source
    assert "shell=True" not in safe_source
    assert "_tool_manager" not in safe_source
    assert "_resource_manager" not in safe_source
    assert "_prompt_manager" not in safe_source
    assert "create_default_idxs" not in safe_source
    assert "pixeltable-new" not in safe_source


def test_legacy_package_contains_only_one_release_shims() -> None:
    legacy = ROOT / "src/mcp_server_pixeltable_stio"
    shipped = sorted(path.relative_to(legacy).as_posix() for path in legacy.rglob("*.py"))
    assert shipped == ["__init__.py", "__main__.py", "server.py"]
    assert "file=sys.stderr" in (legacy / "__init__.py").read_text()


def test_canvas_uses_safe_dom_and_authenticated_fetch() -> None:
    canvas = (ROOT / "canvas.html").read_text()
    assert "innerHTML" not in canvas
    assert "EventSource" not in canvas
    assert "Authorization" in canvas
    assert "Bearer" in canvas
    assert "textContent" in canvas
    server = (ROOT / "src/mcp_server_pixeltable_developer/canvas.py").read_text()
    assert '"*"' not in server
    assert "hmac.compare_digest" in server
    assert "O_NOFOLLOW" in server


def test_evidence_report_is_reproducible_and_explicit_about_boundaries() -> None:
    report = (ROOT / "docs/review-0.1.0.md").read_text()
    for value in [
        "5ae8063e07ea6ad1aa625039e20475986d7a7750",
        "f550e6ed757b48635e4f53900840f4e9a1fb4c93",
        "9e6cdbe54f042b31786bede4a4cd4d68b361b31d75f34ea4e9c415146c901114",
        "2026-09-09",
        "38 passed",
        "not live-tested",
    ]:
        assert value in report
