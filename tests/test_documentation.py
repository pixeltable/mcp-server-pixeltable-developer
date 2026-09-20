"""Package metadata, shipped guidance, and deprecated-pattern release gates."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOOL_NAMES = [
    "pixeltable_list_catalog",
    "pixeltable_describe",
    "pixeltable_rows",
    "pixeltable_get_row",
    "pixeltable_errors",
    "pixeltable_insert_rows",
    "pixeltable_recompute",
    "pixeltable_scaffold_app",
    "pixeltable_schema_check",
    "pixeltable_schema_diff",
    "pixeltable_schema_update",
    "pixeltable_schema_prune",
    "pixeltable_service_check",
    "pixeltable_service_diff",
    "pixeltable_service_update",
    "pixeltable_service_list",
    "pixeltable_service_stop",
    "pixeltable_service_prune",
]


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


def test_packaging_metadata_agrees_with_pyproject_and_the_served_contract() -> None:
    """The bundle manifest, registry entry, and Smithery config are submitted for review, so they must not drift."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = metadata["project"]["version"]

    manifest = json.loads((ROOT / "mcpb" / "manifest.json").read_text())
    assert manifest["version"] == version
    # uv is the MCPB runtime for Python servers: the host manages the interpreter and installs deps.
    assert manifest["server"]["type"] == "uv"
    assert manifest["server"]["mcp_config"]["command"] == "uv"
    # A missing or incomplete privacy policy is an automatic directory rejection.
    assert manifest["privacy_policies"], "the directory requires at least one privacy policy URL"
    assert all(url.startswith("https://") for url in manifest["privacy_policies"])
    assert "## Privacy Policy" in (ROOT / "README.md").read_text()
    # Both env vars are required: without them the server aims at cwd and ~/.pixeltable.
    assert all(manifest["user_config"][key]["required"] is True for key in ("project_root", "pixeltable_home"))
    # The portal syncs tools from the running server; a static list only drifts. If one is ever added back,
    # it must match the served contract.
    if "tools" in manifest:
        assert {tool["name"] for tool in manifest["tools"]} == set(DEFAULT_TOOL_NAMES)

    registry = json.loads((ROOT / "server.json").read_text())
    assert registry["version"] == version
    package = registry["packages"][0]
    assert package["version"] == version
    assert package["identifier"] == metadata["project"]["name"]
    assert {var["name"]: var["isRequired"] for var in package["environmentVariables"]} == {
        "PIXELTABLE_MCP_PROJECT_ROOT": True,
        "PIXELTABLE_HOME": True,
    }

    smithery = (ROOT / "smithery.yaml").read_text()
    assert "required: [projectRoot, pixeltableHome]" in smithery
