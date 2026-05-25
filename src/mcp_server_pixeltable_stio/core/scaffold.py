"""Project scaffolding tools wrapping the `pixeltable-new` CLI.

These tools surface the pixeltable-starter-kit patterns and templates so an MCP
client can bootstrap a new project (FastAPI backend, batch pipeline, `pxt serve`
service, multimodal RAG demo, etc.) without leaving the conversation.

Strategy:
    1. Try to import `pixeltable_new.new` (preferred — same Python env, no shell).
    2. Fall back to `uvx pixeltable-new --json` if the package isn't installed.

If neither is available we return a clear install hint instead of failing silently.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _import_new_module():
    """Return the in-process `pixeltable_new.new` module, or None if not installed."""
    try:
        from pixeltable_new import new as _new  # type: ignore
        return _new
    except ImportError:
        return None


def _uvx_available() -> bool:
    return shutil.which('uvx') is not None


def _install_hint() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "Neither the `pixeltable_new` package nor `uvx` is available. "
            "Install with `uv tool install pixeltable-new` (or `pip install pixeltable-new`) "
            "and retry."
        ),
    }


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def pixeltable_list_project_templates() -> Dict[str, Any]:
    """List the available `pixeltable-new` patterns and templates.

    Returns:
        Dict with `patterns` and `templates` (name -> description), the
        recommended next steps per pattern/template, and the backend used
        (`module` if pixeltable_new is importable, otherwise `uvx`).
    """
    mod = _import_new_module()
    if mod is not None:
        try:
            return {
                "success": True,
                "backend": "module",
                "patterns": {p: f"Structural pattern: {p}" for p in mod.PATTERNS},
                "templates": dict(mod.TEMPLATE_DESCRIPTIONS),
                "next_steps": {
                    "patterns": dict(mod.NEXT_STEPS),
                    "templates": dict(mod.TEMPLATE_NEXT_STEPS),
                },
            }
        except Exception as e:
            logger.error(f"pixeltable_new module call failed: {e}")
            # fall through to uvx attempt

    if not _uvx_available():
        return _install_hint()

    try:
        result = subprocess.run(
            ['uvx', 'pixeltable-new', '--list', '--json'],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "uvx pixeltable-new --list timed out after 60s"}
    except Exception as e:
        return {"success": False, "error": f"uvx invocation failed: {e}"}

    if result.returncode != 0:
        return {
            "success": False,
            "error": f"uvx exited with code {result.returncode}",
            "stderr": result.stderr,
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse --json output: {e}", "stdout": result.stdout}

    return {"success": True, "backend": "uvx", **payload}


def pixeltable_scaffold_project(
    project: Optional[str] = None,
    pattern: Optional[str] = None,
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new Pixeltable project from the starter kit.

    Exactly one of `pattern` or `template` should be supplied; if neither, the
    default pattern (`serving`) is used. The project is created in a sibling
    directory under the current working directory if `project` is given, or
    initialized in the current directory otherwise (matching `pixeltable-new`'s
    CLI semantics).

    Args:
        project: Optional project directory name. Omit to scaffold into cwd.
        pattern: One of `serving`, `backend`, `batch` (mutually exclusive with `template`).
        template: One of the named templates (e.g. `multimodal-rag`, `agent`,
            `video-intel`, `audio-intel`, `data-lab`, `content-pipeline`,
            `full-stack-showcase`).

    Returns:
        Dict mirroring the `pixeltable-new --json` shape: `project`, `files`,
        chosen `pattern`/`template`, and `next_steps`.
    """
    if pattern and template:
        return {
            "success": False,
            "error": "Pass either `pattern` or `template`, not both.",
        }

    mod = _import_new_module()
    if mod is not None:
        chosen_pattern = pattern or 'serving'
        try:
            project_path, files = mod.scaffold(
                project_name=project,
                pattern=chosen_pattern,
                template=template,
            )
        except (ValueError, FileExistsError, RuntimeError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"pixeltable_new.scaffold failed: {e}")
            return {"success": False, "error": f"scaffold failed: {e}"}

        next_steps = mod.TEMPLATE_NEXT_STEPS.get(template) if template else mod.NEXT_STEPS.get(chosen_pattern, [])
        return {
            "success": True,
            "backend": "module",
            "project": str(project_path),
            "files": files,
            "pattern": chosen_pattern if not template else None,
            "template": template,
            "next_steps": next_steps,
        }

    if not _uvx_available():
        return _install_hint()

    cmd = ['uvx', 'pixeltable-new']
    if pattern:
        cmd.append(f'--{pattern}')
    if template:
        cmd.extend(['--template', template])
    if project:
        cmd.append(project)
    cmd.append('--json')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"uvx pixeltable-new timed out after 180s ({' '.join(cmd)})"}
    except Exception as e:
        return {"success": False, "error": f"uvx invocation failed: {e}"}

    payload: Dict[str, Any] = {}
    # success goes to stdout; errors go to stderr (also JSON in 0.1.3+)
    output = result.stdout.strip() or result.stderr.strip()
    if output:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            payload = {"raw": output}

    if result.returncode != 0:
        return {
            "success": False,
            "error": payload.get('message') or 'uvx pixeltable-new failed',
            "stderr": result.stderr,
            **{k: v for k, v in payload.items() if k != 'message'},
        }
    payload.setdefault('success', True)
    payload['backend'] = 'uvx'
    return payload
