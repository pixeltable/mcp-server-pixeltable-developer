"""Release-evaluation manifest, fixture, runner, and gate tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "evals"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_matrix_and_conditional_release_gate() -> None:
    validator = _load_module("pixeltable_eval_validator", EVAL_ROOT / "validate_results.py")
    rows, errors = validator.validate_structure(EVAL_ROOT / "trials.csv")
    assert errors == []
    assert len(rows) == 48
    assert len({(row["scenario"], row["condition"], row["repetition"]) for row in rows}) == 48

    gate_errors = validator.validate_release_gate(rows)
    incomplete = [row for row in rows if row["status"] not in {"passed", "failed"}]
    if incomplete:
        assert gate_errors and gate_errors[0].startswith("release gate requires all 48 trials")


def test_eval_fixtures_match_their_manifest() -> None:
    completed = subprocess.run(
        [sys.executable, str(EVAL_ROOT / "fixtures/generate.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload["fixtures"]) == {"document.html", "image.png", "video.mp4", "audio.wav"}


def test_eval_runner_dry_run_selects_the_complete_matrix_without_model_use() -> None:
    completed = subprocess.run(
        [sys.executable, str(EVAL_ROOT / "run_trials.py"), "--all", "--rerun", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[-1] == "selected=48"
    assert len(lines[:-1]) == 48


def test_eval_transcript_tool_counting_and_bounded_capture() -> None:
    sys.path.insert(0, str(EVAL_ROOT))
    try:
        runner = _load_module("pixeltable_eval_runner", EVAL_ROOT / "run_trials.py")
    finally:
        sys.path.remove(str(EVAL_ROOT))
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "one", "type": "command_execution"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "two", "type": "mcp_tool_call"},
                }
            ),
            json.dumps({"type": "item.completed", "item": {"id": "three", "type": "agent_message"}}),
        ]
    )
    assert sum(runner._tool_counts(transcript).values()) == 2
    capped, truncated = runner._cap_text("x" * 200, 50)
    assert truncated is True
    assert "characters omitted" in capped
