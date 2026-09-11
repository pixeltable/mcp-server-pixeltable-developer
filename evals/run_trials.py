#!/usr/bin/env python3
"""Run the isolated 8 x 3 x 2 Pixeltable agent evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from prepare_sources import prepare_sources

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
SCENARIOS_PATH = EVAL_ROOT / "scenarios.json"
SOURCE_LOCK_PATH = EVAL_ROOT / "sources.lock.json"
MATRIX_PATH = EVAL_ROOT / "trials.csv"
FIXTURE_ROOT = EVAL_ROOT / "fixtures"
DEFAULT_RESULTS_ROOT = EVAL_ROOT / "results"

TOOL_ITEM_TYPES = {
    "command_execution",
    "dynamic_tool_call",
    "file_change",
    "mcp_tool_call",
    "web_search",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _guidance_hash(root: Path, files: list[str]) -> str:
    records = [{"path": relative, "sha256": _sha256_file(root / relative)} for relative in sorted(files)]
    return _canonical_hash(records)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _cap_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    half = max(1, limit // 2)
    marker = f"\n... {len(value) - (2 * half)} characters omitted ...\n"
    return value[:half] + marker + value[-half:], True


def _tool_counts(transcript: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", ""))
        if item_type not in TOOL_ITEM_TYPES:
            continue
        item_id = str(item.get("id", ""))
        key = (item_type, item_id)
        if key in seen:
            continue
        seen.add(key)
        counts[item_type] += 1
    return counts


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdin: str | None,
    timeout: float,
) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
    except BaseException:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
        raise
    return process.returncode, stdout, stderr, timed_out


def _stop_daemon(pxt: Path, *, cwd: Path, env: dict[str, str]) -> None:
    with suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            [str(pxt), "daemon", "stop", "--force"],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )


def _copy_inputs(workspace: Path, source_bundle: Path, guidance_files: list[str]) -> None:
    guidance_root = workspace / "guidance"
    for relative in guidance_files:
        source = source_bundle / relative
        destination = guidance_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    fixture_root = workspace / "fixtures"
    fixture_root.mkdir(parents=True)
    manifest = _load_json(FIXTURE_ROOT / "manifest.json")
    for name in manifest["fixtures"]:
        shutil.copy2(FIXTURE_ROOT / name, fixture_root / name)
    shutil.copy2(FIXTURE_ROOT / "manifest.json", fixture_root / "manifest.json")


def _prompt(
    *,
    scenario_prompt: str,
    guidance_files: list[str],
    mcp_enabled: bool,
    max_tool_calls: int,
) -> str:
    paths = "\n".join(f"- guidance/{path}" for path in guidance_files)
    mcp_text = (
        "The Pixeltable Developer MCP server is connected as `pixeltable`; use its safe tools where useful."
        if mcp_enabled
        else "No Pixeltable MCP server is connected."
    )
    return f"""Work autonomously in this isolated evaluation directory. Complete the task and leave every requested
artifact in the current directory. You may run local commands and must test the result. Do not browse the internet,
read another repository, load user/project skills, or use guidance other than the files listed below. Read the listed
guidance before implementing. Do not make paid provider calls or create hosted resources. {mcp_text}

Guidance files:
{paths}

Use no more than {max_tool_calls} tool calls. This is one attempt: diagnose and correct problems within this session.
In the final response, state the commands actually run, concrete observed results, and any provider or Cloud boundary.

Task:
{scenario_prompt.rstrip()}
"""


def _mcp_config_args(*, python: Path, workspace: Path, catalog: Path) -> list[str]:
    environment = {
        "PIXELTABLE_DISABLE_STDOUT": "1",
        "PIXELTABLE_HOME": str(catalog),
        "PIXELTABLE_MCP_PROJECT_ROOT": str(workspace),
    }
    env_toml = ",".join(f"{key}={_toml_string(value)}" for key, value in sorted(environment.items()))
    return [
        "-c",
        f"mcp_servers.pixeltable.command={_toml_string(str(python))}",
        "-c",
        'mcp_servers.pixeltable.args=["-m","mcp_server_pixeltable_developer"]',
        "-c",
        f"mcp_servers.pixeltable.cwd={_toml_string(str(workspace))}",
        "-c",
        f"mcp_servers.pixeltable.env={{{env_toml}}}",
        "-c",
        "mcp_servers.pixeltable.required=true",
    ]


def _read_matrix() -> tuple[list[str], list[dict[str, str]]]:
    with MATRIX_PATH.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Trial matrix has no header")
        return reader.fieldnames, list(reader)


def _write_matrix(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = MATRIX_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(MATRIX_PATH)


def _replace_row(fieldnames: list[str], rows: list[dict[str, str]], trial_id: str, values: dict[str, object]) -> None:
    row = next(row for row in rows if row["trial_id"] == trial_id)
    for field in fieldnames:
        row[field] = ""
    for field, value in values.items():
        if value is None:
            row[field] = ""
        elif isinstance(value, bool):
            row[field] = str(value).lower()
        else:
            row[field] = str(value)
    _write_matrix(fieldnames, rows)


def _quality(row: dict[str, str]) -> float:
    recovery = int(row["recovery"]) if row["recovery"] else 0
    return (
        3 * int(row["task_completion"])
        + 3 * int(row["executable_correctness"])
        + 2 * int(row["first_attempt_success"])
        + recovery
        - 2 * int(row["unsupported_api_count"])
        - int(row["unnecessary_dependency_count"])
    )


def _write_summary(run_root: Path, rows: list[dict[str, str]]) -> None:
    completed = [row for row in rows if row["status"] in {"passed", "failed"}]
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        by_condition[row["condition"]].append(row)
    lines = [
        "# Pixeltable agent evaluation results",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "| Condition | Completed | Passed | Mean quality | Mean tool calls | Mean seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("mcp_skill", "skill_only", "website_docs"):
        condition_rows = by_condition[condition]
        if condition_rows:
            mean_quality = sum(_quality(row) for row in condition_rows) / len(condition_rows)
            mean_tools = sum(int(row["tool_calls"]) for row in condition_rows) / len(condition_rows)
            mean_seconds = sum(float(row["elapsed_seconds"]) for row in condition_rows) / len(condition_rows)
            passed = sum(row["status"] == "passed" for row in condition_rows)
            lines.append(
                f"| `{condition}` | {len(condition_rows)}/16 | {passed} | {mean_quality:.2f} | "
                f"{mean_tools:.2f} | {mean_seconds:.2f} |"
            )
        else:
            lines.append(f"| `{condition}` | 0/16 | 0 | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "This generated summary is descriptive. Use `validate_results.py --require-complete` for the release gate.",
            "Two repetitions are a regression screen and do not establish statistical superiority.",
            "",
        ]
    )
    (run_root / "summary.md").write_text("\n".join(lines))


def _trial_selection(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if args.all:
        selected = rows
    else:
        requested = set(args.trial)
        known = {row["trial_id"] for row in rows}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown trial id(s): {', '.join(sorted(unknown))}")
        selected = [row for row in rows if row["trial_id"] in requested]
    if not args.rerun:
        selected = [row for row in selected if row["status"] not in {"passed", "failed"}]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run all incomplete trials in matrix order.")
    selection.add_argument("--trial", action="append", default=[], help="Run one exact trial id; may be repeated.")
    parser.add_argument("--rerun", action="store_true", help="Replace already completed selected trials.")
    parser.add_argument("--dry-run", action="store_true", help="List the selected trials without network or model use.")
    parser.add_argument("--model", default=os.environ.get("PIXELTABLE_EVAL_MODEL", "gpt-6-astra"))
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout", type=float, help="Override the per-trial timeout in scenarios.json.")
    parser.add_argument("--codex", type=Path, default=Path(shutil.which("codex") or "codex"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pxt", type=Path, default=Path(sys.executable).with_name("pxt"))
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--allow-source-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = _load_json(SCENARIOS_PATH)
    scenario_by_id = {entry["id"]: entry for entry in scenarios["scenarios"]}
    condition_by_id = {entry["id"]: entry for entry in scenarios["conditions"]}
    fieldnames, rows = _read_matrix()
    selected = _trial_selection(args, rows)
    if args.dry_run:
        print("\n".join(row["trial_id"] for row in selected))
        print(f"selected={len(selected)}")
        return 0
    if not selected:
        print("No incomplete selected trials.")
        return 0
    for executable, label in ((args.codex, "codex"), (args.python, "Python"), (args.pxt, "pxt")):
        if not executable.is_file():
            raise FileNotFoundError(f"{label} executable does not exist: {executable}")

    run_root = DEFAULT_RESULTS_ROOT / args.run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    source_bundle = run_root / "sources"
    retrievals = prepare_sources(
        lock_path=SOURCE_LOCK_PATH,
        output_dir=source_bundle,
        timeout=30,
        strict_mutable=not args.allow_source_drift,
        force=False,
    )
    codex_version = subprocess.run(
        [str(args.codex), "--version"], capture_output=True, text=True, check=True, timeout=20
    ).stdout.strip()
    settings = {
        "codex_cli": codex_version,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sandbox": "workspace-write",
        "approval": "approve-for-me",
        "ephemeral": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "pixeltable": version("pixeltable"),
        "mcp": version("mcp"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "prompt_wrapper": 1,
    }
    settings_hash = _canonical_hash(settings)
    (run_root / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")

    had_execution_error = False
    for index, matrix_row in enumerate(selected, start=1):
        trial_id = matrix_row["trial_id"]
        scenario = scenario_by_id[matrix_row["scenario"]]
        condition = condition_by_id[matrix_row["condition"]]
        guidance_key = condition["guidance"]
        guidance_files = list(retrievals["condition_files"][guidance_key])
        scenario_prompt_path = EVAL_ROOT / scenario["prompt_file"]
        trial_root = run_root / trial_id
        workspace = trial_root / "workspace"
        catalog = workspace / "agent-catalog"
        workspace.mkdir(parents=True)
        catalog.mkdir()
        _copy_inputs(workspace, source_bundle, guidance_files)
        prompt = _prompt(
            scenario_prompt=scenario_prompt_path.read_text(),
            guidance_files=guidance_files,
            mcp_enabled=bool(condition["mcp_enabled"]),
            max_tool_calls=int(scenarios["defaults"]["max_tool_calls"]),
        )
        (trial_root / "prompt.txt").write_text(prompt)
        final_message = trial_root / "final.txt"
        argv = [
            str(args.codex),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "--model",
            args.model,
            "-c",
            f"model_reasoning_effort={_toml_string(args.reasoning_effort)}",
            "--approve-for-me",
            "--cd",
            str(workspace),
            "--output-last-message",
            str(final_message),
        ]
        if condition["mcp_enabled"]:
            argv.extend(_mcp_config_args(python=args.python.absolute(), workspace=workspace, catalog=catalog))
        argv.append("-")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{args.python.absolute().parent}{os.pathsep}{environment.get('PATH', '')}",
                "PIXELTABLE_DISABLE_STDOUT": "1",
                "PIXELTABLE_HOME": str(catalog),
                "PIXELTABLE_MCP_PROJECT_ROOT": str(workspace),
            }
        )
        environment.pop("PIXELTABLE_MCP_ENABLE_UNSAFE", None)
        timeout = args.timeout or float(scenarios["defaults"]["timeout_seconds"])
        print(f"[{index}/{len(selected)}] {trial_id}", flush=True)
        _stop_daemon(args.pxt.resolve(), cwd=workspace, env=environment)
        started = time.monotonic()
        try:
            returncode, stdout, stderr, timed_out = _run_process(
                argv,
                cwd=workspace,
                env=environment,
                stdin=prompt,
                timeout=timeout,
            )
        finally:
            _stop_daemon(args.pxt.resolve(), cwd=workspace, env=environment)
        elapsed = time.monotonic() - started
        max_output = int(scenarios["defaults"]["max_output_chars"])
        transcript, transcript_truncated = _cap_text(stdout, max_output)
        stderr_record, stderr_truncated = _cap_text(stderr, max_output)
        (trial_root / "transcript.jsonl").write_text(transcript)
        (trial_root / "stderr.txt").write_text(stderr_record)
        counts = _tool_counts(stdout)
        tool_calls = sum(counts.values())
        verifier_path = trial_root / "verifier.json"
        verify_argv = [
            str(args.python.absolute()),
            str(EVAL_ROOT / "verify_trial.py"),
            "--scenario",
            scenario["id"],
            "--workspace",
            str(workspace),
            "--report",
            str(verifier_path),
            "--python",
            str(args.python.absolute()),
            "--pxt",
            str(args.pxt.resolve()),
        ]
        verify_started = time.monotonic()
        verify_returncode, verify_stdout, verify_stderr, verify_timed_out = _run_process(
            verify_argv,
            cwd=REPO_ROOT,
            env=environment,
            stdin=None,
            timeout=300,
        )
        elapsed += time.monotonic() - verify_started
        verifier = _load_json(verifier_path) if verifier_path.is_file() else None
        limit_exceeded = tool_calls > int(scenarios["defaults"]["max_tool_calls"])
        execution_error = timed_out or verify_timed_out or returncode != 0 or verifier is None
        evidence = {
            "schema_version": 1,
            "trial_id": trial_id,
            "scenario": scenario["id"],
            "condition": condition["id"],
            "repetition": int(matrix_row["repetition"]),
            "started_at": datetime.now(UTC).isoformat(),
            "settings": settings,
            "settings_sha256": settings_hash,
            "prompt_sha256": _sha256_file(scenario_prompt_path),
            "guidance_sha256": _guidance_hash(source_bundle, guidance_files),
            "source_manifest": str((source_bundle / "retrievals.json").relative_to(run_root)),
            "source_manifest_sha256": _sha256_file(source_bundle / "retrievals.json"),
            "codex_returncode": returncode,
            "codex_timed_out": timed_out,
            "transcript_sha256": _sha256_bytes(stdout.encode()),
            "transcript_bytes": len(stdout.encode()),
            "transcript_truncated": transcript_truncated,
            "stderr_sha256": _sha256_bytes(stderr.encode()),
            "stderr_bytes": len(stderr.encode()),
            "stderr_truncated": stderr_truncated,
            "final_sha256": _sha256_file(final_message) if final_message.is_file() else None,
            "tool_calls": tool_calls,
            "tool_calls_by_type": dict(counts),
            "tool_limit_exceeded": limit_exceeded,
            "elapsed_seconds": elapsed,
            "verifier_returncode": verify_returncode,
            "verifier_stdout": verify_stdout[-4000:],
            "verifier_stderr": verify_stderr[-4000:],
            "verifier_timed_out": verify_timed_out,
            "verifier": verifier,
            "boundaries": scenario["live_boundaries"],
        }
        evidence_path = trial_root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        relative_evidence = evidence_path.relative_to(EVAL_ROOT).as_posix()
        if execution_error:
            had_execution_error = True
            _replace_row(
                fieldnames,
                rows,
                trial_id,
                {
                    "trial_id": trial_id,
                    "scenario": scenario["id"],
                    "condition": condition["id"],
                    "repetition": matrix_row["repetition"],
                    "status": "error",
                    "model": args.model,
                    "settings_sha256": settings_hash,
                    "prompt_sha256": _sha256_file(scenario_prompt_path),
                    "guidance_sha256": _guidance_hash(source_bundle, guidance_files),
                    "evidence": relative_evidence,
                },
            )
            print(f"  error: codex={returncode}, verifier={verify_returncode}", flush=True)
            continue
        assert verifier is not None
        passed = bool(verifier["passed"]) and not limit_exceeded
        _replace_row(
            fieldnames,
            rows,
            trial_id,
            {
                "trial_id": trial_id,
                "scenario": scenario["id"],
                "condition": condition["id"],
                "repetition": matrix_row["repetition"],
                "status": "passed" if passed else "failed",
                "model": args.model,
                "settings_sha256": settings_hash,
                "prompt_sha256": _sha256_file(scenario_prompt_path),
                "guidance_sha256": _guidance_hash(source_bundle, guidance_files),
                "task_completion": verifier["task_completion"],
                "executable_correctness": verifier["executable_correctness"],
                "first_attempt_success": int(passed),
                "unsupported_api_count": verifier["unsupported_api_count"],
                "unnecessary_dependency_count": verifier["unnecessary_dependency_count"],
                "recovery": verifier["recovery"],
                "tool_calls": tool_calls,
                "elapsed_seconds": f"{elapsed:.3f}",
                "critical_defect": bool(verifier["critical_defect"] or limit_exceeded),
                "evidence": relative_evidence,
            },
        )
        print(f"  {'passed' if passed else 'failed'}: tools={tool_calls}, seconds={elapsed:.1f}", flush=True)

    _write_summary(run_root, rows)
    validation = subprocess.run(
        [str(args.python.absolute()), str(EVAL_ROOT / "validate_results.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    print(validation.stdout or validation.stderr, end="")
    return 1 if had_execution_error or validation.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
