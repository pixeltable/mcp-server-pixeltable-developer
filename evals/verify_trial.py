#!/usr/bin/env python3
"""Verify one agent-produced scenario in a second, clean Pixeltable catalog."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = EVAL_ROOT / "scenarios.json"
FIXTURE_ROOT = EVAL_ROOT / "fixtures"
OUTPUT_LIMIT = 12_000

UNSUPPORTED_PATTERNS = {
    "pxt.Required": re.compile(r"\bpxt\.Required\s*\["),
    "create_default_idxs": re.compile(r"\bcreate_default_idxs\b"),
    "pixeltable-new": re.compile(r"\bpixeltable-new\b"),
    "--template": re.compile(r"(?:^|\s)--template(?:\s|$)"),
    "deprecated iterator package": re.compile(r"(?:from|import)\s+pixeltable\.iterators\b"),
    "openai.vision": re.compile(r"\b(?:openai|pxtf\.openai)\.vision\s*\("),
    "positional similarity": re.compile(
        r"\.similarity\(\s*(?!string\s*=|image\s*=|audio\s*=|video\s*=|document\s*=|vector\s*=|idx\s*=)[^)]+"
    ),
    "eval": re.compile(r"\beval\s*\("),
    "exec": re.compile(r"\bexec\s*\("),
}
UNNECESSARY_IMPORTS = re.compile(
    r"^\s*(?:from|import)\s+(langchain|llama_index|haystack|pandas|chromadb|faiss|qdrant|pinecone|weaviate)\b",
    re.MULTILINE,
)


@dataclass(slots=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class TrialVerifier:
    def __init__(self, *, scenario_id: str, workspace: Path, python: Path, pxt: Path) -> None:
        scenario_config = json.loads(SCENARIOS_PATH.read_text())
        self.scenario = next(entry for entry in scenario_config["scenarios"] if entry["id"] == scenario_id)
        self.workspace = workspace.resolve()
        # Preserve a virtual-environment symlink: resolving it selects the base
        # interpreter and loses that environment's installed packages.
        self.python = python.absolute()
        self.pxt = pxt.resolve()
        self.checks: list[Check] = []
        self.commands: list[CommandResult] = []
        self.unsupported_api_count = 0
        self.unnecessary_dependency_count = 0
        self.source_valid = True
        self.project: Path | None = None
        self.environment: dict[str, str] = {}

    def add_check(self, name: str, passed: bool, detail: str) -> bool:
        self.checks.append(Check(name=name, passed=passed, detail=detail[:OUTPUT_LIMIT]))
        return passed

    def run(self, argv: list[str], *, timeout: float = 120.0, expected: set[int] | None = None) -> CommandResult:
        assert self.project is not None
        expected = expected or {0}
        try:
            completed = subprocess.run(
                argv,
                cwd=self.project,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            result = CommandResult(
                argv=argv,
                returncode=completed.returncode,
                stdout=completed.stdout[-OUTPUT_LIMIT:],
                stderr=completed.stderr[-OUTPUT_LIMIT:],
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                argv=argv,
                returncode=124,
                stdout=(exc.stdout or "")[-OUTPUT_LIMIT:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-OUTPUT_LIMIT:] if isinstance(exc.stderr, str) else "",
            )
        self.commands.append(result)
        self.add_check(
            f"command {len(self.commands)}",
            result.returncode in expected,
            f"returncode={result.returncode}; argv={argv}; stderr={result.stderr}",
        )
        return result

    def pxt_command(
        self, arguments: list[str], *, expected: set[int] | None = None, timeout: float = 120.0
    ) -> CommandResult:
        return self.run([str(self.pxt), *arguments], expected=expected, timeout=timeout)

    def probe(self, code: str, *, timeout: float = 120.0) -> Any:
        marker = "PIXELTABLE_EVAL_JSON:"
        wrapped = f"{code.rstrip()}\nprint({marker!r} + json.dumps(result, sort_keys=True, default=str))\n"
        result = self.run([str(self.python), "-c", wrapped], timeout=timeout)
        if result.returncode != 0:
            return None
        line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith(marker)), None)
        if line is None:
            self.add_check("probe output", False, "probe did not emit its JSON result marker")
            return None
        try:
            return json.loads(line[len(marker) :])
        except json.JSONDecodeError as exc:
            self.add_check("probe output", False, f"invalid probe JSON: {exc}")
            return None

    def _copy_workspace(self, destination: Path) -> bool:
        blocked_parts = {".git", ".pixeltable", "catalog", "agent-catalog", "verification-catalog", "__pycache__"}
        copied = 0
        for source in self.workspace.rglob("*"):
            relative = source.relative_to(self.workspace)
            if any(part in blocked_parts or part.startswith(".venv") for part in relative.parts):
                continue
            if source.is_symlink():
                self.add_check("workspace containment", False, f"symlink is not allowed in trial output: {relative}")
                continue
            if not source.is_file():
                continue
            if source.stat().st_size > 10_000_000:
                self.add_check("workspace size", False, f"file exceeds 10 MB verification limit: {relative}")
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        return self.add_check("workspace copied", copied > 0, f"copied {copied} files into clean verification project")

    def _required_files(self) -> bool:
        missing = [name for name in self.scenario["required_files"] if not (self.workspace / name).is_file()]
        return self.add_check(
            "required artifacts",
            not missing,
            "all required artifacts exist" if not missing else f"missing: {', '.join(missing)}",
        )

    def _scan_source(self) -> None:
        unsupported: list[str] = []
        unnecessary: list[str] = []
        for path in self.workspace.rglob("*.py"):
            if path.is_symlink() or any(part.startswith(".") for part in path.relative_to(self.workspace).parts):
                continue
            try:
                source = path.read_text()
                ast.parse(source)
                self.add_check(f"parse {path.relative_to(self.workspace)}", True, "valid Python syntax")
            except (OSError, SyntaxError) as exc:
                self.source_valid = False
                self.add_check(f"parse {path.relative_to(self.workspace)}", False, str(exc))
                continue
            for name, pattern in UNSUPPORTED_PATTERNS.items():
                count = len(pattern.findall(source))
                if count:
                    unsupported.append(f"{path.relative_to(self.workspace)}: {name} x{count}")
                    self.unsupported_api_count += count
            imports = UNNECESSARY_IMPORTS.findall(source)
            if imports:
                unnecessary.extend(f"{path.relative_to(self.workspace)}: {name}" for name in imports)
                self.unnecessary_dependency_count += len(imports)
        self.add_check(
            "supported APIs",
            not unsupported,
            "no unsupported APIs" if not unsupported else "; ".join(unsupported),
        )
        self.add_check(
            "dependency discipline",
            not unnecessary,
            "no unnecessary frameworks or stores" if not unnecessary else "; ".join(unnecessary),
        )

    def _contains_all(self, filename: str, tokens: list[str], name: str) -> bool:
        path = self.workspace / filename
        if not path.is_file():
            return self.add_check(name, False, f"missing {filename}")
        text = path.read_text()
        normalized = re.sub(r"\s+", "", text).replace('"', "'").lower()
        missing = [token for token in tokens if re.sub(r"\s+", "", token).replace('"', "'").lower() not in normalized]
        return self.add_check(
            name, not missing, "required constructs present" if not missing else f"missing: {missing}"
        )

    def _semantic_source_check(self) -> bool:
        scenario_id = self.scenario["id"]
        requirements = {
            "initial_http_app": ("app.py", ["TableModel", "FastAPIRouter", "name='notes'", "upper"]),
            "document_retrieval": (
                "app.py",
                ["TableModel", "pxt.Document", "document_splitter", "separators='paragraph'", "search_documents"],
            ),
            "image_video": ("app.py", ["pxt.Image", "pxt.Video", "rotate(90)", "frame_iterator", "fps=2"]),
            "audio_processing": ("app.py", ["pxt.Audio", "audio_splitter", "duration=0.25"]),
            "tool_calling": ("app.py", ["add_numbers", "pxt.tools", "invoke_tools", "agent_runs"]),
            "failed_computation_recovery": ("app.py", ["recovery.ready", "ValueError", "recovered"]),
            "cloud_preparation": ("app.py", ["TableModel", "FastAPIRouter", "messages"]),
        }
        if scenario_id == "schema_evolution":
            checks = [
                self._contains_all("schema_v1.py", ["TableModel", "name='items'", "normalized"], "schema v1 shape"),
                self._contains_all("schema_v2.py", ["note:", "| None"], "schema v2 additive shape"),
                self._contains_all(
                    "schema_expression_change.py", ["lower", "normalized"], "unsupported expression fixture"
                ),
                self._contains_all(
                    "SCHEMA_EVOLUTION.md",
                    ["schema diff", "schema update", "allow-destructive", "unsupported", "rename"],
                    "schema evolution evidence",
                ),
            ]
            return all(checks)
        filename, tokens = requirements[scenario_id]
        checks = [self._contains_all(filename, tokens, "scenario source shape")]
        if scenario_id == "failed_computation_recovery":
            checks.append(
                self._contains_all("RECOVERY.md", ["errors", "recompute", "errors-only", "dry"], "recovery evidence")
            )
        if scenario_id == "cloud_preparation":
            checks.extend(
                [
                    self._contains_all(
                        "pixeltable.cloud.toml.example",
                        ["pxt://acme:main", "PIXELTABLE_API_KEY"],
                        "Cloud configuration",
                    ),
                    self._contains_all(
                        "CLOUD.md",
                        ["pxt db update", "pxt schema update", "pxt service update", "not live-tested"],
                        "Cloud runbook",
                    ),
                ]
            )
            cloud_text = (self.workspace / "CLOUD.md").read_text() if (self.workspace / "CLOUD.md").is_file() else ""
            order = [
                cloud_text.find(command) for command in ("pxt db update", "pxt schema update", "pxt service update")
            ]
            checks.append(
                self.add_check(
                    "Cloud command order", all(index >= 0 for index in order) and order == sorted(order), str(order)
                )
            )
        return all(checks)

    def _initialize(self) -> bool:
        result = self.pxt_command(["init", "--json"])
        return result.returncode == 0

    def _schema(self, filename: str, target: str, *, apply: bool = True) -> bool:
        check = self.pxt_command(["schema", "check", filename, "--json"])
        if check.returncode != 0:
            return False
        if apply:
            update = self.pxt_command(["schema", "update", filename, target, "-f", "--json"], timeout=180)
            return update.returncode == 0
        return True

    def _verify_initial_http_app(self) -> bool:
        if not self._schema("app.py", "eval_http"):
            return False
        started = False
        try:
            update = self.pxt_command(["service", "update", "app.py", "eval_http", "-f", "--json"], timeout=180)
            if update.returncode != 0:
                return False
            started = True
            listing = self.pxt_command(["service", "list", "eval_http", "--json"])
            try:
                payload = json.loads(listing.stdout)
                services = payload if isinstance(payload, list) else payload.get("services", [])
                endpoint = services[0]["endpoint"]
                request = urllib.request.Request(
                    f"{endpoint}/notes",
                    data=json.dumps({"id": 1, "text": "hello"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    body = json.loads(response.read())
                return self.add_check("HTTP behavior", body == {"upper": "HELLO"}, f"response={body}")
            except (IndexError, KeyError, json.JSONDecodeError, OSError, TypeError) as exc:
                return self.add_check("HTTP behavior", False, str(exc))
        finally:
            if started:
                self.pxt_command(["service", "stop", "eval_http/api", "--json"], expected={0, 1})

    def _verify_document_retrieval(self) -> bool:
        if not self._schema("app.py", "eval_docs"):
            return False
        payload = self.probe(
            """
import json
from pathlib import Path
import pixeltable as pxt
import app
app.TableModel.bind_all('eval_docs')
docs = pxt.get_table('eval_docs.docs')
docs.insert([{'id': 1, 'document': str(Path('fixtures/document.html').resolve())}])
chunks = pxt.get_table('eval_docs.chunks')
chunk_rows = list(chunks.select(text=chunks.text).collect())
search_rows = list(app.search_documents('beta').collect())
result = {'chunks': chunk_rows, 'search': search_rows}
"""
        )
        matches = [] if not isinstance(payload, dict) else payload.get("search", [])
        passed = bool(matches) and "beta second paragraph" in json.dumps(matches).lower()
        return self.add_check("document retrieval behavior", passed, f"probe={payload}")

    def _verify_image_video(self) -> bool:
        if not self._schema("app.py", "eval_media"):
            return False
        payload = self.probe(
            """
import json
from pathlib import Path
import pixeltable as pxt
media = pxt.get_table('eval_media.media')
media.insert([{
    'id': 1,
    'image': str(Path('fixtures/image.png').resolve()),
    'video': str(Path('fixtures/video.mp4').resolve()),
}])
media_rows = list(media.select(rotated=media.rotated).collect())
frames = pxt.get_table('eval_media.frames')
frame_rows = list(frames.select(pos=frames.pos, frame_attrs=frames.frame_attrs).collect())
result = {'rotated': str(media_rows[0]['rotated']), 'frame_count': len(frame_rows), 'first': frame_rows[0]}
""",
            timeout=180,
        )
        passed = (
            isinstance(payload, dict) and payload.get("frame_count", 0) >= 3 and "Image" in payload.get("rotated", "")
        )
        return self.add_check("image/video behavior", passed, f"probe={payload}")

    def _verify_audio_processing(self) -> bool:
        if not self._schema("app.py", "eval_audio"):
            return False
        payload = self.probe(
            """
import json
from pathlib import Path
import pixeltable as pxt
recordings = pxt.get_table('eval_audio.recordings')
recordings.insert([{'id': 1, 'audio': str(Path('fixtures/audio.wav').resolve())}])
segments = pxt.get_table('eval_audio.audio_segments')
rows = list(segments.select(
    pos=segments.pos,
    segment_start=segments.segment_start,
    segment_end=segments.segment_end,
).collect())
result = {'count': len(rows), 'last_end': rows[-1]['segment_end']}
""",
            timeout=180,
        )
        passed = isinstance(payload, dict) and payload.get("count") == 4 and payload.get("last_end") == 1.0
        return self.add_check("audio behavior", passed, f"probe={payload}")

    def _verify_tool_calling(self) -> bool:
        if not self._schema("app.py", "eval_agent"):
            return False
        payload = self.probe(
            """
import json
import pixeltable as pxt
runs = pxt.get_table('eval_agent.agent_runs')
runs.insert([{'id': 1, 'question': 'add two and three'}])
rows = list(runs.select(tool_results=runs.tool_results).collect())
result = rows[0]['tool_results']
"""
        )
        passed = payload == {"add_numbers": [5]}
        return self.add_check("tool-call wiring", passed, f"probe={payload}; provider response was mocked")

    def _json_payload(self, result: CommandResult) -> dict[str, Any]:
        try:
            value = json.loads(result.stdout)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _verify_schema_evolution(self) -> bool:
        if not self._schema("schema_v1.py", "eval_evolution"):
            return False
        additive = self.pxt_command(["schema", "diff", "schema_v2.py", "eval_evolution", "--json"], expected={0, 2})
        additive_summary = self._json_payload(additive).get("summary", {})
        additive_ok = additive_summary.get("update_additive") == 1
        self.add_check("additive diff", additive_ok, f"summary={additive_summary}")
        apply_v2 = self.pxt_command(["schema", "update", "schema_v2.py", "eval_evolution", "-f", "--json"])
        destructive = self.pxt_command(["schema", "diff", "schema_v1.py", "eval_evolution", "--json"], expected={0, 2})
        destructive_summary = self._json_payload(destructive).get("summary", {})
        destructive_ok = destructive_summary.get("update_destructive") == 1
        self.add_check("destructive diff", destructive_ok, f"summary={destructive_summary}")
        refused = self.pxt_command(["schema", "update", "schema_v1.py", "eval_evolution", "-f", "--json"], expected={3})
        allowed = self.pxt_command(
            [
                "schema",
                "update",
                "schema_v1.py",
                "eval_evolution",
                "--allow-destructive",
                "-f",
                "--json",
            ]
        )
        unsupported = self.pxt_command(
            ["schema", "diff", "schema_expression_change.py", "eval_evolution", "--json"], expected={0, 2}
        )
        unsupported_summary = self._json_payload(unsupported).get("summary", {})
        unsupported_ok = unsupported_summary.get("unsupported") == 1
        self.add_check("unsupported expression diff", unsupported_ok, f"summary={unsupported_summary}")
        rejected = self.pxt_command(
            [
                "schema",
                "update",
                "schema_expression_change.py",
                "eval_evolution",
                "--allow-destructive",
                "-f",
                "--json",
            ],
            expected={1},
        )
        return all(
            (
                additive_ok,
                apply_v2.returncode == 0,
                destructive_ok,
                refused.returncode == 3,
                allowed.returncode == 0,
                unsupported_ok,
                rejected.returncode == 1,
            )
        )

    def _verify_failed_computation_recovery(self) -> bool:
        if not self._schema("app.py", "eval_recovery"):
            return False
        inserted = self.probe(
            """
import json
import pixeltable as pxt
items = pxt.get_table('eval_recovery.items')
status = items.insert([{'id': 1, 'value': 'one'}, {'id': 2, 'value': 'two'}], on_error='ignore')
result = {'exceptions': status.num_excs}
"""
        )
        errors_before = self.pxt_command(["errors", "eval_recovery/items", "--col", "recovered", "--json"])
        try:
            error_count = len(json.loads(errors_before.stdout))
        except (TypeError, json.JSONDecodeError):
            error_count = -1
        preview = self.pxt_command(
            ["recompute", "eval_recovery/items", "recovered", "--errors-only", "-n", "--json"], expected={0, 2}
        )
        assert self.project is not None
        (self.project / "recovery.ready").touch()
        applied = self.pxt_command(
            ["recompute", "eval_recovery/items", "recovered", "--errors-only", "-f", "--json"], timeout=180
        )
        errors_after = self.pxt_command(["errors", "eval_recovery/items", "--json"])
        try:
            remaining = len(json.loads(errors_after.stdout))
        except (TypeError, json.JSONDecodeError):
            remaining = -1
        row = self.probe(
            """
import json
import pixeltable as pxt
items = pxt.get_table('eval_recovery.items')
rows = list(items.where(items.id == 2).select(recovered=items.recovered).collect())
result = rows[0]
"""
        )
        passed = (
            isinstance(inserted, dict)
            and inserted.get("exceptions") == 2
            and error_count == 2
            and preview.returncode in {0, 2}
            and applied.returncode == 0
            and remaining == 0
            and isinstance(row, dict)
            and row.get("recovered") == "TWO"
        )
        return self.add_check(
            "failed-computation recovery",
            passed,
            f"insert={inserted}; before={error_count}; after={remaining}; row={row}",
        )

    def _verify_cloud_preparation(self) -> bool:
        schema_ok = self._schema("app.py", "eval_cloud_local_check", apply=False)
        config_path = self.workspace / "pixeltable.cloud.toml.example"
        runbook_path = self.workspace / "CLOUD.md"
        text = "\n".join(path.read_text() for path in (config_path, runbook_path) if path.is_file())
        secret_like = re.search(r"(?:api[_-]?key|PIXELTABLE_API_KEY)\s*=\s*['\"](?!\$|\{|<)[^'\"]{12,}", text, re.I)
        no_literal = self.add_check(
            "Cloud secret handling",
            secret_like is None,
            "no literal API credential" if secret_like is None else secret_like.group(0),
        )
        no_deploy_log = self.add_check(
            "Cloud boundary",
            not (self.workspace / "cloud-deployed.json").exists(),
            "no hosted deployment evidence or resource marker exists",
        )
        return schema_ok and no_literal and no_deploy_log

    def _stop_daemon(self) -> None:
        if self.project is None:
            return
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [str(self.pxt), "daemon", "stop", "--force"],
                cwd=self.project,
                env=self.environment,
                capture_output=True,
                timeout=20,
                check=False,
            )

    def verify(self) -> dict[str, Any]:
        required_ok = self._required_files()
        self._scan_source()
        semantic_ok = self._semantic_source_check()
        completion = 2 if required_ok and semantic_ok and self.source_valid else 1 if required_ok else 0
        runtime_ok = False
        with tempfile.TemporaryDirectory(prefix=f"pxt-eval-{self.scenario['id']}-") as temporary:
            self.project = Path(temporary) / "project"
            self.project.mkdir(parents=True)
            catalog = Path(temporary) / "catalog"
            self.environment = os.environ.copy()
            self.environment.update(
                {
                    "PIXELTABLE_HOME": str(catalog),
                    "PIXELTABLE_DISABLE_STDOUT": "1",
                    "PIXELTABLE_MCP_PROJECT_ROOT": str(self.project),
                }
            )
            self.environment.pop("PIXELTABLE_MCP_ENABLE_UNSAFE", None)
            copied = self._copy_workspace(self.project)
            self._stop_daemon()
            try:
                if copied and self._initialize():
                    handler: Callable[[], bool] = getattr(self, f"_verify_{self.scenario['verification']}")
                    runtime_ok = handler()
            finally:
                self._stop_daemon()
        executable = (
            2
            if runtime_ok
            else 1
            if any(check.passed and check.name.startswith("command") for check in self.checks)
            else 0
        )
        recovery = None
        if self.scenario["recovery_applicable"]:
            recovery = 2 if runtime_ok else 1 if semantic_ok else 0
        critical = completion == 0 or executable == 0 or self.unsupported_api_count > 0
        return {
            "schema_version": 1,
            "scenario": self.scenario["id"],
            "task_completion": completion,
            "executable_correctness": executable,
            "unsupported_api_count": self.unsupported_api_count,
            "unnecessary_dependency_count": self.unnecessary_dependency_count,
            "recovery": recovery,
            "critical_defect": critical,
            "passed": completion == 2 and executable == 2 and not critical,
            "checks": [asdict(check) for check in self.checks],
            "commands": [asdict(command) for command in self.commands],
            "boundaries": self.scenario["live_boundaries"],
        }


def find_pxt(python: Path) -> Path:
    sibling = python.parent / "pxt"
    if sibling.is_file():
        return sibling
    executable = shutil.which("pxt")
    if executable is None:
        raise FileNotFoundError("Cannot find pxt next to Python or on PATH")
    return Path(executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pxt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pxt = args.pxt.resolve() if args.pxt else find_pxt(args.python.absolute())
        verifier = TrialVerifier(
            scenario_id=args.scenario,
            workspace=args.workspace,
            python=args.python.absolute(),
            pxt=pxt,
        )
        report = verifier.verify()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (KeyError, OSError, StopIteration, subprocess.SubprocessError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("scenario", "passed", "task_completion", "executable_correctness")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
