#!/usr/bin/env python3
"""Validate the 48-trial matrix and enforce the release parity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX = EVAL_ROOT / "trials.csv"
SCENARIOS_PATH = EVAL_ROOT / "scenarios.json"
SOURCE_LOCK_PATH = EVAL_ROOT / "sources.lock.json"

COLUMNS = [
    "trial_id",
    "scenario",
    "condition",
    "repetition",
    "status",
    "model",
    "settings_sha256",
    "prompt_sha256",
    "guidance_sha256",
    "task_completion",
    "executable_correctness",
    "first_attempt_success",
    "unsupported_api_count",
    "unnecessary_dependency_count",
    "recovery",
    "tool_calls",
    "elapsed_seconds",
    "critical_defect",
    "evidence",
]
RUN_STATUSES = {"passed", "failed"}
ALL_STATUSES = RUN_STATUSES | {"not_run", "error"}
EMPTY_WHEN_NOT_RUN = set(COLUMNS) - {"trial_id", "scenario", "condition", "repetition", "status"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _int_field(row: dict[str, str], field: str, minimum: int, maximum: int | None = None) -> int:
    try:
        value = int(row[field])
    except ValueError as exc:
        raise ValueError(f"{row['trial_id']}: {field} must be an integer") from exc
    if value < minimum or maximum is not None and value > maximum:
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{row['trial_id']}: {field} must be {minimum}{suffix}")
    return value


def _float_field(row: dict[str, str], field: str, minimum: float) -> float:
    try:
        value = float(row[field])
    except ValueError as exc:
        raise ValueError(f"{row['trial_id']}: {field} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{row['trial_id']}: {field} must be finite and at least {minimum}")
    return value


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


def validate_structure(matrix_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    scenarios = _load_json(SCENARIOS_PATH)
    source_lock = _load_json(SOURCE_LOCK_PATH)
    scenario_entries = scenarios.get("scenarios", [])
    condition_entries = scenarios.get("conditions", [])
    if len(scenario_entries) != 8:
        errors.append(f"scenarios.json must define 8 scenarios, found {len(scenario_entries)}")
    if len(condition_entries) != 3:
        errors.append(f"scenarios.json must define 3 conditions, found {len(condition_entries)}")
    if scenarios.get("repetitions") != 2:
        errors.append("scenarios.json repetitions must equal 2")

    scenario_ids = [str(entry.get("id", "")) for entry in scenario_entries]
    condition_ids = [str(entry.get("id", "")) for entry in condition_entries]
    if len(scenario_ids) != len(set(scenario_ids)) or "" in scenario_ids:
        errors.append("scenario ids must be unique and non-empty")
    if len(condition_ids) != len(set(condition_ids)) or "" in condition_ids:
        errors.append("condition ids must be unique and non-empty")
    if set(condition_ids) != {"mcp_skill", "skill_only", "website_docs"}:
        errors.append("condition ids must be mcp_skill, skill_only, and website_docs")

    for scenario in scenario_entries:
        prompt_file = EVAL_ROOT / str(scenario.get("prompt_file", ""))
        if not prompt_file.is_file():
            errors.append(f"missing prompt file: {prompt_file.relative_to(EVAL_ROOT)}")
    for source in source_lock.get("sources", []):
        digest = str(source.get("sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"invalid source checksum for {source.get('id')}")
    source_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in source_lock.get("sources", []):
        source_records[str(source.get("condition", ""))].append(
            {"path": str(source.get("path", "")), "sha256": str(source.get("sha256", ""))}
        )
    expected_guidance = {
        str(condition.get("id", "")): _canonical_hash(
            sorted(source_records[str(condition.get("guidance", ""))], key=lambda record: record["path"])
        )
        for condition in condition_entries
    }

    try:
        with matrix_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if reader.fieldnames != COLUMNS:
                errors.append(f"CSV columns differ from the required schema: {reader.fieldnames}")
    except OSError as exc:
        return [], [f"cannot read trial matrix: {exc}"]

    expected = {
        (scenario_id, condition_id, repetition)
        for scenario_id in scenario_ids
        for condition_id in condition_ids
        for repetition in (1, 2)
    }
    observed: set[tuple[str, str, int]] = set()
    seen_trial_ids: set[str] = set()
    completed_models: set[str] = set()
    completed_settings: set[str] = set()
    guidance_by_condition: dict[str, set[str]] = defaultdict(set)
    prompt_by_scenario: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        trial_id = row.get("trial_id", "")
        try:
            repetition = int(row.get("repetition", ""))
        except ValueError:
            errors.append(f"{trial_id or '<unknown>'}: repetition must be 1 or 2")
            continue
        key = (row.get("scenario", ""), row.get("condition", ""), repetition)
        expected_id = f"{key[0]}__{key[1]}__r{repetition}"
        if trial_id != expected_id:
            errors.append(f"{trial_id or '<empty>'}: expected trial_id {expected_id}")
        if trial_id in seen_trial_ids:
            errors.append(f"duplicate trial_id: {trial_id}")
        seen_trial_ids.add(trial_id)
        if key in observed:
            errors.append(f"duplicate trial combination: {key}")
        observed.add(key)
        status = row.get("status", "")
        if status not in ALL_STATUSES:
            errors.append(f"{trial_id}: invalid status {status!r}")
            continue
        if status == "not_run":
            populated = sorted(field for field in EMPTY_WHEN_NOT_RUN if row.get(field, ""))
            if populated:
                errors.append(f"{trial_id}: not_run row has populated fields: {', '.join(populated)}")
            continue
        evidence_document: dict[str, Any] | None = None
        evidence_value = row.get("evidence", "")
        if evidence_value:
            evidence = (matrix_path.parent / evidence_value).resolve()
            try:
                evidence.relative_to(EVAL_ROOT.resolve())
            except ValueError:
                errors.append(f"{trial_id}: evidence path escapes evals/: {evidence_value}")
            else:
                if not evidence.is_file():
                    errors.append(f"{trial_id}: evidence file does not exist: {evidence}")
                else:
                    try:
                        raw_evidence = json.loads(evidence.read_text())
                        if isinstance(raw_evidence, dict):
                            evidence_document = raw_evidence
                        else:
                            errors.append(f"{trial_id}: evidence must be a JSON object")
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(f"{trial_id}: cannot parse evidence: {exc}")
        if status == "error":
            if not row.get("model") or not row.get("settings_sha256") or not row.get("evidence"):
                errors.append(f"{trial_id}: error row must retain model, settings checksum, and evidence")
            continue
        try:
            _int_field(row, "task_completion", 0, 2)
            _int_field(row, "executable_correctness", 0, 2)
            _int_field(row, "first_attempt_success", 0, 1)
            _int_field(row, "unsupported_api_count", 0)
            _int_field(row, "unnecessary_dependency_count", 0)
            _int_field(row, "tool_calls", 0)
            _float_field(row, "elapsed_seconds", 0.0)
            scenario = next((entry for entry in scenario_entries if entry.get("id") == row["scenario"]), {})
            if scenario.get("recovery_applicable"):
                _int_field(row, "recovery", 0, 2)
            elif row.get("recovery"):
                errors.append(f"{trial_id}: recovery must be blank when the scenario does not exercise recovery")
        except ValueError as exc:
            errors.append(str(exc))
        if row.get("critical_defect") not in {"true", "false"}:
            errors.append(f"{trial_id}: critical_defect must be true or false")
        for field in ("model", "settings_sha256", "prompt_sha256", "guidance_sha256", "evidence"):
            if not row.get(field):
                errors.append(f"{trial_id}: completed row is missing {field}")
        completed_models.add(row.get("model", ""))
        completed_settings.add(row.get("settings_sha256", ""))
        guidance_by_condition[row["condition"]].add(row.get("guidance_sha256", ""))
        prompt_by_scenario[row["scenario"]].add(row.get("prompt_sha256", ""))
        if row.get("guidance_sha256") != expected_guidance.get(row["condition"]):
            errors.append(f"{trial_id}: guidance checksum does not match sources.lock.json")
        if evidence_document is not None:
            for field, expected_value in (
                ("trial_id", trial_id),
                ("scenario", row["scenario"]),
                ("condition", row["condition"]),
                ("repetition", int(row["repetition"])),
                ("settings_sha256", row["settings_sha256"]),
                ("prompt_sha256", row["prompt_sha256"]),
                ("guidance_sha256", row["guidance_sha256"]),
                ("tool_calls", int(row["tool_calls"])),
            ):
                if evidence_document.get(field) != expected_value:
                    errors.append(f"{trial_id}: evidence {field} does not match the matrix")
            settings = evidence_document.get("settings")
            if not isinstance(settings, dict) or _canonical_hash(settings) != row["settings_sha256"]:
                errors.append(f"{trial_id}: evidence settings checksum is invalid")
            for field in ("transcript_sha256", "source_manifest_sha256"):
                digest = evidence_document.get(field)
                if not isinstance(digest, str) or len(digest) != 64:
                    errors.append(f"{trial_id}: evidence {field} is not a SHA-256 digest")
            if evidence_document.get("codex_returncode") != 0 or evidence_document.get("codex_timed_out") is not False:
                errors.append(f"{trial_id}: completed trial has an unsuccessful Codex execution")
            verifier = evidence_document.get("verifier")
            if not isinstance(verifier, dict):
                errors.append(f"{trial_id}: evidence has no verifier result")
            else:
                for field in (
                    "task_completion",
                    "executable_correctness",
                    "unsupported_api_count",
                    "unnecessary_dependency_count",
                ):
                    if str(verifier.get(field)) != row[field]:
                        errors.append(f"{trial_id}: verifier {field} does not match the matrix")
                expected_recovery = "" if verifier.get("recovery") is None else str(verifier.get("recovery"))
                if expected_recovery != row["recovery"]:
                    errors.append(f"{trial_id}: verifier recovery does not match the matrix")
                expected_critical = bool(
                    verifier.get("critical_defect") or evidence_document.get("tool_limit_exceeded")
                )
                if str(expected_critical).lower() != row["critical_defect"]:
                    errors.append(f"{trial_id}: verifier critical_defect does not match the matrix")
                expected_status = (
                    "passed"
                    if verifier.get("passed") is True and evidence_document.get("tool_limit_exceeded") is False
                    else "failed"
                )
                if row["status"] != expected_status:
                    errors.append(f"{trial_id}: status does not match the verifier result")
            try:
                evidence_elapsed = float(evidence_document.get("elapsed_seconds"))
                if abs(evidence_elapsed - float(row["elapsed_seconds"])) > 0.01:
                    errors.append(f"{trial_id}: elapsed_seconds does not match evidence")
            except (TypeError, ValueError):
                errors.append(f"{trial_id}: evidence elapsed_seconds is invalid")

    missing = expected - observed
    extra = observed - expected
    if missing:
        errors.append(f"missing trial combinations: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected trial combinations: {sorted(extra)}")
    if len(rows) != 48:
        errors.append(f"trial matrix must contain 48 rows, found {len(rows)}")
    if len(completed_models) > 1:
        errors.append(f"completed trials use different models: {sorted(completed_models)}")
    if len(completed_settings) > 1:
        errors.append("completed trials use different settings")
    for condition, checksums in guidance_by_condition.items():
        if len(checksums) > 1:
            errors.append(f"condition {condition} used more than one guidance snapshot")
    for scenario, checksums in prompt_by_scenario.items():
        if len(checksums) > 1:
            errors.append(f"scenario {scenario} used more than one prompt")
        prompt = next((entry for entry in scenario_entries if entry.get("id") == scenario), None)
        if prompt is not None and checksums and checksums != {_sha256(EVAL_ROOT / prompt["prompt_file"])}:
            errors.append(f"scenario {scenario} prompt checksum does not match the committed prompt")
    return rows, errors


def validate_release_gate(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    not_completed = [row["trial_id"] for row in rows if row["status"] not in RUN_STATUSES]
    if not_completed:
        return [f"release gate requires all 48 trials; incomplete: {', '.join(not_completed)}"]
    critical = [row["trial_id"] for row in rows if row["critical_defect"] == "true"]
    if critical:
        errors.append(f"unresolved critical defects: {', '.join(critical)}")

    scores: dict[tuple[str, str, int], float] = {
        (row["scenario"], row["condition"], int(row["repetition"])): _quality(row) for row in rows
    }
    by_condition: dict[str, list[float]] = defaultdict(list)
    for (_, condition, _), score in scores.items():
        by_condition[condition].append(score)
    mcp_mean = sum(by_condition["mcp_skill"]) / len(by_condition["mcp_skill"])
    for baseline in ("skill_only", "website_docs"):
        baseline_mean = sum(by_condition[baseline]) / len(by_condition[baseline])
        if mcp_mean < baseline_mean:
            errors.append(
                f"overall parity failed: mcp_skill mean {mcp_mean:.3f} is below {baseline} mean {baseline_mean:.3f}"
            )

    scenario_ids = sorted({row["scenario"] for row in rows})
    for scenario in scenario_ids:
        for baseline in ("skill_only", "website_docs"):
            comparisons = [
                scores[(scenario, "mcp_skill", repetition)] < scores[(scenario, baseline, repetition)]
                for repetition in (1, 2)
            ]
            if all(comparisons):
                errors.append(f"repeated scenario regression: {scenario} is below {baseline} in both repetitions")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_MATRIX, help="Trial CSV to validate.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all trials and enforce overall parity, repeated-regression, and critical-defect gates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows, errors = validate_structure(args.results.resolve())
        if args.require_complete and not errors:
            errors.extend(validate_release_gate(rows))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    qualifier = "complete release-gate" if args.require_complete else "structural"
    print(f"OK: {qualifier} validation passed for {args.results.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
