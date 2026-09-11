#!/usr/bin/env python3
"""Materialize one immutable guidance bundle for a complete evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = EVAL_ROOT / "sources.lock.json"
USER_AGENT = "mcp-server-pixeltable-developer-eval/0.2"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source path escapes output directory: {relative!r}") from exc
    return destination


def _download(url: str, timeout: float) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS sources are allowed: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def prepare_sources(
    *,
    lock_path: Path,
    output_dir: Path,
    timeout: float,
    strict_mutable: bool,
    force: bool,
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    sources = lock.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources.lock.json must contain a non-empty sources array")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    drifted: list[str] = []
    condition_files: dict[str, list[str]] = {"skill": [], "website": []}

    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise ValueError("Each source entry must be an object")
        source_id = str(raw_source["id"])
        condition = str(raw_source["condition"])
        relative_path = str(raw_source["path"])
        url = str(raw_source["url"])
        expected_hash = str(raw_source["sha256"])
        expected_bytes = int(raw_source["bytes"])
        immutable = bool(raw_source["immutable"])

        content = _download(url, timeout)
        observed_hash = _sha256(content)
        observed_bytes = len(content)
        changed = observed_hash != expected_hash or observed_bytes != expected_bytes
        if changed:
            drifted.append(source_id)
            if immutable or strict_mutable:
                kind = "immutable" if immutable else "mutable"
                raise RuntimeError(
                    f"{kind} source drift for {source_id}: expected {expected_hash}/{expected_bytes}, "
                    f"observed {observed_hash}/{observed_bytes}"
                )

        destination = _safe_destination(output_dir, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        condition_files.setdefault(condition, []).append(relative_path)
        records.append(
            {
                "id": source_id,
                "condition": condition,
                "path": relative_path,
                "url": url,
                "immutable": immutable,
                "retrieved_at": retrieved_at,
                "bytes": observed_bytes,
                "sha256": observed_hash,
                "locked_bytes": expected_bytes,
                "locked_sha256": expected_hash,
                "drifted": changed,
            }
        )

    run_manifest = {
        "schema_version": 1,
        "retrieved_at": retrieved_at,
        "source_lock": str(lock_path.resolve()),
        "lock_sha256": _sha256(lock_path.read_bytes()),
        "environment": lock.get("environment", {}),
        "drifted_sources": drifted,
        "sources": records,
        "condition_files": condition_files,
    }
    (output_dir / "retrievals.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--strict-mutable",
        action="store_true",
        help="Fail if either mutable website document differs from the recorded review snapshot.",
    )
    parser.add_argument("--force", action="store_true", help="Replace a non-empty output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_sources(
            lock_path=args.lock.resolve(),
            output_dir=args.output_dir.resolve(),
            timeout=args.timeout,
            strict_mutable=args.strict_mutable,
            force=args.force,
        )
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"source preparation failed: {exc}", file=sys.stderr)
        return 1

    drifted = manifest["drifted_sources"]
    if drifted:
        print(f"prepared sources with mutable drift: {', '.join(drifted)}", file=sys.stderr)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "sources": len(manifest["sources"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
