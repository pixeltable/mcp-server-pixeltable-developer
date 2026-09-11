"""Internal fixed-operation subprocess used to isolate Pixeltable's stdout."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from typing import Any


def _insert(payload: dict[str, Any]) -> dict[str, Any]:
    with redirect_stdout(sys.stderr):
        import pixeltable as pxt

        table = pxt.get_table(payload["path"], if_not_exists="ignore")  # pyright: ignore[reportCallIssue]
        if table is None:
            raise ValueError(f"Table does not exist: {payload['path']}")
        rows = payload["rows"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
            raise ValueError("rows must contain between 1 and 1000 objects")
        status = table.insert(rows, on_error=payload["on_error"], print_stats=False)
    return {
        "path": payload["path"],
        "num_rows": status.num_rows,
        "num_computed_values": status.num_computed_values,
        "num_exceptions": status.num_excs,
        "columns_with_exceptions": [str(column) for column in status.cols_with_excs],
        "updated_columns": [str(column) for column in status.updated_cols],
    }


def main() -> None:
    os.environ.setdefault("PIXELTABLE_DISABLE_STDOUT", "1")
    if len(sys.argv) != 2:
        raise SystemExit("Internal worker requires one operation")
    payload = json.load(sys.stdin)
    operation = sys.argv[1]
    try:
        if operation == "insert":
            result = _insert(payload)
        else:
            raise ValueError(f"Unknown internal operation: {operation}")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
