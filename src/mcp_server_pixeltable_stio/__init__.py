"""One-release compatibility shim for the package name misspelled before 0.2."""

from __future__ import annotations

import sys

print(
    "mcp_server_pixeltable_stio is deprecated; import mcp_server_pixeltable_developer instead.",
    file=sys.stderr,
)

from mcp_server_pixeltable_developer import __version__  # noqa: E402

__all__ = ["__version__"]
