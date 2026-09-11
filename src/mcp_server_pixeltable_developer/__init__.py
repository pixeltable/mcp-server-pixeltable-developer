"""Pixeltable Developer MCP server."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-server-pixeltable-developer")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0.2.0"


__all__ = ["__version__"]
