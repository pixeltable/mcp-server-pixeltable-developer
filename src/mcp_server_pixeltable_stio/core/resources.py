"""
MCP Resource handlers for the Pixeltable MCP server.

Resources expose read-only data via URIs, reducing tool count and token usage.
Each function returns a JSON string suitable for MCP resource responses.

Imports point to the specific modules where each function lives.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _to_json(data: Any) -> str:
    """Convert data to a JSON string, handling non-serializable types."""
    return json.dumps(data, default=str, indent=2)


# ---------------------------------------------------------------------------
# Table resources
# ---------------------------------------------------------------------------

def resource_list_tables() -> str:
    """List all Pixeltable tables with count."""
    from mcp_server_pixeltable_stio.core.tables import pixeltable_list_tables
    return _to_json(pixeltable_list_tables())


def resource_get_table_schema(path: str) -> str:
    """Get the schema of a specific Pixeltable table."""
    from mcp_server_pixeltable_stio.core.tables import pixeltable_get_table_schema
    return _to_json(pixeltable_get_table_schema(path))


def resource_get_table(path: str) -> str:
    """Get information about a specific Pixeltable table, view, or snapshot."""
    from mcp_server_pixeltable_stio.core.tables import pixeltable_get_table
    return _to_json(pixeltable_get_table(path))


# ---------------------------------------------------------------------------
# Directory resources
# ---------------------------------------------------------------------------

def resource_list_dirs() -> str:
    """List all Pixeltable directories."""
    from mcp_server_pixeltable_stio.core.directories import pixeltable_list_dirs
    return _to_json(pixeltable_list_dirs())


def resource_ls(path: str = '') -> str:
    """List contents of a Pixeltable directory."""
    from mcp_server_pixeltable_stio.core.directories import pixeltable_ls
    return _to_json(pixeltable_ls(path))


# ---------------------------------------------------------------------------
# Config / utility resources
# ---------------------------------------------------------------------------

def resource_get_version() -> str:
    """Get Pixeltable version information."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_get_version
    return _to_json(pixeltable_get_version())


def resource_get_datastore() -> str:
    """Get the current Pixeltable datastore configuration."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_get_datastore
    return _to_json(pixeltable_get_datastore())


def resource_get_types() -> str:
    """Get available Pixeltable data types."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_get_types
    return _to_json(pixeltable_get_types())


def resource_list_functions() -> str:
    """List all registered Pixeltable functions."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_list_functions
    return _to_json(pixeltable_list_functions())


def resource_list_tools() -> str:
    """List all available Pixeltable MCP tools with categories."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_list_tools
    return _to_json(pixeltable_list_tools())


def resource_get_help() -> str:
    """Get comprehensive Pixeltable help, concepts, and workflow guidance."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_get_help
    return _to_json(pixeltable_get_help())


# ---------------------------------------------------------------------------
# Diagnostics resource
# ---------------------------------------------------------------------------

def resource_system_diagnostics() -> str:
    """Get system diagnostics for Pixeltable and its dependencies."""
    from mcp_server_pixeltable_stio.core.dependencies import pixeltable_system_diagnostics
    return _to_json(pixeltable_system_diagnostics())
