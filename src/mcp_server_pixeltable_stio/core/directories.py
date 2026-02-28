"""
Directory management for the Pixeltable MCP server.

Covers creating, dropping, listing, browsing, and moving Pixeltable
directories (the namespace layer above tables).
"""

import logging
from typing import Any, Dict

from .helpers import pxt, ensure_pixeltable_available

logger = logging.getLogger(__name__)


def pixeltable_create_dir(
    path: str,
    if_exists: str = 'error',
    parents: bool = False
) -> Dict[str, Any]:
    """Create a directory."""
    try:
        ensure_pixeltable_available()
        directory = pxt.create_dir(path, if_exists=if_exists, parents=parents)
        return {
            "success": True,
            "message": f"Directory '{path}' created successfully",
            "directory_path": path
        }
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        raise ValueError(f"Failed to create directory: {e}")


def pixeltable_drop_dir(
    path: str,
    force: bool = False,
    if_not_exists: str = 'error'
) -> Dict[str, Any]:
    """Remove a directory."""
    try:
        ensure_pixeltable_available()
        pxt.drop_dir(path, force=force, if_not_exists=if_not_exists)
        return {
            "success": True,
            "message": f"Directory '{path}' dropped successfully"
        }
    except Exception as e:
        logger.error(f"Error dropping directory: {e}")
        raise ValueError(f"Failed to drop directory: {e}")


def pixeltable_list_dirs(path: str = '', recursive: bool = True) -> Dict[str, Any]:
    """List directories in a directory."""
    try:
        ensure_pixeltable_available()
        directories = pxt.list_dirs(path, recursive=recursive)
        return {
            "success": True,
            "directories": directories,
            "count": len(directories)
        }
    except Exception as e:
        logger.error(f"Error listing directories: {e}")
        raise ValueError(f"Failed to list directories: {e}")


def pixeltable_ls(path: str = '') -> Dict[str, Any]:
    """List the contents of a Pixeltable directory."""
    try:
        ensure_pixeltable_available()
        df = pxt.ls(path)
        result = df.to_dict('records')
        return {
            "success": True,
            "contents": result,
            "count": len(result)
        }
    except Exception as e:
        logger.error(f"Error listing directory contents: {e}")
        raise ValueError(f"Failed to list directory contents: {e}")


def pixeltable_move(path: str, new_path: str) -> Dict[str, Any]:
    """Move a schema object to a new directory and/or rename it."""
    try:
        ensure_pixeltable_available()
        pxt.move(path, new_path)
        return {
            "success": True,
            "message": f"Moved '{path}' to '{new_path}'"
        }
    except Exception as e:
        logger.error(f"Error moving object: {e}")
        raise ValueError(f"Failed to move object: {e}")
