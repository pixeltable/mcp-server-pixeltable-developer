"""
Pixeltable MCP Server.

This module provides an MCP server implementation that exposes Pixeltable functionality
using all three MCP primitives: Tools, Resources, and Prompts.
"""

import os
import sys
import json
import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# Import utilities
from mcp_server_pixeltable_stio.utils import setup_resilient_process

# ---------------------------------------------------------------------------
# Import core functionality from specialised modules
# ---------------------------------------------------------------------------

# Table management (tables.py)
from mcp_server_pixeltable_stio.core.tables import (
    pixeltable_create_table,
    pixeltable_drop_table,
    pixeltable_create_view,
    pixeltable_create_snapshot,
    pixeltable_create_replica,
    pixeltable_query_table,
    pixeltable_insert_data,
    pixeltable_add_computed_column,
    pixeltable_query,
)

# Directory management (directories.py)
from mcp_server_pixeltable_stio.core.directories import (
    pixeltable_create_dir,
    pixeltable_drop_dir,
    pixeltable_move,
)

# Dependency management (dependencies.py)
from mcp_server_pixeltable_stio.core.dependencies import (
    pixeltable_check_dependencies,
    pixeltable_install_dependency,
)

# UDF / type / tools / MCP connection (udf.py)
from mcp_server_pixeltable_stio.core.udf import (
    pixeltable_create_udf,
    pixeltable_create_array,
    pixeltable_create_tools,
    pixeltable_connect_mcp,
    pixeltable_create_type,
)

# Configuration & utility helpers (helpers.py)
from mcp_server_pixeltable_stio.core.helpers import (
    pixeltable_configure_logging,
    pixeltable_set_datastore,
    pixeltable_search_docs,
)

# MCP resource handlers (resources.py)
from mcp_server_pixeltable_stio.core.resources import (
    resource_list_tables,
    resource_get_table_schema,
    resource_get_table,
    resource_list_dirs,
    resource_ls,
    resource_get_version,
    resource_get_datastore,
    resource_get_types,
    resource_list_functions,
    resource_list_tools,
    resource_get_help,
    resource_system_diagnostics,
)

# REPL and bug logging functions
from mcp_server_pixeltable_stio.core.repl_functions import (
    execute_python,
    introspect_function,
    list_available_functions,
    install_package,
    log_bug,
    log_missing_feature,
    log_success,
    generate_bug_report,
    get_session_summary
)

# Canvas server functions
from mcp_server_pixeltable_stio.core.canvas_server import (
    run_canvas_server_thread,
    broadcast_to_canvas
)

# Prompt helpers
from mcp_server_pixeltable_stio.prompt import (
    PIXELTABLE_USAGE_PROMPT,
    GETTING_STARTED_PROMPT,
    COMPUTER_VISION_PROMPT,
    RAG_PIPELINE_PROMPT,
    VIDEO_ANALYSIS_PROMPT,
    AUDIO_PROCESSING_PROMPT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Create the MCP server instance at MODULE LEVEL so `mcp dev` can find it
# ===========================================================================
mcp = FastMCP(name="pixeltable-developer")


# ===================
# MCP RESOURCES (read-only data exposed via URIs)
# ===================

@mcp.resource("pixeltable://tables")
def tables_resource() -> str:
    """List all Pixeltable tables with count."""
    return resource_list_tables()

@mcp.resource("pixeltable://tables/{path}/schema")
def table_schema_resource(path: str) -> str:
    """Get the schema of a Pixeltable table."""
    return resource_get_table_schema(path)

@mcp.resource("pixeltable://tables/{path}")
def table_resource(path: str) -> str:
    """Get info about a specific Pixeltable table, view, or snapshot."""
    return resource_get_table(path)

@mcp.resource("pixeltable://directories")
def directories_resource() -> str:
    """List all Pixeltable directories."""
    return resource_list_dirs()

@mcp.resource("pixeltable://ls")
def ls_root_resource() -> str:
    """List contents of the Pixeltable root directory."""
    return resource_ls('')

@mcp.resource("pixeltable://ls/{path}")
def ls_resource(path: str) -> str:
    """List contents of a Pixeltable directory."""
    return resource_ls(path)

@mcp.resource("pixeltable://version")
def version_resource() -> str:
    """Get Pixeltable version information."""
    return resource_get_version()

@mcp.resource("pixeltable://config/datastore")
def datastore_resource() -> str:
    """Get the current Pixeltable datastore configuration."""
    return resource_get_datastore()

@mcp.resource("pixeltable://types")
def types_resource() -> str:
    """Get available Pixeltable data types."""
    return resource_get_types()

@mcp.resource("pixeltable://functions")
def functions_resource() -> str:
    """List all registered Pixeltable functions."""
    return resource_list_functions()

@mcp.resource("pixeltable://tools")
def tools_resource() -> str:
    """List all available MCP tools with descriptions."""
    return resource_list_tools()

@mcp.resource("pixeltable://help")
def help_resource() -> str:
    """Get comprehensive Pixeltable help and workflow guidance."""
    return resource_get_help()

@mcp.resource("pixeltable://diagnostics")
def diagnostics_resource() -> str:
    """Get system diagnostics for Pixeltable and its dependencies."""
    return resource_system_diagnostics()


# ===================
# MCP TOOLS (action-oriented operations)
# ===================

# Core table management (pixeltable_init removed -- ensure_pixeltable_available() runs automatically)
mcp.tool()(pixeltable_create_table)
mcp.tool()(pixeltable_drop_table)
mcp.tool()(pixeltable_create_view)
mcp.tool()(pixeltable_create_snapshot)

# Directory management
mcp.tool()(pixeltable_create_dir)
mcp.tool()(pixeltable_drop_dir)
mcp.tool()(pixeltable_move)

# Configuration
mcp.tool()(pixeltable_configure_logging)

# Data operations
mcp.tool()(pixeltable_create_replica)
mcp.tool()(pixeltable_query_table)
mcp.tool()(pixeltable_insert_data)
mcp.tool()(pixeltable_add_computed_column)

# Dependency management (unified: replaces 7 individual tools)
mcp.tool()(pixeltable_check_dependencies)
mcp.tool()(pixeltable_install_dependency)

# High-priority functions
mcp.tool()(pixeltable_query)
mcp.tool()(pixeltable_create_udf)
mcp.tool()(pixeltable_create_array)
mcp.tool()(pixeltable_create_tools)
mcp.tool()(pixeltable_connect_mcp)

# Type helper (unified: replaces 5 individual type creators)
mcp.tool()(pixeltable_create_type)

# Datastore and docs
mcp.tool()(pixeltable_set_datastore)
mcp.tool()(pixeltable_search_docs)

# REPL and interactive functions
mcp.tool()(execute_python)
mcp.tool()(introspect_function)
mcp.tool()(list_available_functions)
mcp.tool()(install_package)


# Canvas display tool
@mcp.tool()
def display_in_browser(content_type: str, data: Any, title: str = None) -> Dict[str, Any]:
    """Send content to browser canvas for display.

    Args:
        content_type: Type of content ('image', 'text', 'html', 'table', etc.)
        data: The content data to display
            - For 'image': base64 data URL or image URL
            - For 'text': plain text string
            - For 'html': HTML string
            - For 'table': list of dictionaries (rows)
        title: Optional title to display above the content

    Returns:
        Success status

    Example:
        display_in_browser('image', 'data:image/png;base64,...')
        display_in_browser('text', 'Hello from Claude!')
        display_in_browser('table', [{'name': 'Alice', 'age': 30}], title='User Data')
        display_in_browser('mermaid', 'graph TD...', title='Schema DAG')
    """
    try:
        # Convert file:// URLs to /media/ URLs for serving
        processed_data = data
        if isinstance(data, str) and data.startswith('file://'):
            file_path = data.replace('file://', '')
            processed_data = f'http://localhost:7777/media{file_path}'
        elif isinstance(data, list):
            processed_data = []
            for item in data:
                if isinstance(item, dict) and 'url' in item:
                    if item['url'].startswith('file://'):
                        file_path = item['url'].replace('file://', '')
                        item['url'] = f'http://localhost:7777/media{file_path}'
                processed_data.append(item)

        message = {
            'content_type': content_type,
            'data': processed_data
        }
        if title:
            message['title'] = title

        broadcast_to_canvas(message)
        return {"success": True, "message": f"Displayed {content_type} in browser"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Bug logging functions
mcp.tool()(log_bug)
mcp.tool()(log_missing_feature)
mcp.tool()(log_success)
mcp.tool()(generate_bug_report)
mcp.tool()(get_session_summary)


# ===================
# MCP PROMPTS
# ===================

@mcp.prompt()
def pixeltable_usage_guide() -> str:
    """A comprehensive guide on how to effectively use the Pixeltable MCP server for multimodal AI data workflows."""
    return PIXELTABLE_USAGE_PROMPT

@mcp.prompt()
def getting_started() -> str:
    """Step-by-step guide for first-time Pixeltable users: create a directory, table, insert data, query, and add AI columns."""
    return GETTING_STARTED_PROMPT

@mcp.prompt()
def computer_vision_pipeline() -> str:
    """Build a computer vision pipeline: image ingestion, YOLOX object detection, GPT-4 Vision descriptions, and metadata extraction."""
    return COMPUTER_VISION_PROMPT

@mcp.prompt()
def rag_pipeline() -> str:
    """Build a RAG pipeline: document ingestion, chunking with DocumentSplitter, embedding generation, and similarity search."""
    return RAG_PIPELINE_PROMPT

@mcp.prompt()
def video_analysis_pipeline() -> str:
    """Build a video analysis pipeline: frame extraction, per-frame AI analysis, audio transcription, and temporal queries."""
    return VIDEO_ANALYSIS_PROMPT

@mcp.prompt()
def audio_processing_pipeline() -> str:
    """Build an audio processing pipeline: transcription with Whisper, LLM-based analysis, and semantic search over spoken content."""
    return AUDIO_PROCESSING_PROMPT


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    """Start the Pixeltable MCP server."""
    logger.info("Pixeltable MCP server initializing")

    try:
        # Disable Pixeltable console output to prevent JSON parsing issues
        os.environ['PIXELTABLE_DISABLE_STDOUT'] = '1'

        # Start canvas server in background thread
        run_canvas_server_thread(port=7777)
        logger.info("Canvas server started on http://localhost:7777/canvas")

        logger.info("Pixeltable MCP server ready")

    except Exception as e:
        logger.error(f"Failed during startup: {e}")

    # Start the server
    logger.info("Pixeltable MCP server starting...")
    mcp.run()


if __name__ == "__main__":
    main()
