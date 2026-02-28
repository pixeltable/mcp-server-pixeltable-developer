#!/usr/bin/env python3
"""
Quick script to list all registered tools, resources, and prompts in the Pixeltable MCP server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ---------------------------------------------------------------------------
# Import tool functions from their canonical modules
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

# Dependencies (dependencies.py)
from mcp_server_pixeltable_stio.core.dependencies import (
    pixeltable_check_dependencies,
    pixeltable_install_dependency,
)

# UDF / type / tools / MCP (udf.py)
from mcp_server_pixeltable_stio.core.udf import (
    pixeltable_create_udf,
    pixeltable_create_array,
    pixeltable_create_tools,
    pixeltable_connect_mcp,
    pixeltable_create_type,
)

# Helpers (helpers.py)
from mcp_server_pixeltable_stio.core.helpers import (
    pixeltable_configure_logging,
    pixeltable_set_datastore,
    pixeltable_search_docs,
)

# REPL & bug logging
from mcp_server_pixeltable_stio.core.repl_functions import (
    execute_python,
    introspect_function,
    list_available_functions,
    install_package,
    log_bug,
    log_missing_feature,
    log_success,
    generate_bug_report,
    get_session_summary,
)

# Resource URIs registered in server.py
RESOURCES = [
    ("pixeltable://tables", "List all Pixeltable tables with count"),
    ("pixeltable://tables/{path}/schema", "Get the schema of a Pixeltable table"),
    ("pixeltable://tables/{path}", "Get info about a specific table, view, or snapshot"),
    ("pixeltable://directories", "List all Pixeltable directories"),
    ("pixeltable://ls", "List contents of the Pixeltable root directory"),
    ("pixeltable://ls/{path}", "List contents of a Pixeltable directory"),
    ("pixeltable://version", "Get Pixeltable version information"),
    ("pixeltable://config/datastore", "Get the current datastore configuration"),
    ("pixeltable://types", "Get available Pixeltable data types"),
    ("pixeltable://functions", "List all registered Pixeltable functions"),
    ("pixeltable://tools", "List all available MCP tools with descriptions"),
    ("pixeltable://help", "Get comprehensive Pixeltable help and workflow guidance"),
    ("pixeltable://diagnostics", "Get system diagnostics for Pixeltable and dependencies"),
]

# Prompt names registered in server.py
PROMPTS = [
    ("pixeltable_usage_guide", "Comprehensive guide for multimodal AI data workflows"),
    ("getting_started", "Step-by-step guide for first-time users"),
    ("computer_vision_pipeline", "Build a CV pipeline with YOLOX / GPT-4 Vision"),
    ("rag_pipeline", "Build a RAG pipeline with chunking, embeddings, search"),
    ("video_analysis_pipeline", "Build a video analysis pipeline with frame extraction"),
    ("audio_processing_pipeline", "Build an audio processing pipeline with Whisper"),
]


def list_all():
    """List all available tools, resources, and prompts."""

    tools = [
        # Core table management
        pixeltable_create_table,
        pixeltable_drop_table,
        pixeltable_create_view,
        pixeltable_create_snapshot,
        # Directory management
        pixeltable_create_dir,
        pixeltable_drop_dir,
        pixeltable_move,
        # Configuration
        pixeltable_configure_logging,
        # Data operations
        pixeltable_create_replica,
        pixeltable_query_table,
        pixeltable_insert_data,
        pixeltable_add_computed_column,
        # Dependencies (unified)
        pixeltable_check_dependencies,
        pixeltable_install_dependency,
        # High-priority
        pixeltable_query,
        pixeltable_create_udf,
        pixeltable_create_array,
        pixeltable_create_tools,
        pixeltable_connect_mcp,
        # Type helper (unified)
        pixeltable_create_type,
        # Datastore & docs
        pixeltable_set_datastore,
        pixeltable_search_docs,
        # REPL & debug
        execute_python,
        introspect_function,
        list_available_functions,
        install_package,
        # Bug logging
        log_bug,
        log_missing_feature,
        log_success,
        generate_bug_report,
        get_session_summary,
    ]
    # NOTE: display_in_browser is defined inline in server.py and not listed here

    # ---- TOOLS ----
    print("=" * 60)
    print("  Pixeltable MCP Server — Registered Primitives")
    print("=" * 60)

    categories = {
        "Table Management": ["create_table", "drop_table", "create_view", "create_snapshot"],
        "Data Operations": ["query_table", "query", "insert_data", "add_computed_column", "create_replica"],
        "Directory Management": ["create_dir", "drop_dir", "move"],
        "Configuration": ["configure_logging", "set_datastore"],
        "AI/ML Integration": ["create_udf", "create_array", "create_tools", "connect_mcp"],
        "Dependencies": ["check_dependencies", "install_dependency"],
        "Type Helper": ["create_type"],
        "Documentation": ["search_docs"],
        "REPL & Debug": ["execute_python", "introspect_function", "list_available_functions", "install_package"],
        "Bug Logging": ["log_bug", "log_missing_feature", "log_success", "generate_bug_report", "get_session_summary"],
    }

    categorized = {cat: [] for cat in categories}
    uncategorized = []

    for tool in tools:
        name = tool.__name__
        doc = (tool.__doc__ or "No description").strip().split('\n')[0]
        found = False
        for cat, keywords in categories.items():
            if any(name.endswith(kw) or name == kw or f"pixeltable_{kw}" == name for kw in keywords):
                categorized[cat].append((name, doc))
                found = True
                break
        if not found:
            uncategorized.append((name, doc))

    print(f"\n--- TOOLS ({len(tools)} + 1 inline) ---\n")
    for category, tool_list in categorized.items():
        if tool_list:
            print(f"  {category}:")
            for name, doc in tool_list:
                print(f"    • {name}: {doc}")
    if uncategorized:
        print(f"  Other:")
        for name, doc in uncategorized:
            print(f"    • {name}: {doc}")

    # ---- RESOURCES ----
    print(f"\n--- RESOURCES ({len(RESOURCES)}) ---\n")
    for uri, desc in RESOURCES:
        print(f"  • {uri}: {desc}")

    # ---- PROMPTS ----
    print(f"\n--- PROMPTS ({len(PROMPTS)}) ---\n")
    for name, desc in PROMPTS:
        print(f"  • {name}: {desc}")

    print(f"\n{'=' * 60}")
    print(f"  Total: {len(tools) + 1} tools, {len(RESOURCES)} resources, {len(PROMPTS)} prompts")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    list_all()
