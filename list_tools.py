#!/usr/bin/env python3
"""
Quick script to list all registered tools in the Pixeltable MCP server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server_pixeltable_stio.server import main
from mcp.server.fastmcp import FastMCP

# Import all the tool functions to get their names and docstrings
from mcp_server_pixeltable_stio.core.pixeltable_functions import *
from mcp_server_pixeltable_stio.core.repl_functions import *

def list_tools():
    """List all available tools in the Pixeltable MCP server."""
    
    # Create a temporary MCP instance to access registered tools
    mcp = FastMCP(name="pixeltable-developer")
    
    # Register all tools (same as in server.py)
    tools = [
        # Core Pixeltable functions
        pixeltable_init,
        pixeltable_create_table,
        pixeltable_get_table,
        pixeltable_list_tables,
        pixeltable_drop_table,
        pixeltable_create_view,
        pixeltable_create_snapshot,
        pixeltable_create_dir,
        pixeltable_drop_dir,
        pixeltable_list_dirs,
        pixeltable_ls,
        pixeltable_move,
        pixeltable_list_functions,
        pixeltable_configure_logging,
        pixeltable_get_types,
        pixeltable_get_version,
        pixeltable_create_replica,
        pixeltable_query_table,
        pixeltable_get_table_schema,
        pixeltable_insert_data,
        pixeltable_add_computed_column,
        pixeltable_check_dependencies,
        pixeltable_install_yolox,
        pixeltable_install_openai,
        pixeltable_install_huggingface,
        pixeltable_install_all_dependencies,
        pixeltable_smart_install,
        pixeltable_auto_install_for_expression,
        pixeltable_suggest_install_from_error,
        pixeltable_system_diagnostics,
        pixeltable_query,
        pixeltable_create_udf,
        pixeltable_create_array,
        pixeltable_create_tools,
        pixeltable_connect_mcp,
        pixeltable_create_image_type,
        pixeltable_create_video_type,
        pixeltable_create_audio_type,
        pixeltable_create_array_type,
        pixeltable_create_json_type,
        pixeltable_set_datastore,
        pixeltable_get_datastore,
        # REPL functions
        execute_python,
        introspect_function,
        list_available_functions,
        install_package,
        log_bug,
        log_missing_feature,
        log_success,
        generate_bug_report,
        get_session_summary,
    ]
    
    print("=== Pixeltable MCP Server Tools ===\n")
    
    categories = {
        "Table Management": ["create_table", "get_table", "list_tables", "drop_table", "query_table", "get_table_schema"],
        "Data Operations": ["insert_data", "add_computed_column", "create_view", "create_snapshot", "create_replica"],
        "Directory Management": ["create_dir", "drop_dir", "list_dirs", "ls", "move"],
        "Configuration": ["init", "set_datastore", "get_datastore", "configure_logging", "get_version"],
        "AI/ML Integration": ["create_udf", "create_array", "create_tools", "connect_mcp", "query"],
        "Data Types": ["create_image_type", "create_video_type", "create_audio_type", "create_array_type", "create_json_type"],
        "Dependencies": ["check_dependencies", "install_yolox", "install_openai", "install_huggingface", "install_all_dependencies", "smart_install"],
        "REPL & Debug": ["execute_python", "introspect_function", "list_available_functions", "log_bug", "log_missing_feature", "generate_bug_report"],
        "Utilities": ["list_functions", "get_types", "system_diagnostics", "auto_install_for_expression", "suggest_install_from_error"],
    }
    
    # Group tools by category
    categorized = {cat: [] for cat in categories}
    uncategorized = []
    
    for tool in tools:
        name = tool.__name__
        doc = tool.__doc__.split('\n')[0] if tool.__doc__ else "No description"
        found = False
        
        for cat, keywords in categories.items():
            if any(kw in name for kw in keywords):
                categorized[cat].append((name, doc))
                found = True
                break
        
        if not found:
            uncategorized.append((name, doc))
    
    # Print categorized tools
    for category, tool_list in categorized.items():
        if tool_list:
            print(f"\n{category}:")
            print("-" * len(category))
            for name, doc in sorted(tool_list):
                print(f"  • {name}: {doc}")
    
    if uncategorized:
        print(f"\nOther:")
        print("-----")
        for name, doc in sorted(uncategorized):
            print(f"  • {name}: {doc}")
    
    print(f"\n\nTotal tools available: {len(tools)}")

if __name__ == "__main__":
    list_tools()