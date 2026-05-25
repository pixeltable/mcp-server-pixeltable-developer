#!/usr/bin/env python3
"""Introspect the live FastMCP server and print its tools, resources, and prompts.

This used to maintain a hand-written list that drifted from server.py. It now
loads server.mcp directly so the inventory is always accurate.

Used as both a developer sanity check and (via CI) a smoke test that the server
module imports cleanly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def list_all() -> None:
    """Print every primitive registered on the FastMCP server."""
    from mcp_server_pixeltable_stio.server import mcp

    tool_manager = getattr(mcp, '_tool_manager', None)
    resource_manager = getattr(mcp, '_resource_manager', None)
    prompt_manager = getattr(mcp, '_prompt_manager', None)

    if tool_manager is None:
        raise RuntimeError("FastMCP tool manager unavailable (mcp._tool_manager missing).")

    # Tools, grouped roughly by category for human reading.
    tools = list(tool_manager.list_tools())
    resources = list(resource_manager.list_resources()) if resource_manager else []
    resource_templates = list(resource_manager.list_templates()) if resource_manager else []
    prompts = list(prompt_manager.list_prompts()) if prompt_manager else []

    categories = {
        "Initialization": ["pixeltable_init"],
        "Table Management": ["pixeltable_create_table", "pixeltable_drop_table",
                             "pixeltable_create_view", "pixeltable_create_snapshot"],
        "Data Operations": ["pixeltable_create_replica", "pixeltable_query_table",
                            "pixeltable_insert_data", "pixeltable_add_computed_column",
                            "pixeltable_query"],
        "Directory Management": ["pixeltable_create_dir", "pixeltable_drop_dir", "pixeltable_move"],
        "Configuration": ["pixeltable_configure_logging", "pixeltable_set_datastore"],
        "AI/ML Integration": ["pixeltable_create_udf", "pixeltable_create_array",
                              "pixeltable_create_tools", "pixeltable_connect_mcp"],
        "Dependencies": ["pixeltable_check_dependencies", "pixeltable_install_dependency"],
        "Data Types": ["pixeltable_create_type"],
        "Documentation": ["pixeltable_search_docs"],
        "Scaffolding": ["pixeltable_scaffold_project", "pixeltable_list_project_templates"],
        "REPL & Debug": ["execute_python", "introspect_function",
                         "list_available_functions", "install_package"],
        "Bug Logging": ["log_bug", "log_missing_feature", "log_success",
                        "generate_bug_report", "get_session_summary"],
        "Display": ["display_in_browser"],
    }

    categorized: dict[str, list[tuple[str, str]]] = {cat: [] for cat in categories}
    other: list[tuple[str, str]] = []

    for tool in tools:
        name = tool.name
        description = (tool.description or 'No description').strip().split('\n', 1)[0]
        placed = False
        for cat, names in categories.items():
            if name in names:
                categorized[cat].append((name, description))
                placed = True
                break
        if not placed:
            other.append((name, description))

    print("=" * 60)
    print("  Pixeltable MCP Server — Registered Primitives")
    print("=" * 60)

    print(f"\n--- TOOLS ({len(tools)}) ---\n")
    for cat, items in categorized.items():
        if not items:
            continue
        print(f"  {cat}:")
        for name, desc in sorted(items):
            print(f"    \u2022 {name}: {desc}")
    if other:
        print("  Other:")
        for name, desc in sorted(other):
            print(f"    \u2022 {name}: {desc}")

    print(f"\n--- RESOURCES ({len(resources) + len(resource_templates)}) ---\n")
    for r in sorted(resources, key=lambda x: str(x.uri)):
        desc = (r.description or '').strip().split('\n', 1)[0]
        print(f"  \u2022 {r.uri}: {desc}")
    for t in sorted(resource_templates, key=lambda x: x.uri_template):
        desc = (t.description or '').strip().split('\n', 1)[0]
        print(f"  \u2022 {t.uri_template}: {desc}")

    print(f"\n--- PROMPTS ({len(prompts)}) ---\n")
    for p in sorted(prompts, key=lambda x: x.name):
        desc = (p.description or '').strip().split('\n', 1)[0]
        print(f"  \u2022 {p.name}: {desc}")

    print(f"\n{'=' * 60}")
    print(f"  Total: {len(tools)} tools, "
          f"{len(resources) + len(resource_templates)} resources, "
          f"{len(prompts)} prompts")
    print("=" * 60)


if __name__ == "__main__":
    list_all()
