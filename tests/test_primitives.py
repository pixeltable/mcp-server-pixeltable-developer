"""Smoke-test that every MCP primitive registers cleanly on the live server."""

from __future__ import annotations

import pytest

# Importing the server triggers FastMCP registration of every tool/resource/prompt.
# The Pydantic UserWarning about the `schema` field shadowing is expected and benign.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field name \"schema\".*shadows.*:UserWarning"
)


def _server():
    from mcp_server_pixeltable_stio.server import mcp

    return mcp


def test_tools_register():
    mcp = _server()
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    # 35 tools after the v0.6 alignment (init + scaffold tools).
    assert len(tool_names) == 35, f"unexpected tool count: {tool_names}"

    expected = {
        # initialization & catalog mgmt
        "pixeltable_init",
        "pixeltable_create_table", "pixeltable_drop_table",
        "pixeltable_create_view", "pixeltable_create_snapshot",
        # data ops
        "pixeltable_create_replica", "pixeltable_query_table",
        "pixeltable_insert_data", "pixeltable_add_computed_column",
        "pixeltable_query",
        # directories
        "pixeltable_create_dir", "pixeltable_drop_dir", "pixeltable_move",
        # config / docs / types / deps
        "pixeltable_configure_logging", "pixeltable_set_datastore",
        "pixeltable_search_docs",
        "pixeltable_check_dependencies", "pixeltable_install_dependency",
        "pixeltable_create_type",
        # ai/ml integration
        "pixeltable_create_udf", "pixeltable_create_array",
        "pixeltable_create_tools", "pixeltable_connect_mcp",
        # scaffolding (pixeltable-new wrappers)
        "pixeltable_scaffold_project", "pixeltable_list_project_templates",
        # repl / debug
        "execute_python", "introspect_function",
        "list_available_functions", "install_package",
        # bug logging
        "log_bug", "log_missing_feature", "log_success",
        "generate_bug_report", "get_session_summary",
        # display
        "display_in_browser",
    }
    assert expected.issubset(set(tool_names)), (
        f"missing tools: {expected - set(tool_names)}"
    )


def test_resources_register():
    mcp = _server()
    rm = mcp._resource_manager
    resource_uris = {str(r.uri) for r in rm.list_resources()}
    template_uris = {t.uri_template for t in rm.list_templates()}

    static_expected = {
        "pixeltable://tables", "pixeltable://directories", "pixeltable://ls",
        "pixeltable://version", "pixeltable://config/datastore",
        "pixeltable://types", "pixeltable://functions",
        "pixeltable://tools", "pixeltable://help", "pixeltable://diagnostics",
    }
    template_expected = {
        "pixeltable://tables/{path}",
        "pixeltable://tables/{path}/schema",
        "pixeltable://ls/{path}",
    }

    assert static_expected.issubset(resource_uris), (
        f"missing resources: {static_expected - resource_uris}"
    )
    assert template_expected.issubset(template_uris), (
        f"missing resource templates: {template_expected - template_uris}"
    )
    # Total surface stays at 13 (10 static + 3 templates).
    assert len(resource_uris) + len(template_uris) == 13


def test_prompts_register():
    mcp = _server()
    prompt_names = [p.name for p in mcp._prompt_manager.list_prompts()]

    expected = {
        # existing
        "pixeltable_usage_guide", "getting_started",
        "computer_vision_pipeline", "rag_pipeline",
        "video_analysis_pipeline", "audio_processing_pipeline",
        # added in the v0.6 alignment
        "tool_calling_agent_pipeline", "agent_with_memory_pipeline",
        "video_rag_agent_pipeline", "agentic_patterns_guide",
        "ml_data_pipeline",
    }
    assert set(prompt_names) == expected, (
        f"prompt drift -- unexpected={set(prompt_names) - expected}, "
        f"missing={expected - set(prompt_names)}"
    )


def test_list_tools_resource_matches_server():
    """pixeltable://tools must reflect actual FastMCP registration."""
    from mcp_server_pixeltable_stio.core.helpers import pixeltable_list_tools

    mcp = _server()
    live = {t.name for t in mcp._tool_manager.list_tools()}

    result = pixeltable_list_tools()
    assert result["success"]
    reported = {tool["name"] for cat in result["categories"].values() for tool in cat}
    assert reported == live, f"resource drift: {reported.symmetric_difference(live)}"
