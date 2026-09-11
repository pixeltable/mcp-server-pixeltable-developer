"""Focused workflow prompts for Pixeltable application development."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field


def register_prompts(server: MCPServer[Any]) -> None:
    """Register the four supported workflow prompts."""

    @server.prompt(
        name="pixeltable_build_app",
        title="Build a Pixeltable application",
        description="Plan and implement one app.py with TableModel tables and FastAPIRouter routes.",
    )
    def build_app(
        goal: Annotated[str, Field(min_length=1, description="What the application should do.")],
    ) -> str:
        return f"""Build a Pixeltable application for this goal:

{goal}

Read pixeltable://guidance/app first. Work in one source-controlled application file.
Start from `pxt init` and `pxt service example --out app.py`; declare stored columns
with annotations, computed columns with assignments, and HTTP routes with
`FastAPIRouter` imported from `pixeltable.serving`. Validate, diff, and update in
that order. Use deterministic local fixtures before adding provider calls."""

    @server.prompt(
        name="pixeltable_build_rag",
        title="Build Pixeltable retrieval",
        description="Design document or multimodal ingestion, chunking, embedding, and retrieval in Pixeltable.",
    )
    def build_rag(
        source_media: Annotated[
            str,
            Field(min_length=1, description="Document, image, video, or audio inputs to ingest."),
        ],
        query_behavior: Annotated[
            str,
            Field(min_length=1, description="How callers should search and use the retrieved context."),
        ],
    ) -> str:
        return f"""Build a Pixeltable retrieval application.

Inputs: {source_media}
Query behavior: {query_behavior}

Use a TableModel application and Pixeltable computed columns. Use the appropriate
document, image, video, or audio iterator; persist chunks in a view; add the
modality-appropriate embedding index; and call similarity with a named modality
argument. Keep the catalog as the system of record. Add a FastAPIRouter retrieval
route, then validate, diff, update, insert deterministic fixtures, and test
retrieval."""

    @server.prompt(
        name="pixeltable_build_agent",
        title="Build a Pixeltable tool-calling agent",
        description="Design an agent whose model responses and tool executions are persisted as computed columns.",
    )
    def build_agent(
        objective: Annotated[str, Field(min_length=1, description="The agent's objective.")],
        tools: Annotated[str, Field(min_length=1, description="The tools the model may call.")],
    ) -> str:
        return f"""Build a persistent Pixeltable tool-calling agent.

Objective: {objective}
Tools: {tools}

Declare request, model response, tool-call, tool-result, and final-response columns
in a TableModel application. Construct tool definitions with `pxt.tools()`, pass
them to the provider function, and execute requested calls with `invoke_tools()`.
Preserve provider outputs in their native type. Add an HTTP route, use mocked
provider results for local wiring tests, and do not claim live-provider coverage."""

    @server.prompt(
        name="pixeltable_debug_computation",
        title="Debug a Pixeltable computation",
        description="Inspect failed computed values and build a bounded recovery procedure.",
    )
    def debug_computation(
        table: Annotated[str, Field(min_length=1, description="Table or view path with failures.")],
        column: Annotated[str, Field(min_length=1, description="Computed column to diagnose.")],
    ) -> str:
        return f"""Diagnose failures in `{table}.{column}`.

Describe the table, list errors filtered to `{column}`, and inspect the input
columns the expression depends on. Fix the source data, configuration, provider
credentials, or application code that caused the error. Preview
`pxt recompute {table} {column} --errors-only -n`; apply it with `-f` only after the
preview is correct. If the computed expression itself changed, rename the column
or drop and re-add it because in-place expression migration is unsupported."""
