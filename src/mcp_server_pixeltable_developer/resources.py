"""Focused MCP resources for runtime context and Pixeltable guidance."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import __version__
from .models import CatalogEntry, CatalogResult, StatusResource
from .runtime import CommandRunner, ServerConfig

APP_GUIDANCE = """# Pixeltable application workflow

1. Run `pxt init` once in the project root.
2. Start with `pxt service example --out app.py` (or `pxt schema example --brief --out schema.py`).
3. Declare tables on `TableModel = pxt.model_base()`. Stored columns use annotations; computed columns use assignments.
4. Import `FastAPIRouter` from `pixeltable.serving` and declare HTTP routes on it.
   Keep tables, computed columns, and routes in the source-controlled application file.
5. Validate with `pxt service check app.py`, inspect with
   `pxt service diff app.py TARGET`, then apply with
   `pxt service update app.py TARGET`.

Stored columns are non-nullable by default. Write `pxt.String | None` when null is allowed. Do not use `pxt.Required`.
Create computed columns declaratively. To change a computed expression, edit it and
run `pxt schema update`: it records the new expression and keeps the stored values,
so run `pxt recompute TABLE COLUMN -f` to refresh existing rows. Changing a column's
type is UNSUPPORTED and applies nothing, even with `--allow-destructive`: rename a
computed column instead, and declare a stored column `T | None` up front.
Use Pixeltable iterators, embedding indexes, and `pxt.tools()` / `invoke_tools()`
for multimodal retrieval and agents. Do not add a separate dataframe store or
vector database.
For failures, inspect `pxt errors TABLE` and preview
`pxt recompute TABLE COLUMN --errors-only -n` before applying it with `-f`.

Row-addressed operations (`pxt get`, `pxt errors`) require a primary key.
`pxt service example` generates one: `id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)`,
computed on insert. Declare a key in every model before its first schema update:
Pixeltable cannot add a primary key to a table that already exists.
"""


CLOUD_GUIDANCE = """# Pixeltable Cloud preparation

Cloud behavior in this server is source-reviewed but not live-tested. Review each diff before applying it.

1. Sign in with `pxt login`, or export `PIXELTABLE_API_KEY`, which wins over a sign-in.
2. Add `[[pixeltable.database]]` with `name = 'pxt://ORG:DATABASE'` to `pixeltable.toml`.
3. Set provider keys with `pxt secret set pxt://ORG OPENAI_API_KEY=...`. Never put
   secrets in `pixeltable.toml`: a database entry with a `secrets` mapping fails to load.
4. Run `pxt db diff`, then `pxt db update pxt://ORG:DATABASE -f`. The first update creates the database.
5. Run `pxt schema diff` and `pxt schema update` against the hosted target.
6. Run `pxt service diff` and `pxt service update` only after the database code and schema are current.
7. To read the hosted database's status, rows, and logs from an agent, start or stop it, restart its services, or
   set a secret, connect the hosted Cloud MCP at https://www.pixeltable.com/mcp/cloud. It cannot deploy, change
   schema, or delete anything.

This server's schema and service tools accept explicit `pxt://` targets. They never
create databases or set secrets. A hosted `schema_update` or `service_update` call
changes that hosted target, so inspect the corresponding diff first.
"""


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def register_resources(server: MCPServer[Any], config: ServerConfig, runner: CommandRunner) -> None:
    """Register the four supported resources."""

    @server.resource(
        "pixeltable://status",
        name="pixeltable_status",
        title="Pixeltable MCP status",
        description="Redacted server, package, project, and feature-mode versions.",
        mime_type="application/json",
    )
    def status_resource() -> str:
        status = StatusResource(
            server_name="pixeltable-developer",
            server_version=__version__,
            mcp_version=_distribution_version("mcp"),
            pixeltable_version=_distribution_version("pixeltable"),
            project=config.project_root.name,
            transport=config.transport,
            unsafe_tools_enabled=config.unsafe_enabled,
        )
        return status.model_dump_json()

    @server.resource(
        "pixeltable://catalog",
        name="pixeltable_catalog",
        title="Pixeltable catalog",
        description="Tables, views, and directories in the configured local catalog.",
        mime_type="application/json",
    )
    async def catalog_resource() -> str:
        result = await runner.pxt(["ls", "--tree", "--json"])
        if not isinstance(result.data, dict):
            raise RuntimeError("Pixeltable returned an unexpected catalog listing")
        tree_payload = result.data.get("tree")
        entries_payload = result.data.get("entries", [])
        if isinstance(tree_payload, dict):
            entries_payload = tree_payload.get("entries", entries_payload)
        if not isinstance(entries_payload, list):
            raise RuntimeError("Pixeltable returned an unexpected catalog listing")
        tree = CatalogEntry.model_validate(tree_payload) if tree_payload is not None else None
        catalog = CatalogResult(
            path="",
            entries=[CatalogEntry.model_validate(entry) for entry in entries_payload],
            tree=tree,
        )
        return catalog.model_dump_json()

    @server.resource(
        "pixeltable://guidance/app",
        name="pixeltable_application_guidance",
        title="Pixeltable application guidance",
        description="Application-first workflow aligned with Pixeltable 0.7.16 and skill 2.12.0.",
        mime_type="text/markdown",
    )
    def application_guidance_resource() -> str:
        return APP_GUIDANCE

    @server.resource(
        "pixeltable://guidance/cloud",
        name="pixeltable_cloud_guidance",
        title="Pixeltable Cloud preparation",
        description="Reviewed deployment ordering and configuration guidance; not live-tested.",
        mime_type="text/markdown",
    )
    def cloud_guidance_resource() -> str:
        return CLOUD_GUIDANCE
