# Pixeltable Developer MCP Server

A local Model Context Protocol server for building, inspecting, and operating
[Pixeltable](https://pixeltable.com/) applications. Version 0.2 follows the
application-first workflow in [pixeltable-skill 2.8.3](https://github.com/pixeltable/pixeltable-skill/blob/f550e6ed757b48635e4f53900840f4e9a1fb4c93/skills/pixeltable-skill/SKILL.md): generate one `app.py`, apply its schema, and serve its routes with the `pxt` CLI.

## Compatibility

| Component | Supported line |
|---|---|
| Python | 3.11 or newer |
| Pixeltable | `>=0.7.6,<0.8` with the `serve` extra |
| MCP Python SDK | `>=2.2,<3` |
| Pixeltable skill | 2.8.3 |
| Production transport | local `stdio` |

The server is a **beta developer tool**. It runs with the permissions of the
process that launched it and can mutate the selected Pixeltable catalog. Point
`PIXELTABLE_MCP_PROJECT_ROOT` at the application directory and `PIXELTABLE_HOME`
at the intended catalog before startup. Both locations are fixed for the life of
the process. Use a separate catalog for evaluation and automated tests.

## Install and connect

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then install the server:

```bash
uv tool install --from git+https://github.com/pixeltable/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer
mcp-server-pixeltable-developer --version
```

For reproducible development from a clone:

```bash
git clone https://github.com/pixeltable/mcp-server-pixeltable-developer.git
cd mcp-server-pixeltable-developer
uv sync --frozen --extra test
```

Configure a client to launch the server over `stdio`. Replace the repository,
application project, and catalog paths with absolute paths:

```json
{
  "mcpServers": {
    "pixeltable": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/mcp-server-pixeltable-developer",
        "mcp-server-pixeltable-developer"
      ],
      "env": {
        "PIXELTABLE_MCP_PROJECT_ROOT": "/absolute/path/to/pixeltable-app",
        "PIXELTABLE_HOME": "/absolute/path/to/.pixeltable"
      }
    }
  }
}
```

Restart the client after changing its MCP configuration. Keep one server
process per catalog when performing schema or service mutations.

## Default interface

Version 0.2 exposes a focused interface instead of mirroring the whole
Pixeltable Python API.

### Tools

| Area | Tools |
|---|---|
| Inspect | `pixeltable_list_catalog`, `pixeltable_describe`, `pixeltable_rows`, `pixeltable_get_row`, `pixeltable_errors` |
| Data | `pixeltable_insert_rows`, `pixeltable_recompute` |
| App scaffold | `pixeltable_scaffold_app` |
| Schema lifecycle | `pixeltable_schema_check`, `pixeltable_schema_diff`, `pixeltable_schema_update`, `pixeltable_schema_prune` |
| Service lifecycle | `pixeltable_service_check`, `pixeltable_service_diff`, `pixeltable_service_update`, `pixeltable_service_list`, `pixeltable_service_stop`, `pixeltable_service_prune` |

Read tools declare read-only MCP annotations. Mutation tools declare their
write and destructive behavior. Arguments and results use typed schemas, and
recoverable failures are returned as MCP tool errors so clients can retry with
corrected input.

### Resources

| URI | Purpose |
|---|---|
| `pixeltable://status` | Server, dependency, transport, catalog, and unsafe-mode status |
| `pixeltable://catalog` | Current catalog inventory |
| `pixeltable://guidance/app` | Pixeltable 0.7.6 application workflow |
| `pixeltable://guidance/cloud` | Cloud preparation guidance; it does not deploy resources |

### Prompts

- `pixeltable_build_app`
- `pixeltable_build_rag`
- `pixeltable_build_agent`
- `pixeltable_debug_computation`

The prompts are short task guides aligned with pixeltable-skill 2.8.3. The
skill remains the detailed source for provider output shapes, multimodal views,
indexes, tool calling, serving, debugging, and Cloud workflows.

## Unsafe mode

Host-code execution, package installation, and browser display are disabled by
default. A trusted local user can opt in before starting the server:

```bash
export PIXELTABLE_MCP_ENABLE_UNSAFE=1
```

This adds:

- `pixeltable_unsafe_execute_python`
- `pixeltable_unsafe_install_package`
- `pixeltable_unsafe_display`

These tools are available only on `stdio`. They can execute arbitrary Python,
change the server environment, read files available to the server process, and
render untrusted content. Enable them only for a trusted, local MCP client and
a disposable development environment. Do not expose unsafe mode through an
HTTP transport.

The repository may use an in-process or HTTP harness in tests. HTTP is not a
supported production transport for version 0.2. A future HTTP release must add
authorization, concurrency and request limits, origin and host validation,
per-principal isolation, and an explicit threat model before it is supported.

## Recommended Pixeltable workflow

Start an application with the current Pixeltable CLI:

```bash
pip install 'pixeltable[serve]>=0.7.6,<0.8'
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt service check app.py
pxt service update app.py my_app
pxt service list
```

`my_app` is a catalog directory. It is not a filesystem directory. Edit
`app.py` and apply the schema again when the model changes. Apply the service
again after route changes. Use `pxt service list` to discover the assigned URL.

The application file should declare `TableModel` classes and, when HTTP routes
are needed, a `FastAPIRouter`. Stored columns use annotations. Computed columns
use assignments. Types are non-nullable by default; write `T | None` for an
optional column. Do not use `pxt.Required`.

Schema updates intentionally require a review step. `schema_check` validates
the file, `schema_diff` previews catalog changes, and `schema_update` applies
them. Prune operations are separate because they can remove catalog objects.
Changing an existing computed-column expression in place is unsupported by
Pixeltable; rename the column, or drop and re-add it in separate updates.

For Cloud, prepare the same `app.py`, set `PIXELTABLE_API_KEY`, configure the
target `pxt://org:db`, and review `pixeltable://guidance/cloud`. The MCP server
does not create paid resources or deploy to Cloud automatically.

## Develop and verify

Use a disposable catalog:

```bash
export PIXELTABLE_MCP_PROJECT_ROOT="$PWD"
export PIXELTABLE_HOME="$(mktemp -d)/catalog"
uv sync --frozen --extra test
uv run pytest -q
PIXELTABLE_DISABLE_STDOUT=1 uv run pytest --run-slow -q
uv run python list_tools.py
./scripts/run-conformance.sh
```

Run the MCP Inspector only as a local test harness:

```bash
uv run mcp dev src/mcp_server_pixeltable_developer/server.py:mcp
```

Before release, also run the latest-dependency compatibility job, the MCP
in-memory contract tests, the subprocess `stdio` smoke test, lint, type checks,
and package build verification described in the review report.

## Documentation

- [0.1.0 evidence review](docs/review-0.1.0.md)
- [0.2.0 implementation review](docs/review-0.2.0.md)
- [Migration from 0.1 to 0.2](docs/migration-0.1-to-0.2.md)
- [Agent evaluation protocol](evals/README.md)
- [Pixeltable documentation](https://docs.pixeltable.com/)
- [Pixeltable get started](https://www.pixeltable.com/get-started.md)
- [Pixeltable LLM reference](https://www.pixeltable.com/llms.txt)
- [MCP Python SDK 2 documentation](https://py.sdk.modelcontextprotocol.io/v2/)

## License

Apache-2.0. See [LICENSE](LICENSE).
