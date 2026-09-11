# Migrating from 0.1 to 0.2

Version 0.2 updates the server to Pixeltable 0.7.6, the MCP Python SDK 2.2,
Python 3.11+, and pixeltable-skill 2.8.3. It also changes the server from a
broad Python-SDK wrapper into a focused local operator for Pixeltable
application files.

This is a breaking release. Read the interface and security changes before
replacing a 0.1 installation.

## Compatibility changes

| Component | 0.1 | 0.2 |
|---|---|---|
| Python | `>=3.10` | `>=3.11` |
| Pixeltable | declared `>=0.6.3`, locked 0.6.3 | `>=0.7.6,<0.8` with `serve` |
| MCP SDK | declared `>=1.10`, locked 1.11.0 | `>=2.2,<3` |
| MCP server class | `mcp.server.fastmcp.FastMCP` | `mcp.server.mcpserver.MCPServer` |
| Production transport | implicit `stdio` | explicit local `stdio` |
| Package import | `mcp_server_pixeltable_stio` | `mcp_server_pixeltable_developer` |
| Guidance | large embedded 0.6-era templates | focused prompts aligned with pixeltable-skill 2.8.3 |
| Unsafe capabilities | mixed into the normal interface | absent unless explicitly enabled |

The console command remains `mcp-server-pixeltable-developer`. The misspelled
0.1 Python package name remains as a stderr-warning compatibility shim for this
release only; new code must import `mcp_server_pixeltable_developer`.

## Before upgrading

1. Record the current server version, Pixeltable version, client configuration,
   `PIXELTABLE_HOME`, and tool names used by automations.
2. Stop every 0.1 server process that points at the catalog.
3. Back up the catalog according to your local data policy. The server does not
   perform or verify a backup.
4. Copy any useful UDF, computed-column, or pipeline code out of the persistent
   0.1 REPL and into a reviewed `app.py`; REPL state is not a migration format.
5. Remove any workflow that calls replica/publish. Pixeltable 0.7.6 removed
   replica support and 0.2 has no substitute tool.
6. Move any reliance on the browser canvas or package installer behind an
   explicit local-only unsafe-mode decision.

Use a separate catalog for the first 0.2 verification:

```bash
export PIXELTABLE_MCP_PROJECT_ROOT="$PWD"
export PIXELTABLE_HOME="$(mktemp -d)/catalog"
```

Do not point the new process at the production catalog until its client
handshake and application workflow have passed in the temporary catalog.

## Install 0.2

From a source checkout:

```bash
git pull --ff-only
uv sync --frozen --extra test
uv run mcp-server-pixeltable-developer --version
```

For a tool installation after 0.2 is released:

```bash
uv tool install --reinstall --from git+https://github.com/pixeltable/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer
mcp-server-pixeltable-developer --version
```

Use an immutable tag or commit in managed environments. A bare Git URL follows
the repository's moving default branch.

Keep the client on `stdio`:

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

Restart the MCP client. Initialization must fail visibly if the server cannot
load its declared dependency versions or open the selected catalog. An empty
fallback server is no longer a valid recovery path.

## New default interface

Version 0.2 exposes these 18 tools:

| Workflow | Tool |
|---|---|
| Inspect catalog | `pixeltable_list_catalog` |
| Inspect one object | `pixeltable_describe` |
| Read a bounded preview | `pixeltable_rows` |
| Read one row | `pixeltable_get_row` |
| Inspect failed cells | `pixeltable_errors` |
| Insert input rows | `pixeltable_insert_rows` |
| Retry computations | `pixeltable_recompute` |
| Generate the application file | `pixeltable_scaffold_app` |
| Validate schema source | `pixeltable_schema_check` |
| Preview schema changes | `pixeltable_schema_diff` |
| Apply schema changes | `pixeltable_schema_update` |
| Remove schema objects | `pixeltable_schema_prune` |
| Validate service source | `pixeltable_service_check` |
| Preview service changes | `pixeltable_service_diff` |
| Apply or refresh a service | `pixeltable_service_update` |
| Discover service URLs | `pixeltable_service_list` |
| Stop a service | `pixeltable_service_stop` |
| Remove service state | `pixeltable_service_prune` |

It exposes four resources:

- `pixeltable://status`
- `pixeltable://catalog`
- `pixeltable://guidance/app`
- `pixeltable://guidance/cloud`

It exposes four prompts:

- `pixeltable_build_app`
- `pixeltable_build_rag`
- `pixeltable_build_agent`
- `pixeltable_debug_computation`

Clients should discover schemas, annotations, and descriptions with
`tools/list`, `resources/list`, and `prompts/list` instead of copying a fixed
count or reading a private SDK manager.

## Tool mapping

The 0.2 design routes application changes through `app.py` and the released
`pxt` CLI. This gives users a reviewable source file and a diff before a catalog
or service mutation.

| 0.1 operation | 0.2 path |
|---|---|
| `pixeltable_init` | Server startup plus `pixeltable://status`; use `pxt init` once in the project |
| `pixeltable_create_table`, `pixeltable_create_view`, `pixeltable_create_snapshot` | Declare a `TableModel` in `app.py`, then schema check → diff → update |
| `pixeltable_drop_table`, directory drop/move | Edit `app.py`, review schema diff, then use update or the explicit prune tool |
| `pixeltable_query_table`, `pixeltable_query` | `pixeltable_rows` and `pixeltable_get_row` with bounded, typed results |
| `pixeltable_insert_data` | `pixeltable_insert_rows` |
| `pixeltable_add_computed_column` | Add or rename an assignment in `app.py`, then schema check → diff → update |
| Computation failure inspection | `pixeltable_errors` |
| Retry failed computed cells | `pixeltable_recompute` |
| `pixeltable_scaffold_project`, `pixeltable_list_project_templates` | `pixeltable_scaffold_app`; there is one generated application shape |
| `pixeltable_get_version`, diagnostics/config resources | `pixeltable://status` |
| Table, directory, and schema resources | `pixeltable://catalog`, `pixeltable_list_catalog`, `pixeltable_describe` |
| Large pipeline prompt catalog | Four focused 0.2 prompts plus pixeltable-skill 2.8.3 |
| `pixeltable_search_docs` | Guidance resources and links to official Pixeltable documentation |
| `pixeltable_create_replica` | Removed; Pixeltable 0.7.6 has no replica/publish API |
| Type and transient object factories | Removed; write normal Pixeltable declarations in `app.py` |
| UDF and tool creation from strings | Removed from default; write reviewed Python in `app.py` |
| Dependency auto-detection and installation | Removed from default; use explicit project dependencies |
| Bug-log tools | Removed; use normal server logs and the repository issue tracker |

## Migrate a Pixeltable application

Create the canonical application file:

```bash
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt service check app.py
pxt service update app.py my_app
pxt service list
```

The final argument, `my_app`, is a Pixeltable catalog directory. It is not a
directory on disk. Use `pxt.get_table('my_app.docs')` to get a local table after
the schema exists.

Application files follow these rules:

- Define tables as `TableModel` classes. Do not call `pxt.create_table()` or
  `pxt.get_table()` during module import.
- A type annotation creates a stored column. An assignment creates a computed
  column.
- Types are non-nullable by default. Use `T | None` for optional values. Replace
  `pxt.Required[T]` and `"Required[T]"` declarations.
- Put model indexes in `__indexes__`. Use `pxt.EmbeddingIndex(...)` for semantic
  retrieval.
- Import iterators from `pixeltable.functions.*`; do not use the deprecated
  `pixeltable.iterators` package.
- Use `.similarity(string=query)` and name the score in a select, for example
  `score=Docs.body.similarity(string=query)`.
- Extract provider text such as `.choices[0].message.content` or `.text` before
  storing, embedding, or concatenating it.
- Declare HTTP routes with `FastAPIRouter`. Apply the schema before the service.
  Discover the assigned URL with `pxt service list`.

### Computed-column changes

Changing an existing computed-column expression in place is unsupported in the
application schema workflow. `--allow-destructive` does not make that update
valid. Rename the column and apply the reviewed destructive diff, or drop it in
one update and add it in a later update. Use `pixeltable_recompute` for retrying
failed existing cells; it does not replace a column's definition.

### Cloud preparation

The same `app.py` is used for Cloud. Set `PIXELTABLE_API_KEY`, configure the
target `pxt://org:db`, and use `pixeltable://guidance/cloud` to prepare the
commands. The 0.2 server does not make paid provider calls, create Cloud
resources, or deploy automatically. Review every `db`, `schema`, and `service`
change before running it.

## Result and error changes for MCP clients

0.1 commonly returned dictionaries such as `{"success": false, "error": ...}`.
Those are normal return values at the MCP protocol layer, so clients can mistake
them for successful calls.

In 0.2:

- Successful tools return typed structured content.
- Bad tool arguments are rejected by the generated input schema.
- A recoverable execution problem raises MCP `ToolError`; clients receive
  `is_error=true` and may correct the call.
- Protocol and server-state rejections use MCP protocol errors.
- An unknown resource is handled as `ResourceNotFoundError` by MCPServer and
  reaches the client as JSON-RPC `-32602` Invalid params.
- Unexpected exceptions are logged with their traceback and sanitized before
  they reach the client.

Update client code to branch on MCP `is_error` or a protocol exception. Do not
parse a `success` Boolean from a text block. Consume `structured_content`
according to the tool's advertised output schema.

Read-only and mutation annotations are behavioral hints for clients. They are
not an authorization boundary. Review destructive schema/service prune calls
even when the client surfaces an annotation-based confirmation.

## Unsafe mode

The three unsafe tools are not registered by default. A trusted local operator
can enable them before server startup:

```bash
export PIXELTABLE_MCP_ENABLE_UNSAFE=1
```

This adds:

- `pixeltable_unsafe_execute_python`
- `pixeltable_unsafe_install_package`
- `pixeltable_unsafe_display`

They are restricted to local `stdio`. Python execution and package installation
have the authority of the host process. Display can expose local media and
renders content in a browser. Use a disposable environment and a dedicated
catalog. Never enable these tools for an untrusted MCP client, a shared machine,
or any HTTP endpoint.

Unsafe state is fixed at startup and reported by `pixeltable://status`. Changing
the environment variable requires a server restart.

## Transport changes

Version 0.2 supports local `stdio` in production. The client launches the server
as a child process; stdout is the protocol wire, so application logging goes to
stderr. Do not print during import or replace global `sys.stdout`/`sys.stderr`
inside handlers.

An HTTP transport may exist inside the test harness to exercise MCP contracts.
It is not a supported deployment interface. Do not expose it as a workaround
for `pxt.mcp_udfs(url)`: that Pixeltable function consumes a URL-based MCP
server, while this server's production interface is `stdio`.

Legacy SSE is superseded by Streamable HTTP and must not be used for new work.
A future production HTTP mode requires OAuth token verification, transport
security, body, concurrency, and rate limits, origin and host validation, and
isolated state before it can be supported.

## Developer migration checklist

1. Replace `FastMCP` with `MCPServer` from `mcp.server.mcpserver`.
2. Construct the server with a stable name, title, description, website URL,
   package version, instructions, and one lifespan.
3. Move startup checks and cleanup into the lifespan. Remove the empty fallback
   server and propagate fatal startup failure.
4. Replace private manager access with public `list_tools`, `list_resources`,
   `list_resource_templates`, and `list_prompts` methods.
5. Model every input and output with precise Python/Pydantic types, `Literal`
   enums, bounds, and descriptions.
6. Add accurate `read_only_hint`, `open_world_hint`, `destructive_hint`, and
   `idempotent_hint` annotations to every tool.
7. Replace returned error envelopes with `ToolError` for recoverable tool
   failures and `ResourceNotFoundError` for missing resources. Reserve
   `MCPError` for protocol or server-state rejection.
8. Set `PIXELTABLE_HOME` and any stdout configuration before importing
   Pixeltable. Do not support a live datastore switch.
9. Replace `create_default_idxs` with `has_default_idxs` anywhere a notebook
   adapter still calls the Python API. Remove replica/publish and `pxt.Required`.
10. Replace `pixeltable-new` and template names with `pxt init`, `pxt service
    example`, and the schema/service lifecycle.
11. Keep arbitrary execution, installation, and display in a separate unsafe
    registration path. Enforce both the environment flag and `stdio` transport.
12. Preserve JSON types in row results and enforce stable result-size bounds.
13. Test through an MCP client, not only by calling Python functions or private
    registration managers.

The official [MCP v1-to-v2 migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/) is the source of truth for SDK changes. The [0.1 evidence review](review-0.1.0.md) records the repository-specific failures and acceptance gates.

## Verify the migration

Run with a temporary catalog:

```bash
export PIXELTABLE_MCP_PROJECT_ROOT="$PWD"
export PIXELTABLE_HOME="$(mktemp -d)/catalog"
uv sync --frozen --extra test
uv run pytest -q
PIXELTABLE_DISABLE_STDOUT=1 uv run pytest --run-slow -q
uv run python list_tools.py
uv build
./scripts/run-conformance.sh
```

Then verify through an MCP 2 client:

1. Initialize a session and assert the server reports version 0.2.0.
2. Assert exactly the 18 default tools, four resources, and four prompts are
   discoverable with unsafe mode unset.
3. Check that every tool advertises its input schema, output schema, and
   behavior annotations.
4. Scaffold `app.py`; run schema check, diff, and update; run service check,
   update, list, stop, and prune.
5. Insert deterministic rows, read a bounded page, retrieve one row, induce a
   deterministic computation failure, inspect it, and recompute it.
6. Exercise negative cases and assert MCP error flags and sanitized messages.
7. Start the server as a subprocess, speak MCP over `stdio`, and verify that
   concurrent calls do not corrupt JSON-RPC output.
8. Enable unsafe mode in a disposable process, verify the three tools appear,
   exercise timeout and cleanup, then verify they remain unavailable to the
   HTTP test harness.
9. Install the built wheel into a clean Python 3.11 environment and repeat the
   handshake and core app scenario.
10. Freshly resolve the newest versions still admitted by the declared ranges
    and repeat the compatibility suite.

Two repetitions can reveal a deterministic agent regression but cannot prove
statistical superiority. Provider mocks validate wiring only. Cloud behavior
remains unverified until an explicitly authorized live deployment test is run.

## Rollback

Stop the 0.2 server before starting 0.1. Restore the catalog backup if any 0.2
schema or service mutation was applied. Restore the prior client command,
environment, and pinned 0.1 checkout together; 0.1 cannot run against MCP 2.2
and its table/view wrappers cannot run against Pixeltable 0.7.6. Do not combine
the 0.1 source with the 0.2 dependency lock.
