# Pixeltable MCP server 0.1.0 evidence review

**Review date:** 2026-09-09
**Latest-release recheck:** 2026-09-10
**Repository:** `pixeltable/mcp-server-pixeltable-developer`
**Audited commit:** [`5ae8063e07ea6ad1aa625039e20475986d7a7750`](https://github.com/pixeltable/mcp-server-pixeltable-developer/tree/5ae8063e07ea6ad1aa625039e20475986d7a7750)
**Audited package version:** 0.1.0
**Comparison targets:** Pixeltable 0.7.6, MCP Python SDK 2.2.0, pixeltable-skill 2.8.3

## Verdict

Version 0.1.0 works only with its old lock and is not compatible with the latest
versions allowed by its own package metadata. A fresh install resolves MCP 2.x
and fails while importing the server. If MCP is held back and Pixeltable is
upgraded to 0.7.6, the core table and view creation tools fail because they pass
a removed keyword. The replica/publish tool also targets a feature removed from
Pixeltable.

The frozen environment's tests pass, but they primarily prove that the 0.1
registration surface and a small Pixeltable 0.6.3 workflow remain internally
consistent. They do not prove compatibility with the declared open dependency
ranges, MCP protocol behavior, concurrency safety, or the security of the
optional canvas and execution tools.

The required 0.2 direction is a smaller local `stdio` server built on the MCP 2
`MCPServer` API, Pixeltable 0.7.6's application CLI, typed tool contracts, MCP
error semantics, explicit lifecycle cleanup, and unsafe capabilities that are
disabled by default.

## Reproducibility record

| Item | Recorded value |
|---|---|
| Repository default branch | `main` |
| Repository commit | `5ae8063e07ea6ad1aa625039e20475986d7a7750` |
| Commit timestamp | `2026-08-04T15:29:58-07:00` |
| Pixeltable skill | 2.8.3 at local source commit `f550e6ed757b48635e4f53900840f4e9a1fb4c93` |
| Latest stable Pixeltable | 0.7.6; Python `>=3.11`; uploaded 2026-09-09 17:02:40 UTC |
| Pixeltable wheel SHA-256 | `9e6cdbe54f042b31786bede4a4cd4d68b361b31d75f34ea4e9c415146c901114` |
| Latest stable MCP Python SDK | 2.2.0; Python `>=3.10`; uploaded 2026-09-07 16:06 UTC |
| MCP wheel SHA-256 | `bde982589473a060ae145e3406e9a5333fe538c97229ba841f5a7f92be004f81` |
| MCP source archive SHA-256 | `2dc37ecb1974becdcebdbf7561e7c15a07dbbf20ba21ba16c3593b3038b3afbd` |
| Initial retrieval date | 2026-09-09 |
| Latest-release recheck | 2026-09-10; Pixeltable 0.7.6, MCP SDK 2.2.0, and skill commit `f550e6e` remained current |

Release metadata came from the official [Pixeltable 0.7.6 PyPI release](https://pypi.org/project/pixeltable/0.7.6/) and [MCP 2.2.0 PyPI release](https://pypi.org/project/mcp/2.2.0/). Package implementation evidence was checked against the corresponding upstream tags [`pixeltable/v0.7.6`](https://github.com/pixeltable/pixeltable/tree/v0.7.6) and [`python-sdk/v2.2.0`](https://github.com/modelcontextprotocol/python-sdk/tree/v2.2.0).

The review inspected the published package artifacts, every module in `src/`,
the test suite, CI workflow, package metadata, lock file, README, prompt text,
and canvas. No paid provider call, Cloud deployment, or external MCP mutation
was performed. Provider and Cloud conclusions are source-reviewed only. The
canvas issues were established by code inspection; no browser exploit was run.

## Architecture and 0.1 surface

The 0.1 server builds one module-level `FastMCP` instance and registers 35
tools, 13 resources, and 11 prompts. Most functions call Pixeltable directly.
A second group controls a persistent Python subprocess, installs packages,
scaffolds projects, writes bug logs, searches documentation over HTTP, and
feeds an optional browser canvas.

| Area | Implementation | Main concern |
|---|---|---|
| Registration and startup | [`server.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/server.py#L149-L152), [`__main__.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/__main__.py#L40-L75) | MCP 1 import, no server lifespan, fail-open fallback |
| Catalog and rows | [`core/tables.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/tables.py) | Removed Pixeltable APIs, private internals, unbounded results |
| UDF and MCP helpers | [`core/udf.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/udf.py) | Arbitrary `exec`, open-world URLs, transient in-process objects |
| Python subprocess | [`core/repl_session.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/repl_session.py) | Global mutable state, incomplete timeout and cleanup behavior |
| Dependency installer | [`core/dependencies.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/dependencies.py) | False-positive checks and arbitrary environment mutation |
| Canvas | [`core/canvas_server.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/canvas_server.py), [`canvas.html`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/canvas.html) | Local file disclosure, cross-origin access, raw HTML injection |
| Guidance | [`prompt.py`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/prompt.py), README | Large duplicated examples had already drifted from the skill |

## Executable evidence

### Frozen 0.1 lock

The audited lock contains [MCP 1.11.0](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/uv.lock#L1290-L1308) and [Pixeltable 0.6.3](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/uv.lock#L2327-L2334). The following commands ran successfully on macOS with Python 3.10:

```bash
uv sync --frozen --extra test --extra canvas
uv run python list_tools.py
uv run pytest -q
PIXELTABLE_DISABLE_STDOUT=1 uv run pytest --run-slow -q
```

Observed results:

- Inventory: 35 tools, 13 resources, 11 prompts.
- Fast suite: `35 passed, 3 skipped in 0.96s`.
- Slow-inclusive suite: `38 passed in 1.85s`.

This is useful regression evidence for 0.1 behavior. It is not latest-version
evidence because CI resolves the checked-in lock and the slow tests exercise
only create, insert, query, drop, a document-splitter view, and an invalid type.
The tests themselves pin the old surface through private manager access and
fixed counts ([`test_primitives.py:20-24`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/tests/test_primitives.py#L20-L24), [`test_smoke_pxt.py:13-88`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/tests/test_smoke_pxt.py#L13-L88)).

### Latest MCP probe

An isolated Python 3.12 environment installed `mcp==2.2.0`,
`pixeltable==0.7.6`, and the runtime dependencies, then ran:

```bash
PYTHONPATH=src python -c 'import mcp_server_pixeltable_stio.server'
```

Observed result:

```text
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x,
where FastMCP was renamed to MCPServer
```

The failure is deterministic. The server imports the old class at
[`server.py:22`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/server.py#L22), while MCP 2.2 deliberately raises from the old module and directs users to `mcp.server.mcpserver.MCPServer` ([upstream source](https://github.com/modelcontextprotocol/python-sdk/blob/v2.2.0/src/mcp/server/fastmcp.py#L1-L16), [migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver)).

### Latest Pixeltable probes

With the same isolated environment and a separate `PIXELTABLE_HOME`, invoking
the 0.1 wrapper produced:

```text
pixeltable=0.7.6
{'success': False, 'error': "create_table() got an unexpected keyword argument 'create_default_idxs'"}
{'success': False, 'error': "create_view() got an unexpected keyword argument 'create_default_idxs'"}
```

Pixeltable 0.7.6 exposes `has_default_idxs` on
[`create_table`](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/globals.py#L64-L80) and [`create_view`](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/globals.py#L302-L314). The 0.1 wrappers always pass `create_default_idxs` ([table call](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/tables.py#L170-L227), [view call](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/tables.py#L331-L385)).

The same environment reported `hasattr(pxt, "replicate") == False` and
`hasattr(pxt, "publish") == False`. Pixeltable's v51 metadata migration states
that replica support was removed ([upstream source](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/metadata/converters/convert_50.py#L64-L73)); the 0.1 tool calls both removed functions ([`tables.py:473-509`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/tables.py#L473-L509)).

## Claim matrix

| 0.1 claim or behavior | Classification against latest | Evidence |
|---|---|---|
| `pixeltable>=0.6.3` and `mcp[cli]>=1.10.0` are sufficient | **Incorrect** | Open ranges in [`pyproject.toml:26-33`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/pyproject.toml#L26-L33) resolve MCP 2.2, which fails import; Pixeltable 0.7.6 breaks create wrappers |
| Python 3.10 is supported | **Outdated** | 0.1 metadata and CI include 3.10; Pixeltable 0.7.6 requires Python 3.11 or newer |
| Table and view creation work on the allowed Pixeltable range | **Incorrect** | Reproduced keyword failures above |
| Replica/publish is a supported data operation | **Incorrect** | Pixeltable 0.7.6 removed replica support |
| `pxt.Required[T]` is preferred | **Outdated** | Pixeltable marks `Required` deprecated; skill 2.8.3 says types are non-nullable and uses `T | None` for optional columns ([Pixeltable source](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/type_system.py#L1858-L1863)) |
| `pixeltable-new` and `pxt serve` are the starting workflow | **Outdated** | Skill 2.8.3 starts with `pxt init`, `pxt service example`, `pxt schema update`, and `pxt service update` |
| Returning `{"success": false}` communicates an MCP failure | **Incorrect** | MCP 2 documents that returned values have `is_error=false`; a recoverable tool failure must raise `ToolError` ([official guidance](https://py.sdk.modelcontextprotocol.io/v2/servers/handling-errors/)) |
| Resource and tool inventory through private managers is stable | **Unsupported** | 0.1 reads `_tool_manager` directly; MCP 2 keeps public `list_tools`, `list_resources`, `list_resource_templates`, and `list_prompts` methods ([migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/#what-is-unchanged-on-mcpserver)) |
| The optional canvas is safe because it binds to loopback | **Incorrect** | It serves arbitrary absolute files with wildcard cross-origin access and renders raw HTML |
| Importing a provider module proves its runtime dependency is present | **Incorrect** | Pixeltable defers OpenAI and Anthropic imports until client creation and Hugging Face imports until execution |

## Prioritized findings

### P0 — Fresh installs fail on MCP 2

The package declares no MCP upper bound, yet imports MCP 1's `FastMCP` class.
The README's one-shot Git install ignores the old lock, so the normal fresh
installation path selects MCP 2.2.0 and fails before registering any primitive.
The fallback server repeats the removed import
([`__main__.py:61-75`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/__main__.py#L61-L75)).

**Required correction:** migrate to `MCPServer`, MCP 2 context and exception
types, public listing APIs, and MCP 2 client test helpers; constrain the package
to `mcp>=2.2,<3`. Fail startup with a nonzero exit if initialization fails.

**Verification:** install from the built wheel into a clean Python 3.11
environment with the current resolver; initialize an in-memory MCP client and a
subprocess `stdio` client; list and invoke all primitive kinds; assert the
reported server name and version.

### P0 — Core create operations fail on Pixeltable 0.7.6

Both create wrappers pass the removed `create_default_idxs` argument on every
call, including when the caller accepts the default. This makes first-use table
and view scenarios fail on the latest package.

**Required correction:** the 0.2 surface should use the application CLI rather
than recreate its schemas with an ad hoc wrapper. If a notebook compatibility
adapter remains, rename the option to `has_default_idxs` and test it directly
against 0.7.6. Raise typed `ToolError` failures instead of returning a success
envelope containing an error.

**Verification:** in an isolated catalog, scaffold an app, run schema check and
diff, apply it, list and describe the resulting table, insert rows, read rows,
and run the same flow for an iterator view.

### P0 — Replica/publish exposes removed behavior

`pixeltable_create_replica` is registered as a normal data tool, but both
implementation paths use APIs absent from Pixeltable 0.7.6. Its exception path
also reverses the practical source/destination diagnosis, which makes recovery
harder.

**Required correction:** remove the tool, its prompt guidance, and its tests.
Cloud preparation should use the current `pxt db`, `pxt schema`, and `pxt
service` workflow documented by Pixeltable 0.7.6. Do not invent a replacement
replica API.

**Verification:** the 0.2 tool list contains no replica/publish operation or
mention, and Cloud guidance contains only released CLI commands.

### P0 — Canvas can disclose local files and execute untrusted markup

When canvas mode is enabled, `/media/{file_path}` converts the route argument
to an absolute path and returns any readable file
([`canvas_server.py:139-147`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/canvas_server.py#L139-L147)). The event stream allows every browser origin
([`canvas_server.py:89-103`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/canvas_server.py#L89-L103)). The browser inserts arbitrary HTML with `innerHTML`
([`canvas.html:180-202`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/canvas.html#L180-L202)) and writes image URLs into a new document ([`canvas.html:304-319`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/canvas.html#L304-L319)). The tool rewrites `file://` URLs to that endpoint and hard-codes port 7777 ([`server.py:315-328`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/server.py#L315-L328)).

**Required correction:** keep display absent from the default interface. In
unsafe local mode, serve only explicitly registered files beneath allowlisted
temporary or Pixeltable media roots through opaque one-use handles. Use an
unpredictable session token, exact-origin policy, a restrictive content
security policy, safe DOM APIs, bounded queues, disconnect cleanup, a readiness
check, and lifespan shutdown. Do not accept raw HTML. Keep the capability
unavailable to HTTP transports.

**Verification:** tests must reject absolute paths, `..` traversal, encoded
traversal, cross-origin reads, raw script/event-handler content, stale handles,
and access without the session token.

### P0 — Normal tools provide arbitrary host-code execution

`pixeltable_create_udf` calls `exec` with full builtins
([`udf.py:43-80`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/udf.py#L43-L80)). `pixeltable_add_computed_column` evaluates caller-supplied Python in the server process ([`tables.py:696-757`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/tables.py#L696-L757)); because its globals omit `__builtins__`, Python inserts builtins automatically ([Python `eval` specification](https://docs.python.org/3/library/functions.html#eval)). The REPL and two installers add further arbitrary execution and environment mutation.

**Required correction:** remove string evaluation, UDF creation, and dependency
installation from the default surface. Use the application file plus schema
commands for reviewed code. Gate the three explicit unsafe tools behind
`PIXELTABLE_MCP_ENABLE_UNSAFE=1`, annotate them as destructive and open-world,
allow them only on local `stdio`, and describe their host-level authority in
the status resource and README.

**Verification:** unsafe tools are absent with the environment variable unset,
present only when it is exactly enabled, and rejected for any non-`stdio`
transport. Default tools contain no `eval`, `exec`, shell, or package-install
path.

### P1 — Startup fails open and process resources lack a lifecycle

The launcher catches every startup exception and starts an empty fallback
server, making a broken server look connected. The main server also catches
startup exceptions and proceeds to `mcp.run()`
([`server.py:420-451`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/server.py#L420-L451)). A global REPL subprocess has a cleanup method but is never tied to MCP shutdown. The canvas thread has no stop handle.

MCP 2's [lifespan API](https://py.sdk.modelcontextprotocol.io/v2/servers/lifespan/) runs once around the server and provides the intended startup/shutdown boundary.

**Required correction:** initialize dependency and catalog state in an
`MCPServer` lifespan; expose it through typed context; close subprocesses,
threads, clients, and temporary resources in `finally`. Delete the empty-server
fallback and exit nonzero on incompatible dependencies or failed startup.

**Verification:** inject startup failure and assert nonzero exit; start and stop
the server repeatedly and assert no subprocess, thread, port, or temporary file
remains.

### P1 — MCP contracts do not express behavior or failure correctly

The 0.1 registration applies `mcp.tool()` without titles, annotations, or
result models ([`server.py:229-278`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/server.py#L229-L278)). Inputs use broad strings and `Dict[str, Any]`. Many handlers encode failure as a returned dictionary, so the protocol marks the tool call successful.

MCP 2 treats type hints as the input contract, supports `Literal` and Pydantic
constraints, derives structured output schemas, and defines read-only,
open-world, destructive, and idempotent annotations
([tools](https://py.sdk.modelcontextprotocol.io/v2/servers/tools/), [structured output](https://py.sdk.modelcontextprotocol.io/v2/servers/structured-output/)). It requires `ToolError` for recoverable execution failures and reserves `MCPError` for protocol/server-state rejection ([error handling](https://py.sdk.modelcontextprotocol.io/v2/servers/handling-errors/)).

**Required correction:** give every 0.2 tool a constrained input model, typed
result model, stable title and description, and reviewed annotations. Raise
`ToolError` for bad paths, missing objects, CLI validation failures, and row
errors. Raise `MCPError` only when the server cannot honor a valid request.
Raise `ResourceNotFoundError` for missing resources.

**Verification:** inspect `tools/list` for exact schemas and annotations; call
each negative case and assert `is_error=true` with no structured success body;
fuzz enum, path, limit, and row inputs.

### P1 — Global I/O and REPL state are unsafe under concurrency

Several handlers replace process-global `sys.stdout` and `sys.stderr`
([`helpers.py:33-56`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/helpers.py#L33-L56)). MCP 2 runs synchronous handlers in worker threads, so simultaneous calls can capture each other's output or restore streams in the wrong order ([migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/#sync-handler-functions-now-run-on-a-worker-thread)). The environment switch intended to suppress Pixeltable output occurs after Pixeltable modules are imported.

The persistent REPL serializes calls after construction, but global lazy
creation has no lock. Its wait helper returns silently on timeout, and the
result parser can report success without observing the completion marker
([`repl_session.py:121-148`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/repl_session.py#L121-L148)). Long-running code can therefore continue after the tool reports success and interfere with later calls.

**Required correction:** never swap global streams in handlers. Configure
Pixeltable before import, write diagnostics through logging, serialize catalog
mutations where required, and make unsafe execution use a hard timeout that
kills and replaces the worker. Cap output and make session ownership explicit.

**Verification:** run overlapping read and mutation calls while checking that
the `stdio` JSON stream remains valid; force a timeout and assert the process is
terminated, the call is an MCP error result, and the next execution starts from
a clean worker.

### P1 — Datastore switching is stateful and misleading

`pixeltable_set_datastore` creates a directory, writes an MCP-specific TOML
file, and calls `pxt.init()` after Pixeltable may already be initialized
([`helpers.py:160-187`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/helpers.py#L160-L187)). Even reading the custom config creates it when absent ([`config.py:13-67`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/config.py#L13-L67)). The REPL subprocess may retain the old environment.

**Required correction:** remove live catalog switching. Read
`PIXELTABLE_HOME` once before importing Pixeltable, report the effective home in
`pixeltable://status`, and require a process restart to change it. Resource
reads must have no filesystem side effects.

**Verification:** two processes pointed at distinct temporary homes never see
each other's catalogs; changing the parent environment after startup does not
silently switch the active catalog.

### P1 — Dependency checks can report unavailable providers as ready

The checker treats successful import of `pixeltable.functions.openai`,
`anthropic`, or `huggingface` as proof that the provider dependency is installed
([`dependencies.py:146-211`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/dependencies.py#L146-L211)). Pixeltable deliberately delays provider imports until client creation or function execution ([OpenAI](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/functions/openai.py#L34-L45), [Anthropic](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/functions/anthropic.py#L20-L35), [Hugging Face](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/functions/huggingface.py#L63-L79)). The handwritten mapping also probes noncanonical module names such as `pixeltable.functions.google`.

**Required correction:** remove automatic provider installation from the
default server. The status resource may report core versions and missing
optional distributions without claiming credentials or runtime readiness.
Provider setup guidance should come from pixeltable-skill 2.8.3 and the
provider's actual error.

**Verification:** test a clean environment without provider packages, one with
a package but no credential, and one with a mocked provider. Each state must be
reported distinctly.

### P2 — Results, resources, and inventory lose fidelity

Rows are converted through generic stringification, including JSON and media,
and query limits have no safe default or maximum. Resource handlers return JSON
text without declaring an `application/json` MIME type
([`resources.py:1-19`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/resources.py#L1-L19)). Tool inventory reads the private `_tool_manager`
([`helpers.py:380-446`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/src/mcp_server_pixeltable_stio/core/helpers.py#L380-L446)).

**Required correction:** return typed, JSON-preserving row models with column
projection, a positive bounded default/max limit, and stable pagination. Expose
media through safe metadata or scoped resource handles. Use MCPServer's public
listing methods. Declare resource MIME types and keep reads free of mutations.

**Verification:** round-trip nulls, JSON objects, arrays, timestamps, UUIDs,
binary metadata, and media metadata; reject zero/negative/oversized limits; run
pagination without duplicates or omissions.

### P2 — Prompts and scaffolding duplicate stale guidance

The 1,000-line prompt module repeats large Pixeltable examples. It contains
deprecated `Required` types, examples that store provider response JSON without
extracting text, positional similarity calls, a video transcription example
that passes video where audio is expected, and a UDF that mutates another table.
Regex-only prompt tests cannot catch those runtime errors. The scaffold module
wraps `pixeltable-new` and describes `pxt serve`, while skill 2.8.3 starts from
`pxt init` and `pxt service example`.

**Required correction:** replace the template catalog with one
`pixeltable_scaffold_app` tool around the released Pixeltable CLI. Reduce
prompts to `pixeltable_build_app`, `pixeltable_build_rag`,
`pixeltable_build_agent`, and `pixeltable_debug_computation`. Link to the skill
for detailed provider and multimodal recipes rather than copying them.

**Verification:** execute every fenced Python and shell example in a temporary
project, then run `pxt schema check`; run deterministic app, RAG, agent wiring,
and failed-computation scenarios without paid calls.

### P2 — CI validates the lock, not the supported range

The 0.1 workflow tests Python 3.10–3.12 and runs `uv sync` against the committed
lock ([`ci.yml:15-53`](https://github.com/pixeltable/mcp-server-pixeltable-developer/blob/5ae8063e07ea6ad1aa625039e20475986d7a7750/.github/workflows/ci.yml#L15-L53)). That is why a green build did not detect the MCP 2 or Pixeltable 0.7.6 breaks. There are no end-to-end MCP client tests, subprocess wire test, lifecycle test, static type check, lint check, package build check, or canvas security tests.

**Required correction:** test Python 3.11 through currently supported releases,
the frozen lock, minimum dependency versions, and freshly resolved latest
allowed versions. Add in-memory MCP contract tests, a subprocess `stdio`
session, isolated Pixeltable integration, security cases for unsafe mode, lint,
type checking, and wheel/sdist verification.

**Verification:** CI must fail when either dependency range resolves to an
unsupported major, when stdout corrupts JSON-RPC, when tool annotations or
schemas drift, or when the built artifacts omit documentation or compatibility
shims.

## Required 0.2 contract

The reviewed 0.2 surface is intentionally small.

### Default tools

1. `pixeltable_list_catalog`
2. `pixeltable_describe`
3. `pixeltable_rows`
4. `pixeltable_get_row`
5. `pixeltable_errors`
6. `pixeltable_insert_rows`
7. `pixeltable_recompute`
8. `pixeltable_scaffold_app`
9. `pixeltable_schema_check`
10. `pixeltable_schema_diff`
11. `pixeltable_schema_update`
12. `pixeltable_schema_prune`
13. `pixeltable_service_check`
14. `pixeltable_service_diff`
15. `pixeltable_service_update`
16. `pixeltable_service_list`
17. `pixeltable_service_stop`
18. `pixeltable_service_prune`

### Default resources and prompts

Resources: `pixeltable://status`, `pixeltable://catalog`,
`pixeltable://guidance/app`, and `pixeltable://guidance/cloud`.

Prompts: `pixeltable_build_app`, `pixeltable_build_rag`,
`pixeltable_build_agent`, and `pixeltable_debug_computation`.

### Unsafe opt-in

Only `PIXELTABLE_MCP_ENABLE_UNSAFE=1` may register
`pixeltable_unsafe_execute_python`, `pixeltable_unsafe_install_package`, and
`pixeltable_unsafe_display`. They must remain local-`stdio` only.

The production transport for 0.2 is `stdio`. Streamable HTTP may be used by the
test harness, but it is not a supported deployment path. The MCP 2 SDK describes
`stdio` as the local transport, Streamable HTTP as the deployment transport, and
legacy SSE as superseded ([official transport guidance](https://py.sdk.modelcontextprotocol.io/v2/run/)). If a future version supports HTTP, it must implement MCP authorization, transport security, body and session limits, and per-principal isolation first; MCP's authorization layer does not protect `stdio` ([official authorization guidance](https://py.sdk.modelcontextprotocol.io/v2/run/authorization/)).

## 0.2 release gates

The upgrade is complete only when all of the following are true:

1. A clean install resolves Python 3.11+, Pixeltable `>=0.7.6,<0.8`, and MCP
   `>=2.2,<3`, and the console entry point starts without fallback.
2. In-memory and subprocess clients observe exactly the reviewed default tools,
   resources, and prompts, with typed schemas and accurate annotations.
3. An isolated application can be scaffolded, checked, diffed, applied, served,
   listed, stopped, and pruned through the tools.
4. Catalog reads, row insertion, failed-computation inspection, and
   `errors_only` recomputation work against deterministic fixtures.
5. Negative calls produce MCP `is_error=true`; missing resources produce
   protocol resource errors; unexpected exceptions are sanitized to clients
   and logged server-side.
6. Unsafe tools are absent by default and restricted to local `stdio` when
   enabled. Canvas traversal, origin, token, markup, and queue tests pass.
7. Concurrent calls do not replace global streams, corrupt the wire, share an
   unsafe execution session accidentally, or leak subprocesses and threads.
8. Every README and prompt command is executed in CI against Pixeltable 0.7.6.
9. CI passes with the frozen lock, minimum constraints, and a fresh resolution
   of the latest versions admitted by the declared ranges.
10. The wheel and sdist contain the canonical
    `mcp_server_pixeltable_developer` package, a temporary 0.1 import shim if
    promised, README, license, review, migration guide, and evaluation fixtures.

Two repetitions of an agent scenario can catch repeatable regressions but do
not establish statistical superiority. Provider mocks prove wiring only. Cloud
and provider behavior was not live-tested. Cloud guidance remains source-reviewed
until a separately authorized live deployment test is run.
