# Pixeltable Developer MCP Server

<!-- mcp-name: io.github.pixeltable/mcp-server-pixeltable-developer -->

[![PyPI](https://img.shields.io/pypi/v/mcp-server-pixeltable-developer)](https://pypi.org/project/mcp-server-pixeltable-developer/)
[![CI](https://github.com/pixeltable/mcp-server-pixeltable-developer/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/mcp-server-pixeltable-developer/actions/workflows/ci.yml)
[![Glama score](https://glama.ai/mcp/servers/pixeltable/mcp-server-pixeltable-developer/badges/score.svg)](https://glama.ai/mcp/servers/pixeltable/mcp-server-pixeltable-developer)

A local MCP server for building and operating [Pixeltable](https://pixeltable.com/)
applications. Through the `pxt` CLI, an agent scaffolds one `app.py`, applies its
tables to a catalog, inserts and inspects rows, recovers failed computations, and
serves HTTP routes. It is a beta developer tool: it runs with your permissions and
changes the catalog you point it at.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The server is on
[PyPI](https://pypi.org/project/mcp-server-pixeltable-developer/), and `uvx` downloads and runs it.

Claude Code:

```bash
claude mcp add pixeltable -e PIXELTABLE_MCP_PROJECT_ROOT=/absolute/path/to/app -e PIXELTABLE_HOME=/absolute/path/to/.pixeltable -- uvx mcp-server-pixeltable-developer
```

Other clients (Claude Desktop, Cursor, VS Code, and so on):

```json
{
  "mcpServers": {
    "pixeltable": {
      "command": "uvx",
      "args": ["mcp-server-pixeltable-developer"],
      "env": {
        "PIXELTABLE_MCP_PROJECT_ROOT": "/absolute/path/to/app",
        "PIXELTABLE_HOME": "/absolute/path/to/.pixeltable"
      }
    }
  }
}
```

The app directory must exist. An empty one is fine: the server can scaffold `app.py`
there. Restart the client after changing its configuration.

| Variable | Required | Effect |
|---|---|---|
| `PIXELTABLE_MCP_PROJECT_ROOT` | yes | Directory holding `app.py`. File arguments outside it are refused. |
| `PIXELTABLE_HOME` | yes | Catalog the server reads and changes. `~/.pixeltable` is the one the `pxt` CLI uses; use a separate catalog for tests. |
| `PXT_PORT` | no | Port of the server's `pxt` daemon, 22089 by default. One daemon serves one project per port, and a `pxt` command from another project restarts it, so in-flight calls fail with `RemoteDisconnected`. If you run `pxt` for another project at the same time, set a free port such as `22090`. |
| `PIXELTABLE_MCP_ENABLE_UNSAFE` | no | `1` adds tools that run arbitrary Python and install packages. See [Unsafe mode](#unsafe-mode). |
| `PIXELTABLE_MCP_PXT` | no | A specific `pxt` executable. By default the server uses the one installed beside it. |

Both paths are fixed for the life of the process. Run one server per catalog when it
applies schema or service changes.

## Other ways to run it

| Channel | What you get |
|---|---|
| Official MCP Registry | `io.github.pixeltable/mcp-server-pixeltable-developer`. Clients that browse the registry install the PyPI package. |
| [Claude plugin](plugin/) | Uses the open project as the app directory, `~/.pixeltable` as the default catalog, and port 22090. |
| [Glama](https://glama.ai/mcp/servers/pixeltable/mcp-server-pixeltable-developer) | Directory listing with a quality score. |
| Docker | An image built from this repository. See below. |
| A clone | `"command": "uv"`, `"args": ["run", "--directory", "/absolute/path/to/clone", "mcp-server-pixeltable-developer"]` |

Docker: mount the app directory, and keep the catalog in a named volume so it persists.
Do not bind-mount a catalog from a macOS host: Postgres cannot set permissions on that
directory, and `initdb` fails. On macOS, the app path must also be shared with your
container engine, or the mount lands inside its VM.

```bash
docker build -t mcp-server-pixeltable-developer .
docker run --rm -i \
  -e PIXELTABLE_MCP_PROJECT_ROOT=/work/app -e PIXELTABLE_HOME=/work/catalog \
  -v /absolute/path/to/app:/work/app \
  -v pixeltable-catalog:/work/catalog \
  mcp-server-pixeltable-developer
```

## What the server exposes

| Area | Tools |
|---|---|
| Inspect | `pixeltable_list_catalog`, `pixeltable_describe`, `pixeltable_rows`, `pixeltable_get_row`, `pixeltable_errors` |
| Data | `pixeltable_insert_rows`, `pixeltable_recompute` |
| App scaffold | `pixeltable_scaffold_app` |
| Schema lifecycle | `pixeltable_schema_check`, `pixeltable_schema_diff`, `pixeltable_schema_update`, `pixeltable_schema_prune` |
| Service lifecycle | `pixeltable_service_check`, `pixeltable_service_diff`, `pixeltable_service_update`, `pixeltable_service_list`, `pixeltable_service_stop`, `pixeltable_service_prune` |

Read tools carry read-only annotations; mutation tools declare write and destructive
behavior. Prune tools default to dry runs. Failures come back as MCP tool errors, so a
client can retry with corrected input.

| Resource | Content |
|---|---|
| `pixeltable://status` | Server, Pixeltable, and MCP versions, transport, and unsafe-mode state |
| `pixeltable://catalog` | Catalog inventory |
| `pixeltable://guidance/app` | The application workflow below |
| `pixeltable://guidance/cloud` | Cloud preparation steps; the server deploys nothing |

Prompts: `pixeltable_build_app`, `pixeltable_build_rag`, `pixeltable_build_agent`, and
`pixeltable_debug_computation`. They follow
[pixeltable-skill 2.12.0](https://github.com/pixeltable/pixeltable-skill/blob/67f5a4222ce325420c461606c099e9ee0f9c9c02/skills/pixeltable-skill/SKILL.md),
which stays the detailed reference for providers, multimodal views, indexes, tool
calling, and serving.

## Pixeltable workflow

These are the commands the tools wrap:

```bash
pip install 'pixeltable[serve]>=0.7.10,<0.8'
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt service update app.py my_app
pxt service list
```

- `my_app` is a catalog directory, not a folder on disk. `pxt service list` shows the assigned URL.
- In `app.py`, stored columns are annotations and computed columns are assignments.
  Types are non-nullable; write `T | None` for an optional column. Do not use `pxt.Required`.
- To change a computed expression, edit it and update the schema. The update keeps
  existing values, so run `pxt recompute my_app/docs COLUMN -f` to refresh them. A type
  change is unsupported in place; rename the computed column instead.
- Cloud uses the same `app.py`: `pxt login`, a `[[pixeltable.database]]` entry in
  `pixeltable.toml`, and provider keys through `pxt secret set`, never in the TOML.
  `pixeltable://guidance/cloud` lists the order. The server creates no paid resources.

## Unsafe mode

`PIXELTABLE_MCP_ENABLE_UNSAFE=1` adds `pixeltable_unsafe_execute_python`,
`pixeltable_unsafe_install_package`, and `pixeltable_unsafe_display`. They run arbitrary
Python, change the server's environment, read any file the process can, and render
untrusted content in a browser canvas. They exist only over `stdio`: enable them for a
trusted local client in a disposable environment. HTTP is not a supported transport.

## Compatibility

| Component | Supported |
|---|---|
| Python | 3.11 or newer; CI tests 3.11 to 3.14 |
| Pixeltable | `>=0.7.10,<0.8` with the `serve` extra, locked at 0.7.16 |
| MCP Python SDK | `>=2.2,<3`, locked at 2.3.0 |
| Transport | local `stdio` |

## Releases

[CHANGELOG.md](CHANGELOG.md) lists what each version changed. 0.2.1 is the first PyPI release.

To ship a version:

1. Bump the version in `pyproject.toml`, `src/mcp_server_pixeltable_developer/__init__.py`,
   `server.json` (twice), `mcpb/manifest.json`, `plugin/.claude-plugin/plugin.json`
   (version and `uvx` pin), and the assertions in `.github/workflows/ci.yml` and the tests.
   Add a `## X.Y.Z` section to `CHANGELOG.md` and run `uv lock`. The tests fail until
   every copy agrees.
2. Merge, then push the tag `vX.Y.Z`. CI reruns every check, publishes to PyPI with
   trusted publishing, creates the GitHub Release from the changelog section, then
   publishes `server.json` to the MCP Registry with GitHub OIDC.
3. If the registry step fails, rerun it without a new tag:
   `gh workflow run mcp-registry.yml --repo pixeltable/mcp-server-pixeltable-developer --ref main`.

Glama syncs from GitHub. The Docker MCP Catalog entry (`servers/pixeltable/server.yaml`
in docker/mcp-registry) pins a commit, so it needs a PR per release.

## Develop and verify

```bash
uv sync --frozen --extra test --extra canvas
uv run pytest -q
PIXELTABLE_DISABLE_STDOUT=1 uv run pytest -q --run-slow
uv run ruff check . && uv run ruff format --check . && uv run pyright
./scripts/run-conformance.sh
./scripts/build-mcpb.sh
```

`--run-slow` runs real catalogs and services in temporary directories, each on its own
daemon port. `build-mcpb.sh` writes the Claude Desktop bundle to `dist/`. For manual
testing, run the MCP Inspector against the server:
`uv run mcp dev src/mcp_server_pixeltable_developer/server.py:mcp`.

## Privacy Policy

The server runs entirely on the machine that starts it. It collects no telemetry
and no analytics, and it sends nothing to Pixeltable or to any third party.

**What it handles.** The Pixeltable catalog at `PIXELTABLE_HOME` and the
application files under `PIXELTABLE_MCP_PROJECT_ROOT`, both chosen by you. Tool
results are returned to the MCP client that launched the server, which is how
your assistant reads them. Nothing else receives them.

**Storage and retention.** The server stores nothing of its own. Catalog data
stays in `PIXELTABLE_HOME` on your disk, under your control, and is removed when
you remove it.

**Secrets.** Command output is passed through a redactor that masks common
credential forms, including URL passwords and API tokens, before it reaches the
client.

**Third parties.** If your application declares computed columns that call model
providers, Pixeltable makes those calls with the credentials you configured. That
traffic is your application's, not this server's.

**Unsafe mode.** With `PIXELTABLE_MCP_ENABLE_UNSAFE=1`, the added tools execute
arbitrary Python, install packages, and read files available to the server
process. Enable it only on a trusted local machine.

**Contact.** Report privacy questions through
[GitHub issues](https://github.com/pixeltable/mcp-server-pixeltable-developer/issues).

## Documentation

- [Changelog](CHANGELOG.md)
- [Migration from 0.1 to 0.2](docs/migration-0.1-to-0.2.md)
- [0.2.0 implementation review](docs/review-0.2.0.md) and [0.1.0 evidence review](docs/review-0.1.0.md)
- [Agent evaluation protocol](evals/README.md)
- [Pixeltable documentation](https://docs.pixeltable.com/) and [LLM reference](https://www.pixeltable.com/llms.txt)
- [MCP Python SDK 2 documentation](https://py.sdk.modelcontextprotocol.io/v2/)

## License

Apache-2.0. See [LICENSE](LICENSE).
