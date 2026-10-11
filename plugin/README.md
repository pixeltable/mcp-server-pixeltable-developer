# Pixeltable Developer (Local MCP)

The backend agents build with, on your machine: multimodal database, orchestration, and serving in one file.

This plugin runs the Pixeltable developer MCP server for the project you open. The server scaffolds `app.py`, inspects the catalog (tables, schemas, rows, and computation errors), inserts rows, recomputes columns, and checks, diffs, applies, and prunes schema and service changes. For the full Pixeltable skill, also install the **Pixeltable** plugin.

## What it runs

- `uvx` downloads `mcp-server-pixeltable-developer` 0.2.1 from PyPI and runs it on your machine over stdio. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
- The app directory is the project you opened. The catalog is `~/.pixeltable`, the one the `pxt` CLI uses; to use another, set `PIXELTABLE_HOME` in the environment you start Claude Code from.
- The server runs its own `pxt` daemon on port 22090, so it does not restart the daemon that your terminal's `pxt` uses on port 22089.
- Mutation tools change the catalog and start or stop local services. The tools that run arbitrary Python or install packages stay off: the plugin does not set `PIXELTABLE_MCP_ENABLE_UNSAFE`.

It runs in Claude Code and in Cowork sessions on your computer. Chat on claude.ai does not run local MCP servers.

## Privacy

The server runs locally, collects no telemetry, and sends nothing to Pixeltable or to any third party. If your application's computed columns call model providers, Pixeltable makes those calls with the credentials you configured. Full policy: [Privacy Policy](https://github.com/pixeltable/mcp-server-pixeltable-developer#privacy-policy).

## License

Apache 2.0
