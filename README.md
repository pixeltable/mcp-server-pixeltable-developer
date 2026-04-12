# Pixeltable MCP Server (Developer Edition)

Multimodal AI data infrastructure as an MCP server. **32 tools · 13 resources · 6 prompts** for table management, AI/ML pipelines, dependency management, an interactive REPL, and more.

Uses **sync endpoints** + **uvloop** for best performance with Pixeltable **≥ 0.5.27** (see `pyproject.toml`).

---

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Claude Code (easiest)** — just say:
> "Install https://github.com/pixeltable/mcp-server-pixeltable-developer as a uv tool and add it to your MCPs"

**Manual install:**

```bash
uv tool install --from git+https://github.com/pixeltable/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer
claude mcp add pixeltable mcp-server-pixeltable-developer   # Claude Code
```

If `uv` warns that `~/.local/bin` is not on your `PATH`, run `uv tool update-shell` (or add that directory to `PATH`) so `mcp-server-pixeltable-developer` is found. Check with `mcp-server-pixeltable-developer --version`.

**From source:**

```bash
git clone https://github.com/pixeltable/mcp-server-pixeltable-developer && cd mcp-server-pixeltable-developer
uv sync
```

### Client Configuration

<details><summary><strong>Claude Desktop</strong></summary>

```json
{
  "mcpServers": {
    "pixeltable": {
      "command": "mcp-server-pixeltable-developer",
      "env": {
        "PIXELTABLE_HOME": "/Users/{you}/.pixeltable",
        "PIXELTABLE_FILE_CACHE_SIZE_G": "10"
      }
    }
  }
}
```

From source — use `"command": "uv"` with `"args": ["run", "--directory", "{repo}", "python", "-m", "mcp_server_pixeltable_stio"]`.

</details>

<details><summary><strong>Cursor</strong></summary>

**User config** — `~/.cursor/mcp.json` (applies to all workspaces):

```json
{
  "mcpServers": {
    "pixeltable-developer": {
      "command": "mcp-server-pixeltable-developer",
      "env": {
        "PIXELTABLE_HOME": "/Users/you/.pixeltable"
      }
    }
  }
}
```

If Cursor reports **command not found**, use the full path from `uv tool update-shell` / `which mcp-server-pixeltable-developer`, e.g. `"command": "/Users/you/.local/bin/mcp-server-pixeltable-developer"`.

**Develop this repo from source** — optional project `.cursor/mcp.json` so the server runs from your clone (replace the path):

```json
{
  "mcpServers": {
    "pixeltable-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-server-pixeltable-developer", "python", "-m", "mcp_server_pixeltable_stio"],
      "env": {
        "PIXELTABLE_HOME": "/Users/you/.pixeltable"
      }
    }
  }
}
```

Do not define the same server name twice (user + project) unless you intend to run two entries.

</details>

Restart your client after configuration changes.

---

## Testing

Use a **dedicated** `PIXELTABLE_HOME` for testing so you do not touch other catalogs.

### MCP Inspector (interactive)

From a clone, after `uv sync`:

```bash
export PIXELTABLE_HOME="$HOME/.pixeltable-mcp-test"
uv run mcp dev src/mcp_server_pixeltable_stio/server.py:mcp
```

This starts the server and opens the **MCP Inspector** in your browser so you can invoke tools, read resources, and try prompts without an IDE.

**Quick checks:** tool `pixeltable_check_dependencies` with expression `openai.chat_completions`; tool `execute_python` with `print(pxt.__version__)`; resource `pixeltable://version`.

### Cursor

Add `PIXELTABLE_HOME` under `env` in `.cursor/mcp.json` (see **Client Configuration** above). Restart Cursor, confirm the server connects, then run a simple tool from the MCP panel.

### CLI sanity (no JSON-RPC)

```bash
mcp-server-pixeltable-developer --version
uv run python list_tools.py
```

`--version` / `--help` exit immediately. `list_tools.py` only prints registered tools, resources, and prompts (import check, not a full MCP session).

---

## Tools (32)

| Category | Tools |
|---|---|
| **Tables** | `create_table` · `drop_table` · `create_view` · `create_snapshot` |
| **Data** | `create_replica` · `query_table` · `insert_data` · `query` · `add_computed_column` |
| **Directories** | `create_dir` · `drop_dir` · `move` |
| **Config** | `configure_logging` · `set_datastore` |
| **AI/ML** | `create_udf` · `create_array` · `create_tools` · `connect_mcp` |
| **Deps** | `check_dependencies` · `install_dependency` |
| **Types** | `create_type` (Image, Video, Audio, Array[Float], …) |
| **Docs** | `search_docs` |
| **REPL** | `execute_python` · `introspect_function` · `list_available_functions` · `install_package` |
| **Logging** | `log_bug` · `log_missing_feature` · `log_success` · `generate_bug_report` · `get_session_summary` |
| **Display** | `display_in_browser` |

All tools are prefixed `pixeltable_` (except REPL/logging helpers). Full docstrings available via `introspect_function`.

## Resources (13)

| URI | What it returns |
|---|---|
| `pixeltable://tables` | All tables with count |
| `pixeltable://tables/{path}` | Info about a table / view / snapshot |
| `pixeltable://tables/{path}/schema` | Column schema |
| `pixeltable://directories` | All directories |
| `pixeltable://ls` / `pixeltable://ls/{path}` | Directory listing |
| `pixeltable://version` | Pixeltable version |
| `pixeltable://config/datastore` | Datastore config |
| `pixeltable://types` | Available data types |
| `pixeltable://functions` | Registered Pixeltable functions |
| `pixeltable://tools` | MCP tool list |
| `pixeltable://help` | Workflow guidance |
| `pixeltable://diagnostics` | System & dependency diagnostics |

## Prompts (6)

`pixeltable_usage_guide` · `getting_started` · `computer_vision_pipeline` · `rag_pipeline` · `video_analysis_pipeline` · `audio_processing_pipeline`

---

## Examples

```text
Create a table called movies with title, year, and rating columns → insert sample data → query ratings above 8.5

Add a computed column that runs YOLOX object detection on every image

Check what deps I need for openai.chat_completions(...) → install them

execute_python("print(pxt.list_tables())")
```

---

## Documentation

- [Pixeltable docs](https://docs.pixeltable.com/)
- [pixeltable-skill](https://github.com/pixeltable/pixeltable-skill/blob/main/skills/pixeltable-skill/SKILL.md) — task router, API pitfalls (`openai.vision` vs `chat_completions`, `frame_iterator`, `similarity(string=...)`, etc.), and workflow examples aligned with current Pixeltable

---

## Architecture

```
src/mcp_server_pixeltable_stio/
  server.py            FastMCP server, tool/resource/prompt registration, uvloop activation
  core/
    tables.py          Table CRUD, views, snapshots, replicas, queries, computed columns
    directories.py     Directory CRUD, listing, moving
    dependencies.py    Dependency checking, unified installer, diagnostics
    udf.py             UDF creation, type system, LLM tool wrappers, MCP connections
    helpers.py         Config, version, docs search, shared utilities
    resources.py       Read-only MCP resource handlers
  prompt.py            Prompt templates for common workflows
  repl_functions.py    Persistent Python REPL, introspection, package management
  canvas_server.py     Browser canvas for rich content display
```

---

## Troubleshooting

- **Restart your client** after any config change
- **Python 3.10+** and **`uv`** are required
- **`command not found` after `uv tool install`:** ensure `~/.local/bin` is on `PATH` (`uv tool update-shell`) or invoke via full path; confirm with `mcp-server-pixeltable-developer --version`
- Check that `PIXELTABLE_HOME` points to a valid directory
- Use `log_bug(...)` / `generate_bug_report()` for structured issue tracking
- File issues at [github.com/pixeltable/mcp-server-pixeltable-developer](https://github.com/pixeltable/mcp-server-pixeltable-developer/issues)
