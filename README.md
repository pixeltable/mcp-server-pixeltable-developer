# Pixeltable MCP Server (Developer Edition)

**⚠️ WARNING: THIS IS AN EXPERIMENTAL DEVELOPMENT TOOL. DO NOT USE WITH IMPORTANT DATA. THIS IS FOR DEMO PURPOSES ONLY UNTIL FURTHER NOTICE.**

**Claude meets Pixeltable.** Multimodal AI data infrastructure — not (just) a database — now available as an MCP server.

The server exposes **32 tools**, **13 resources**, and **6 prompts** across table management, directory operations, AI/ML integration, dependency management, an interactive REPL, and more.

## ⚡ Quick Start

### Prerequisites

You must have `uv` installed. If you don't have it or aren't sure, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or consult the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Claude Code Installation (Easiest!)

Just tell Claude:
> "Install https://github.com/pixeltable/mcp-server-pixeltable-developer as a uv tool and add it to your MCPs"

That's it! Claude will handle the installation and configuration for you.

### Manual Installation with uv tool

```bash
# Install as a global tool
uv tool install --from git+https://github.com/pixeltable/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer

# Add to Claude Code
claude mcp add pixeltable mcp-server-pixeltable-developer

# Update to latest version
uv tool install --force --from git+https://github.com/pixeltable/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer
```

### Installation from source (For development)

```bash
git clone https://github.com/pixeltable/mcp-server-pixeltable-developer
cd mcp-server-pixeltable-developer
uv sync
```

### Configuration for Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "pixeltable": {
      "command": "mcp-server-pixeltable-developer",
      "env": {
        "PIXELTABLE_HOME": "/Users/{your-username}/.pixeltable",
        "PIXELTABLE_FILE_CACHE_SIZE_G": "10"
      }
    }
  }
}
```

Or if running from source:

```json
{
  "mcpServers": {
    "pixeltable": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "{path-to-your-repo}",
        "python",
        "-m",
        "mcp_server_pixeltable_stio"
      ],
      "env": {
        "PIXELTABLE_HOME": "/Users/{your-username}/.pixeltable",
        "PIXELTABLE_FILE_CACHE_SIZE_G": "10"
      }
    }
  }
}
```

**⚠️ Note**: Restart Claude Desktop after adding the MCP server configuration.

### Configuration for Cursor

Add to your Cursor MCP configuration (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "pixeltable-developer": {
      "command": "mcp-server-pixeltable-developer"
    }
  }
}
```

For development/source installations:

```json
{
  "mcpServers": {
    "pixeltable-developer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "{path-to-your-repo}",
        "python",
        "-m",
        "mcp_server_pixeltable_stio"
      ]
    }
  }
}
```

---

## 🛠️ Available Tools (32)

### Table Management
| Tool | Description |
|---|---|
| `pixeltable_create_table` | Create a new base table with schema |
| `pixeltable_drop_table` | Drop a table, view, or snapshot |
| `pixeltable_create_view` | Create a view of an existing table |
| `pixeltable_create_snapshot` | Create a snapshot of an existing table |

### Data Operations
| Tool | Description |
|---|---|
| `pixeltable_create_replica` | Create a replica of a table |
| `pixeltable_query_table` | Execute a simple query on a table |
| `pixeltable_insert_data` | Insert data into a table |
| `pixeltable_query` | Generic query with column selection and limits |
| `pixeltable_add_computed_column` | Add a computed column with auto-dependency management |

### Directory Management
| Tool | Description |
|---|---|
| `pixeltable_create_dir` | Create a directory |
| `pixeltable_drop_dir` | Remove a directory |
| `pixeltable_move` | Move or rename a schema object |

### Configuration
| Tool | Description |
|---|---|
| `pixeltable_configure_logging` | Configure logging level and output |
| `pixeltable_set_datastore` | Change the Pixeltable datastore location |

### AI/ML Integration
| Tool | Description |
|---|---|
| `pixeltable_create_udf` | Create a User Defined Function from code |
| `pixeltable_create_array` | Create array expressions |
| `pixeltable_create_tools` | Wrap UDFs for LLM tool-calling |
| `pixeltable_connect_mcp` | Connect to an external MCP server |

### Dependencies
| Tool | Description |
|---|---|
| `pixeltable_check_dependencies` | Check what packages an expression needs |
| `pixeltable_install_dependency` | Install a dependency by name, hint, or expression |

### Type System
| Tool | Description |
|---|---|
| `pixeltable_create_type` | Get a Pixeltable type (Image, Video, Array[Float], etc.) |

### Documentation
| Tool | Description |
|---|---|
| `pixeltable_search_docs` | Search Pixeltable docs via Mintlify |

### REPL & Debug
| Tool | Description |
|---|---|
| `execute_python` | Execute Python in a persistent REPL session |
| `introspect_function` | Get docs, signature, and details for any function |
| `list_available_functions` | List all available functions in a module |
| `install_package` | Install a Python package in the REPL environment |

### Bug Logging
| Tool | Description |
|---|---|
| `log_bug` | Log a bug or issue |
| `log_missing_feature` | Log a missing feature request |
| `log_success` | Log a successful operation |
| `generate_bug_report` | Generate a summary report of all logged bugs |
| `get_session_summary` | Get a summary of the current testing session |

### Display
| Tool | Description |
|---|---|
| `display_in_browser` | Send content (images, tables, HTML, Mermaid) to the browser canvas |

---

## 📚 Resources (13)

Resources are **read-only** endpoints exposed via URIs. They provide data without consuming tool-call tokens.

| URI | Description |
|---|---|
| `pixeltable://tables` | List all tables with count |
| `pixeltable://tables/{path}/schema` | Get a table's column schema |
| `pixeltable://tables/{path}` | Get info about a specific table/view/snapshot |
| `pixeltable://directories` | List all directories |
| `pixeltable://ls` | List root directory contents |
| `pixeltable://ls/{path}` | List a directory's contents |
| `pixeltable://version` | Get Pixeltable version info |
| `pixeltable://config/datastore` | Get datastore configuration |
| `pixeltable://types` | Get available data types |
| `pixeltable://functions` | List all registered Pixeltable functions |
| `pixeltable://tools` | List all MCP tools with descriptions |
| `pixeltable://help` | Get comprehensive help and workflow guidance |
| `pixeltable://diagnostics` | Get system diagnostics and dependency status |

---

## 📝 Prompts (6)

Pre-built prompt templates for common workflows:

| Prompt | Description |
|---|---|
| `pixeltable_usage_guide` | Comprehensive guide for multimodal AI data workflows |
| `getting_started` | Step-by-step guide for first-time users |
| `computer_vision_pipeline` | Build a CV pipeline with YOLOX / GPT-4 Vision |
| `rag_pipeline` | Build a RAG pipeline with chunking, embeddings, search |
| `video_analysis_pipeline` | Build a video analysis pipeline with frame extraction |
| `audio_processing_pipeline` | Build an audio processing pipeline with Whisper |

---

## 💡 Examples

**Create and populate a table:**
```
Create a table called movies with title, year, and rating columns
Insert some sample data
Query all movies with rating above 8.5
```

**Computed columns and AI models:**
```
Add a computed column that generates embeddings for each title
Add object detection to all images using YOLOX
Transcribe audio files with Whisper
```

**Local AI analysis:**
```
Use Ollama to analyze these images
Generate embeddings for semantic search
Run YOLOX object detection on my photos
```

**Data workflows:**
```
Show me all images with cars detected
Find documents mentioning "AI"
Create a summary of this video
```

**Dependency management:**
```
Check what dependencies I need for openai.vision(...)
Install the openai dependency
```

**Interactive REPL:**
```
execute_python("tables = pxt.list_tables(); print(f'Found {len(tables)} tables')")
introspect_function("pxt.create_table")
list_available_functions("pxt")
```

---

## 🏗️ Architecture

The server is organized into modular components under `src/mcp_server_pixeltable_stio/`:

```
core/
  tables.py          — table CRUD, views, snapshots, replicas, queries, inserts, computed columns
  directories.py     — directory CRUD, listing, browsing, moving
  dependencies.py    — dependency checking, unified installer, diagnostics
  udf.py             — UDF creation, array helpers, type system, LLM tool wrappers, MCP connection
  helpers.py         — shared utilities, config, version, types, docs, search
  resources.py       — MCP resource handlers (read-only, JSON responses)
server.py            — FastMCP server definition, tool/resource/prompt registration
prompt.py            — prompt templates for common workflows
repl_functions.py    — persistent Python REPL, introspection, package management
bug_logger.py        — structured bug/feature/success logging
canvas_server.py     — browser canvas for rich content display
```

---

## 🔧 Troubleshooting

### Claude Desktop Issues
1. Restart Claude Desktop after adding the MCP server configuration
2. Check that the path to your Pixeltable home directory is correct
3. Ensure you have the latest version of Claude Desktop
4. Verify that `uv` is installed and accessible from your PATH

### Cursor Issues
1. Make sure you have MCP support enabled in Cursor settings
2. Restart Cursor after configuration changes
3. Check the Cursor logs for any error messages

### Installation Issues
1. Ensure you have Python 3.10+ installed
2. Make sure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Try installing from source if the GitHub installation fails

### Getting Help
1. Use the built-in bug logging: `log_bug("description", severity="high")`
2. Check the generated bug report: `generate_bug_report()`
3. File an issue on the [GitHub repository](https://github.com/pixeltable/mcp-server-pixeltable-developer/issues)

---

Built while having coffee. ☕
