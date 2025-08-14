# Pixeltable MCP Server (Developer Edition)

**Claude meets Pixeltable.** The AI multimodal database, now available as an MCP server.

## ⚡ Quick Start

### Installation with uv tool (Recommended for Claude Code)

```bash
# Install as a global tool
uv tool install --from git+https://github.com/goodlux/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer

# Update to latest version
uv tool install --force --from git+https://github.com/goodlux/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer
```

### Installation from source (For development)

```bash
git clone https://github.com/goodlux/mcp-server-pixeltable-developer
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

## 💡 Examples

**Create and populate a table:**
```
Claude: Create a table for my screenshots
Claude: Add object detection to all images
Claude: Transcribe any audio files with Whisper
```

**Local AI analysis:**
```
Claude: Use Ollama to analyze these images
Claude: Generate embeddings for semantic search
Claude: Run YOLOX object detection on my photos
```

**Data workflows:**
```
Claude: Show me all images with cars detected
Claude: Find documents mentioning "AI"
Claude: Create a summary of this video
```

## 🚀 New Features

### Configurable Datastore Path
Change where Pixeltable stores its data:
```
Claude: Set datastore to ~/my-pixeltable-data
Claude: Get current datastore configuration
```

The datastore path can be configured through:
1. Environment variable `PIXELTABLE_HOME` (highest priority)
2. Persistent configuration file (survives restarts)
3. System default `~/.pixeltable`

### Interactive Python REPL
Execute Python code with PixelTable pre-loaded in a persistent session:

```
Claude: execute_python("tables = pxt.list_tables(); print(f'Found {len(tables)} tables')")
Claude: introspect_function("pxt.create_table")  # Get docs and signature
Claude: list_available_functions("pxt")          # Discover PixelTable functions
```

### Bug Logging & Testing
Structured logging for testing and bug discovery:

```
Claude: log_bug("Cannot save images", severity="high", function_name="pxt.create_table")
Claude: log_missing_feature("No image resize function", use_case="Standardize image sizes")
Claude: log_success("Table creation works", approach="Used schema parameter")
Claude: generate_bug_report()  # Get summary of all issues
```

Bug logs are saved to `pixeltable_testing_logs/` in both Markdown and JSON formats.

### Why These Features?
- **REPL**: Explore PixelTable dynamically without rebuilding the MCP
- **Introspection**: Discover functions and get docs on-demand
- **Bug Logging**: Document issues systematically during development
- **Future-proof**: Adapts to PixelTable API changes automatically

---

Built while having coffee. ☕