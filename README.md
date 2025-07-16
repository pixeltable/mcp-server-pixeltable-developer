# Pixeltable stio MCP Server

**Claude meets Pixeltable.** The AI multimodal database, now available as an MCP server.

## ⚡ Quick Start

```bash
git clone https://github.com/goodlux/mcp-server-pixeltable
cd mcp-server-pixeltable
uv sync
```

Add to your Claude Desktop config:

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