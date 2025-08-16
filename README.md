# Pixeltable MCP Server (Developer Edition)

**⚠️ WARNING: THIS IS AN EXPERIMENTAL DEVELOPMENT TOOL. DO NOT USE WITH IMPORTANT DATA. THIS IS FOR DEMO PURPOSES ONLY UNTIL FURTHER NOTICE.**

**Claude meets Pixeltable.** Multimodal AI data infrastructure - not (just) a database - now available as an MCP server.

## ⚡ Quick Start

### Prerequisites

You must have `uv` installed. If you don't have it or aren't sure, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or consult the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Claude Code Installation (Easiest!)

Just tell Claude:
> "Install https://github.com/goodlux/mcp-server-pixeltable-developer as a uv tool and add it to your MCPs"

That's it! Claude will handle the installation and configuration for you.

### Manual Installation with uv tool

```bash
# Install as a global tool
uv tool install --from git+https://github.com/goodlux/mcp-server-pixeltable-developer.git mcp-server-pixeltable-developer

# Add to Claude Code
claude mcp add pixeltable mcp-server-pixeltable-developer

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

**⚠️ Note**: If you experience issues with Claude Desktop configuration, you may need to restart Claude Desktop after adding the MCP server configuration.

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

### Configuration for Cursor

Cursor users can add the Pixeltable MCP server to their `.cursorrules` file or configure it through Cursor's MCP settings:

1. **Via Cursor Settings:**
   - Open Cursor Settings
   - Navigate to "Features" → "Model Context Protocol"
   - Add a new MCP server with command: `mcp-server-pixeltable-developer`

2. **Via JSON Configuration:**
   Add to your Cursor MCP configuration:
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

3. **For development/source installations:**
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

## 🔧 Troubleshooting

### Claude Desktop Issues
If you're having trouble with Claude Desktop:
1. Restart Claude Desktop after adding the MCP server configuration
2. Check that the path to your Pixeltable home directory is correct
3. Ensure you have the latest version of Claude Desktop
4. Verify that `uv` is installed and accessible from your PATH

### Cursor Issues
If Cursor isn't recognizing the MCP server:
1. Make sure you have MCP support enabled in Cursor settings
2. Restart Cursor after configuration changes
3. Check the Cursor logs for any error messages

### Installation Issues
If installation fails:
1. Ensure you have Python 3.10+ installed
2. Make sure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Try installing from source if the GitHub installation fails

### Getting Help
If you encounter issues:
1. Use the built-in bug logging: `Claude: log_bug("description", severity="high")`
2. Check the generated bug report: `Claude: generate_bug_report()`
3. File an issue on the [GitHub repository](https://github.com/goodlux/mcp-server-pixeltable-developer/issues)

---

Built while having coffee. ☕