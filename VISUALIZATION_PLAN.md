# Pixeltable Visualization UI - Implementation Plan

## Overview
Add visual interface for exploring Pixeltable pipelines via MCP tools.

## Architecture
```
User asks Claude: "Start visualization server"
    ↓
Claude calls MCP tool: start_visualization_server()
    ↓
Python subprocess runs: npm install && npm run build (first time)
Python starts FastAPI server on port 7777
    ↓
User opens http://localhost:7777
    ↓
React UI shows:
- Table list in sidebar
- Pipeline DAG visualization with React Flow
- WebSocket connection to backend
```

## Implementation Status
- ✅ FastAPI dependencies added to pyproject.toml
- ⏸️ Full implementation in progress (files recreating after git mishap)

## Next Steps
1. Create `src/mcp_server_pixeltable_stio/core/ui_server.py` - Server management functions
2. Add MCP tools to `server.py`: start/stop/status visualization server
3. Create `src/mcp_server_pixeltable_stio/server/api/main.py` - FastAPI + WebSocket
4. Create React UI in `src/mcp_server_pixeltable_stio/server/ui/`
5. Add `.gitignore` for node_modules

## User Experience
```bash
# Install MCP
uv tool install mcp-server-pixeltable-developer

# In Claude:
> "Start the visualization server"

Claude: ✅ Server started at http://localhost:7777!
Open your browser to visualize your Pixeltable pipelines.
```

## Tech Stack
- Backend: FastAPI + WebSocket + MCP REPL
- Frontend: React + Vite + Tailwind + React Flow
- Auto-setup: Python subprocess runs npm commands

## Files to Create
- `core/ui_server.py` - start/stop/status functions
- `server/api/main.py` - FastAPI server
- `server/ui/` - React app (package.json, src/, etc.)
- `.gitignore` - exclude node_modules/
