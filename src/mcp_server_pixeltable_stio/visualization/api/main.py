"""FastAPI server for Pixeltable visualization."""

import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from mcp_server_pixeltable_stio.visualization.executor import get_executor_manager
from mcp_server_pixeltable_stio.visualization.worker import get_tables, get_table_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pixeltable Visualization Server")

# Get executor manager for running Pixeltable operations in isolated processes
executor = get_executor_manager()


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pixeltable-visualization"}


@app.get("/api/tables")
async def list_tables():
    """List all Pixeltable tables."""
    try:
        tables = await executor.run(get_tables)
        return {"success": True, "tables": tables}
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/table/{table_path:path}")
async def get_table_info_endpoint(table_path: str):
    """Get detailed information about a table."""
    try:
        table_info = await executor.run(get_table_info, table_path)

        if 'error' in table_info:
            return {"success": False, "error": table_info['error']}

        return {"success": True, "table": table_info}
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws/pipeline")
async def pipeline_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # Send initial table list
        tables = await executor.run(get_tables)
        await websocket.send_json({
            "type": "tables",
            "data": tables
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "get_table":
                table_path = data.get("table_path")

                table_info = await executor.run(get_table_info, table_path)

                if 'error' in table_info:
                    await websocket.send_json({
                        "type": "error",
                        "error": table_info['error']
                    })
                else:
                    await websocket.send_json({
                        "type": "table_info",
                        "data": table_info
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Serve static files (React build)
ui_dir = Path(__file__).parent.parent / "ui" / "dist"

if ui_dir.exists():
    app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_ui(full_path: str):
        """Serve React UI."""
        file_path = ui_dir / full_path

        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        else:
            # Return index.html for client-side routing
            return FileResponse(ui_dir / "index.html")
else:
    logger.warning(f"UI dist directory not found at {ui_dir}")

    @app.get("/")
    async def no_ui():
        return {
            "error": "UI not built",
            "message": "Run 'npm install && npm run build' in the ui directory"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
