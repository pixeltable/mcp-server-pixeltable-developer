"""FastAPI server for Pixeltable visualization."""

import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pixeltable as pxt

from mcp_server_pixeltable_stio.core.repl_session import get_repl_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pixeltable Visualization Server")

# Get REPL instance for executing Pixeltable operations
repl = get_repl_instance()


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pixeltable-visualization"}


@app.get("/api/tables")
async def list_tables():
    """List all Pixeltable tables."""
    try:
        result = repl.execute_python("import pixeltable as pxt; pxt.list_tables()")

        if result.get("success"):
            tables = result.get("output", [])
            return {"success": True, "tables": tables}
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/table/{table_path:path}")
async def get_table_info(table_path: str):
    """Get detailed information about a table."""
    try:
        code = f"""
import pixeltable as pxt
import json

table = pxt.get_table('{table_path}')
schema = {{}}

for col_name, col_type in table._schema.items():
    schema[col_name] = {{
        'type': str(col_type),
        'computed': hasattr(table, '_computed_columns') and col_name in table._computed_columns
    }}

result = {{
    'name': '{table_path}',
    'schema': schema,
    'row_count': table.count()
}}

print(json.dumps(result))
"""

        result = repl.execute_python(code)

        if result.get("success"):
            import json
            output = result.get("output", "")
            # Parse JSON from output
            try:
                table_info = json.loads(output.strip())
                return {"success": True, "table": table_info}
            except json.JSONDecodeError:
                return {"success": True, "raw_output": output}
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}

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
        tables_result = repl.execute_python("import pixeltable as pxt; pxt.list_tables()")

        if tables_result.get("success"):
            tables = tables_result.get("output", [])
            await websocket.send_json({
                "type": "tables",
                "data": tables
            })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "get_table":
                table_path = data.get("table_path")

                code = f"""
import pixeltable as pxt
import json

table = pxt.get_table('{table_path}')
schema = {{}}

for col_name, col_type in table._schema.items():
    schema[col_name] = {{
        'type': str(col_type)
    }}

result = {{
    'name': '{table_path}',
    'schema': schema,
    'row_count': table.count()
}}

print(json.dumps(result))
"""

                result = repl.execute_python(code)

                if result.get("success"):
                    import json
                    try:
                        table_info = json.loads(result.get("output", "{}").strip())
                        await websocket.send_json({
                            "type": "table_info",
                            "data": table_info
                        })
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Failed to parse table info"
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": result.get("error", "Unknown error")
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
