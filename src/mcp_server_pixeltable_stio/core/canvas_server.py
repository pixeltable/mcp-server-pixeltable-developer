"""Canvas streaming server for browser display.

Provides a Server-Sent Events (SSE) endpoint that the MCP `display_in_browser`
tool can push to. The canvas is OPTIONAL — FastAPI and uvicorn are imported
lazily inside `run_canvas_server_thread`, so the MCP server runs fine without
the `canvas` extra installed.

To enable the canvas, install with the extra (`uv pip install
'mcp-server-pixeltable-developer[canvas]'`) and set ``PIXELTABLE_MCP_CANVAS=1``;
the entry point in ``server.py`` reads that flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from queue import Empty, Queue
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Global queue for canvas messages; populated from any MCP tool thread.
_canvas_queue: Queue = Queue()
_sse_clients: List[asyncio.Queue] = []


def broadcast_to_canvas(message: Dict[str, Any]) -> None:
    """Queue a message for delivery to all connected SSE clients."""
    _canvas_queue.put(message)
    logger.info("Queued canvas message: %s", message.get("content_type", "unknown"))


def canvas_dependencies_available() -> bool:
    """True if FastAPI + uvicorn are installed (the `canvas` extra)."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        return True
    except ImportError:
        return False


async def _event_generator(client_queue: asyncio.Queue):
    """Generate SSE events for a single client."""
    try:
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"
        while True:
            message = await client_queue.get()
            yield f"data: {json.dumps(message)}\n\n"
    except asyncio.CancelledError:
        logger.info("Canvas SSE client disconnected")


async def _message_broadcaster():
    """Background task that fans messages out from the global queue to all SSE clients."""
    while True:
        try:
            message = _canvas_queue.get_nowait()
            for client_queue in _sse_clients:
                await client_queue.put(message)
        except Empty:
            await asyncio.sleep(0.1)


def _create_canvas_app():
    """Build the FastAPI app. Imports FastAPI lazily so the optional dep stays optional."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_message_broadcaster())
        logger.info("Canvas message broadcaster started")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Pixeltable Canvas", lifespan=lifespan)

    @app.get("/canvas/stream")
    async def canvas_stream():
        client_queue = asyncio.Queue()
        _sse_clients.append(client_queue)
        logger.info("Canvas SSE client connected (total: %d)", len(_sse_clients))
        return StreamingResponse(
            _event_generator(client_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    @app.get("/canvas", response_class=HTMLResponse)
    async def serve_canvas_page():
        import os
        import sys

        possible_paths = [
            # Editable install from repo root
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                "canvas.html",
            ),
            # Installed package data
            os.path.join(sys.prefix, "share", "mcp-server-pixeltable-developer", "canvas.html"),
            # Development location
            os.path.join(os.getcwd(), "canvas.html"),
        ]

        canvas_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if canvas_path:
            with open(canvas_path, "r") as fh:
                html_content = fh.read()
            html_content = html_content.replace(
                "const eventSource = new EventSource('http://localhost:8000/canvas/stream');",
                "const eventSource = new EventSource('/canvas/stream');",
            )
            return html_content

        return (
            "<html><body><h1>Canvas not found</h1><p>Searched paths:</p><ul>"
            + "".join(f"<li>{p}</li>" for p in possible_paths)
            + "</ul></body></html>"
        )

    @app.get("/media/{file_path:path}")
    async def serve_media(file_path: str):
        import os

        if not file_path.startswith("/"):
            file_path = "/" + file_path
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return {"error": "File not found", "path": file_path}

    return app


def run_canvas_server_thread(port: int = 7777) -> bool:
    """Start the canvas server in a background thread.

    Returns True if the thread started, False if the optional canvas
    dependencies are missing (no-op). We do NOT kill processes on the
    chosen port; if it's in use the uvicorn thread will log the conflict.
    """
    if not canvas_dependencies_available():
        logger.info(
            "Canvas dependencies not installed; skipping canvas startup. "
            "Install the optional extra: `uv pip install 'mcp-server-pixeltable-developer[canvas]'`."
        )
        return False

    import uvicorn  # lazy import — only when the extra is present

    def run_server():
        try:
            app = _create_canvas_app()
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
        except OSError as e:
            logger.warning(
                "Canvas server failed to bind to 127.0.0.1:%d (%s). "
                "Set PIXELTABLE_MCP_CANVAS_PORT to choose a different port.",
                port, e,
            )
        except Exception as e:
            logger.error("Canvas server crashed: %s", e)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info("Canvas server thread started on http://localhost:%d/canvas", port)
    return True
