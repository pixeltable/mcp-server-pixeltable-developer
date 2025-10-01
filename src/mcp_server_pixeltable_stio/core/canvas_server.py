"""Canvas streaming server for browser display.

Provides SSE (Server-Sent Events) endpoint for pushing display commands to browser.
Runs alongside the main MCP stdio transport using FastAPI.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from queue import Queue
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

logger = logging.getLogger(__name__)

# Global queue for canvas messages
_canvas_queue: Queue = Queue()
_sse_clients: List[asyncio.Queue] = []


def broadcast_to_canvas(message: Dict[str, Any]) -> None:
    """Broadcast a message to all connected canvas clients.

    This is called from MCP tools (which may be in different threads).
    """
    _canvas_queue.put(message)
    logger.info(f"Queued canvas message: {message.get('content_type', 'unknown')}")


async def event_generator(client_queue: asyncio.Queue):
    """Generate SSE events for a single client."""
    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"

        # Stream messages to this client
        while True:
            message = await client_queue.get()
            yield f"data: {json.dumps(message)}\n\n"
    except asyncio.CancelledError:
        logger.info("Canvas SSE client disconnected")


async def message_broadcaster():
    """Background task that broadcasts messages from queue to all SSE clients."""
    while True:
        # Check queue for new messages (non-blocking)
        try:
            message = _canvas_queue.get_nowait()
            # Broadcast to all connected clients
            for client_queue in _sse_clients:
                await client_queue.put(message)
        except:
            # Queue empty, sleep briefly
            await asyncio.sleep(0.1)


def create_canvas_app() -> FastAPI:
    """Create FastAPI app for canvas streaming."""
    app = FastAPI(title="Pixeltable Canvas")

    @app.on_event("startup")
    async def startup():
        # Start message broadcaster
        asyncio.create_task(message_broadcaster())
        logger.info("Canvas message broadcaster started")

    @app.get("/canvas/stream")
    async def canvas_stream():
        """SSE endpoint for canvas streaming."""
        # Create a queue for this client
        client_queue = asyncio.Queue()
        _sse_clients.append(client_queue)

        logger.info(f"Canvas SSE client connected (total: {len(_sse_clients)})")

        return StreamingResponse(
            event_generator(client_queue),
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
        """Serve canvas HTML page from file."""
        import os
        import sys

        # Try multiple locations for canvas.html
        possible_paths = [
            # Editable install from repo root
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'canvas.html'),
            # Installed package data
            os.path.join(sys.prefix, 'share', 'mcp-server-pixeltable-developer', 'canvas.html'),
            # Development location
            os.path.join(os.getcwd(), 'canvas.html'),
        ]

        canvas_path = None
        for path in possible_paths:
            if os.path.exists(path):
                canvas_path = path
                break

        if canvas_path:
            with open(canvas_path, 'r') as f:
                html_content = f.read()
            # Replace the SSE URL to use relative path
            html_content = html_content.replace(
                "const eventSource = new EventSource('http://localhost:8000/canvas/stream');",
                "const eventSource = new EventSource('/canvas/stream');"
            )
            return html_content
        else:
            # Fallback to basic HTML if file not found
            return """
            <html><body>
            <h1>Canvas not found</h1>
            <p>Searched paths:</p>
            <ul>""" + ''.join(f'<li>{p}</li>' for p in possible_paths) + """</ul>
            </body></html>
            """

    return app


def run_canvas_server_thread(port: int = 8000):
    """Run canvas server in a separate thread."""
    def run_server():
        app = create_canvas_app()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"Canvas server thread started on http://localhost:{port}/canvas")
