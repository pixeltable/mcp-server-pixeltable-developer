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
            },
        )

    @app.get("/canvas", response_class=HTMLResponse)
    async def serve_canvas_page():
        """Serve simple canvas HTML page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pixeltable Canvas</title>
            <style>
                body { font-family: sans-serif; padding: 20px; background: #1a1a1a; color: #fff; }
                #canvas { margin-top: 20px; }
                .item { margin: 20px 0; padding: 20px; background: #2a2a2a; border-radius: 8px; }
                img { max-width: 100%; height: auto; }
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; border: 1px solid #444; text-align: left; }
                th { background: #333; }
            </style>
        </head>
        <body>
            <h1>Pixeltable Canvas</h1>
            <div id="status">Connecting...</div>
            <div id="canvas"></div>
            <script>
                const eventSource = new EventSource('/canvas/stream');
                const canvas = document.getElementById('canvas');
                const status = document.getElementById('status');

                eventSource.onopen = () => {
                    status.textContent = 'Connected ✓';
                    status.style.color = '#4ade80';
                };

                eventSource.onerror = () => {
                    status.textContent = 'Disconnected ✗';
                    status.style.color = '#f87171';
                };

                eventSource.onmessage = (event) => {
                    const message = JSON.parse(event.data);

                    if (message.type === 'connected') return;

                    const item = document.createElement('div');
                    item.className = 'item';

                    switch(message.content_type) {
                        case 'image':
                            const img = document.createElement('img');
                            img.src = message.data;
                            item.appendChild(img);
                            break;

                        case 'text':
                            item.textContent = message.data;
                            break;

                        case 'html':
                            item.innerHTML = message.data;
                            break;

                        case 'table':
                            const table = document.createElement('table');
                            // Assume data is array of objects
                            if (message.data.length > 0) {
                                const headers = Object.keys(message.data[0]);
                                const thead = table.createTHead();
                                const headerRow = thead.insertRow();
                                headers.forEach(h => {
                                    const th = document.createElement('th');
                                    th.textContent = h;
                                    headerRow.appendChild(th);
                                });
                                const tbody = table.createTBody();
                                message.data.forEach(row => {
                                    const tr = tbody.insertRow();
                                    headers.forEach(h => {
                                        const td = tr.insertCell();
                                        td.textContent = row[h];
                                    });
                                });
                            }
                            item.appendChild(table);
                            break;

                        default:
                            item.textContent = JSON.stringify(message, null, 2);
                    }

                    canvas.insertBefore(item, canvas.firstChild);
                };
            </script>
        </body>
        </html>
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
