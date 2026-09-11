"""Authenticated localhost canvas used by the opt-in unsafe display tool."""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import os
import secrets
import stat
import sys
import threading
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any
from urllib.parse import quote, unquote, urlsplit

logger = logging.getLogger(__name__)

_MAX_CANVAS_MESSAGE_BYTES = 1_000_000
_MAX_MEDIA_BYTES = 256 * 1024 * 1024
_MAX_REGISTERED_MEDIA = 256
_MAX_CLIENT_QUEUE = 100
_MEDIA_CHUNK_BYTES = 64 * 1024
_ALLOWED_MEDIA_TYPES = ("image/", "audio/", "video/")
_ALLOWED_EXACT_MEDIA_TYPES = {"application/pdf"}


class CanvasError(ValueError):
    """An anticipated canvas validation or startup failure."""


class _CanvasHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class CanvasController:
    """Serve authenticated canvas events and allowlisted local media."""

    def __init__(
        self,
        media_roots: Iterable[Path],
        *,
        port: int = 7777,
        html_path: Path | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise CanvasError("Canvas port must be between 0 and 65535.")

        roots: list[Path] = []
        for root in media_roots:
            resolved = Path(root).expanduser().resolve(strict=False)
            if resolved.exists() and resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
        if not roots:
            raise CanvasError("At least one existing media root is required.")

        self.media_roots = tuple(roots)
        self._token = secrets.token_urlsafe(32)
        self._html = _load_canvas_html(html_path)
        self._clients: set[Queue[bytes]] = set()
        self._clients_lock = threading.Lock()
        self._history: deque[bytes] = deque(maxlen=20)
        self._media: OrderedDict[str, tuple[Path, str, int]] = OrderedDict()
        self._media_lock = threading.Lock()
        self._closed = threading.Event()

        controller = self

        class CanvasRequestHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                controller._handle_get(self)

            def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                controller._handle_options(self)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("Canvas HTTP: " + format, *args)

        try:
            self._server = _CanvasHTTPServer(("127.0.0.1", port), CanvasRequestHandler)
        except OSError as exc:
            raise CanvasError(f"Cannot bind the canvas to 127.0.0.1:{port}: {exc}") from exc

        self.port = int(self._server.server_address[1])
        self.origin = f"http://127.0.0.1:{self.port}"
        self.url = f"{self.origin}/canvas#token={quote(self._token, safe='')}"
        self._thread = threading.Thread(
            target=self._serve,
            name="pixeltable-canvas",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=0.2)
        except Exception:
            if not self._closed.is_set():
                logger.exception("Canvas server stopped unexpectedly")

    def close(self) -> None:
        """Stop the canvas server and disconnect its clients."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        with self._clients_lock:
            self._clients.clear()
            self._history.clear()
        with self._media_lock:
            self._media.clear()

    def broadcast(self, message: Mapping[str, Any]) -> None:
        """Send a JSON-serializable message to every connected canvas."""
        try:
            encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CanvasError(f"Canvas data must be JSON serializable: {exc}") from exc
        if len(encoded) > _MAX_CANVAS_MESSAGE_BYTES:
            raise CanvasError(f"Canvas message exceeds the {_MAX_CANVAS_MESSAGE_BYTES}-byte limit.")

        event = b"data: " + encoded + b"\n\n"
        with self._clients_lock:
            self._history.append(event)
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.put_nowait(event)
            except Full:
                with suppress(Empty):
                    client.get_nowait()
                try:
                    client.put_nowait(event)
                except Full:
                    logger.debug("Dropping a canvas message for a slow client")

    def register_media(self, source: str) -> str:
        """Validate a file URL and return its authenticated canvas URL."""
        path = _path_from_file_url(source)
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CanvasError("The local media file does not exist.") from exc
        if not resolved.is_file() or not _is_within_roots(resolved, self.media_roots):
            raise CanvasError("Local media must be a file inside a configured media root.")

        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        if not (media_type.startswith(_ALLOWED_MEDIA_TYPES) or media_type in _ALLOWED_EXACT_MEDIA_TYPES):
            raise CanvasError(f"Unsupported local media type: {media_type}.")
        size = resolved.stat().st_size
        if size > _MAX_MEDIA_BYTES:
            raise CanvasError(f"Local media exceeds the {_MAX_MEDIA_BYTES}-byte limit.")

        media_id = secrets.token_urlsafe(18)
        with self._media_lock:
            self._media[media_id] = (resolved, media_type, size)
            self._media.move_to_end(media_id)
            while len(self._media) > _MAX_REGISTERED_MEDIA:
                self._media.popitem(last=False)
        return f"/media/{media_id}/{quote(resolved.name, safe='')}"

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._request_origin_is_allowed(handler):
            self._send_error(handler, HTTPStatus.FORBIDDEN, "Cross-origin request rejected.")
            return

        request_path = urlsplit(handler.path).path
        if request_path == "/canvas":
            self._send_bytes(handler, HTTPStatus.OK, self._html, "text/html; charset=utf-8")
            return
        if request_path == "/canvas/stream":
            if not self._is_authorized(handler):
                self._send_unauthorized(handler)
                return
            self._serve_event_stream(handler)
            return
        if request_path.startswith("/media/"):
            if not self._is_authorized(handler):
                self._send_unauthorized(handler)
                return
            self._serve_media(handler, request_path)
            return
        self._send_error(handler, HTTPStatus.NOT_FOUND, "Not found.")

    def _handle_options(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._request_origin_is_allowed(handler):
            self._send_error(handler, HTTPStatus.FORBIDDEN, "Cross-origin request rejected.")
            return
        handler.send_response(HTTPStatus.NO_CONTENT)
        self._send_security_headers(handler)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _request_origin_is_allowed(self, handler: BaseHTTPRequestHandler) -> bool:
        host = handler.headers.get("Host", "").lower()
        allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if host not in allowed_hosts:
            return False
        origin = handler.headers.get("Origin")
        return origin is None or origin == self.origin

    def _is_authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        authorization = handler.headers.get("Authorization", "")
        return hmac.compare_digest(authorization, f"Bearer {self._token}")

    def _serve_event_stream(self, handler: BaseHTTPRequestHandler) -> None:
        client: Queue[bytes] = Queue(maxsize=_MAX_CLIENT_QUEUE)
        with self._clients_lock:
            self._clients.add(client)
            history = tuple(self._history)

        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "keep-alive")
        self._send_security_headers(handler)
        handler.end_headers()
        try:
            handler.wfile.write(b'data: {"type":"connected"}\n\n')
            for event in history:
                handler.wfile.write(event)
            handler.wfile.flush()
            while not self._closed.is_set():
                try:
                    event = client.get(timeout=15)
                except Empty:
                    event = b": keepalive\n\n"
                handler.wfile.write(event)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(client)

    def _serve_media(self, handler: BaseHTTPRequestHandler, request_path: str) -> None:
        parts = request_path.split("/", 4)
        if len(parts) < 4 or not parts[2]:
            self._send_error(handler, HTTPStatus.NOT_FOUND, "Media not found.")
            return
        media_id = parts[2]
        with self._media_lock:
            media = self._media.get(media_id)
        if media is None:
            self._send_error(handler, HTTPStatus.NOT_FOUND, "Media not found.")
            return

        path, media_type, expected_size = media
        try:
            resolved = path.resolve(strict=True)
            if resolved != path or not _is_within_roots(resolved, self.media_roots):
                raise OSError("media path changed")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(resolved, flags)
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError("media is no longer a regular file")
            if file_stat.st_size != expected_size or file_stat.st_size > _MAX_MEDIA_BYTES:
                raise OSError("media size changed")
        except OSError:
            with suppress(NameError, OSError):
                os.close(file_descriptor)
            self._send_error(handler, HTTPStatus.NOT_FOUND, "Media not found.")
            return

        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", media_type)
        handler.send_header("Content-Length", str(file_stat.st_size))
        handler.send_header(
            "Content-Disposition",
            f"inline; filename*=UTF-8''{quote(path.name, safe='')}",
        )
        self._send_security_headers(handler)
        handler.end_headers()
        try:
            with os.fdopen(file_descriptor, "rb") as media_file:
                while chunk := media_file.read(_MEDIA_CHUNK_BYTES):
                    handler.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_unauthorized(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(HTTPStatus.UNAUTHORIZED)
        handler.send_header("WWW-Authenticate", 'Bearer realm="pixeltable-canvas"')
        self._send_json_body(handler, {"error": "Bearer token required."})

    def _send_error(
        self,
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        message: str,
    ) -> None:
        handler.send_response(status)
        self._send_json_body(handler, {"error": message})

    def _send_json_body(self, handler: BaseHTTPRequestHandler, value: Mapping[str, str]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self._send_response_body(handler, body, "application/json; charset=utf-8")

    def _send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        body: bytes,
        media_type: str,
    ) -> None:
        handler.send_response(status)
        self._send_response_body(handler, body, media_type)

    def _send_response_body(
        self,
        handler: BaseHTTPRequestHandler,
        body: bytes,
        media_type: str,
    ) -> None:
        handler.send_header("Content-Type", media_type)
        handler.send_header("Content-Length", str(len(body)))
        self._send_security_headers(handler)
        handler.end_headers()
        handler.wfile.write(body)

    def _send_security_headers(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Referrer-Policy", "no-referrer")
        handler.send_header("Cross-Origin-Opener-Policy", "same-origin")
        handler.send_header("Cross-Origin-Resource-Policy", "same-origin")
        handler.send_header("X-Frame-Options", "DENY")
        handler.send_header(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; connect-src 'self'; "
            "form-action 'none'; frame-ancestors 'none'; frame-src 'self' data:; "
            "img-src 'self' data: blob: https:; media-src 'self' blob: https:; "
            "object-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'",
        )
        if handler.headers.get("Origin") == self.origin:
            handler.send_header("Access-Control-Allow-Origin", self.origin)
            handler.send_header("Vary", "Origin")
            handler.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            handler.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")


def _path_from_file_url(source: str) -> Path:
    parsed = urlsplit(source)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise CanvasError("Local media must use a file:// URL without a remote host.")
    if parsed.query or parsed.fragment:
        raise CanvasError("Local media file URLs cannot contain a query or fragment.")
    decoded = unquote(parsed.path)
    if not decoded or not Path(decoded).is_absolute():
        raise CanvasError("Local media file URLs must contain an absolute path.")
    return Path(decoded)


def _is_within_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _load_canvas_html(explicit_path: Path | None) -> bytes:
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "canvas.html",
            Path(sys.prefix) / "share" / "mcp-server-pixeltable-developer" / "canvas.html",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_bytes()
        except OSError:
            continue
    raise CanvasError("The packaged canvas.html asset is missing.")


def transform_local_media(value: Any, canvas: CanvasController, *, _depth: int = 0) -> Any:
    """Replace nested file URLs with authenticated, allowlisted canvas URLs."""
    if _depth > 20:
        raise CanvasError("Canvas data nesting exceeds the supported depth.")
    if isinstance(value, str):
        return canvas.register_media(value) if value.startswith("file://") else value
    if isinstance(value, list):
        return [transform_local_media(item, canvas, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return [transform_local_media(item, canvas, _depth=_depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: transform_local_media(item, canvas, _depth=_depth + 1) for key, item in value.items()}
    return value
