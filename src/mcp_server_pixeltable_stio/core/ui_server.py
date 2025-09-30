"""UI Server management for Pixeltable visualization."""

import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
_ui_server_process: Optional[subprocess.Popen] = None

def ensure_ui_dependencies() -> Dict[str, Any]:
    """Ensure UI dependencies are installed and built."""
    try:
        ui_dir = Path(__file__).parent.parent / "server" / "ui"
        dist_dir = ui_dir / "dist"
        
        if dist_dir.exists() and (dist_dir / "index.html").exists():
            return {"success": True, "message": "UI already built", "needs_build": False}
        
        logger.info("Building UI for first time...")
        
        # Check npm availability
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"success": False, "error": "npm not found. Install Node.js from https://nodejs.org/"}
        
        # Install dependencies
        if not (ui_dir / "node_modules").exists():
            logger.info("Installing npm dependencies...")
            result = subprocess.run(["npm", "install"], cwd=ui_dir, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "error": f"npm install failed: {result.stderr}"}
        
        # Build UI
        logger.info("Building UI...")
        result = subprocess.run(["npm", "run", "build"], cwd=ui_dir, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "error": f"npm build failed: {result.stderr}"}
        
        return {"success": True, "message": "UI built successfully", "needs_build": True}
    
    except Exception as e:
        logger.error(f"Error ensuring UI dependencies: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def start_visualization_server(port: int = 7777) -> Dict[str, Any]:
    """Start the Pixeltable visualization server."""
    global _ui_server_process
    
    try:
        if _ui_server_process is not None and _ui_server_process.poll() is None:
            return {
                "success": False,
                "error": "Visualization server already running",
                "url": f"http://localhost:{port}"
            }
        
        build_result = ensure_ui_dependencies()
        if not build_result["success"]:
            return build_result
        
        logger.info(f"Starting visualization server on port {port}...")
        
        _ui_server_process = subprocess.Popen(
            ["uvicorn", "mcp_server_pixeltable_stio.server.api.main:app",
             "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(2)
        
        if _ui_server_process.poll() is not None:
            stderr = _ui_server_process.stderr.read() if _ui_server_process.stderr else ""
            return {"success": False, "error": f"Server failed to start: {stderr}"}
        
        return {
            "success": True,
            "message": f"Visualization server started on port {port}",
            "url": f"http://localhost:{port}",
            "websocket": f"ws://localhost:{port}/ws",
            "pid": _ui_server_process.pid,
            "instructions": [
                f"1. Open your browser to http://localhost:{port}",
                "2. You'll see the Pixeltable UI with:",
                "   - List of all tables in the sidebar",
                "   - Click a table to see its pipeline visualization",
                "3. To stop: use stop_visualization_server()"
            ]
        }
    
    except Exception as e:
        logger.error(f"Error starting visualization server: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def stop_visualization_server() -> Dict[str, Any]:
    """Stop the Pixeltable visualization server."""
    global _ui_server_process
    
    try:
        if _ui_server_process is None or _ui_server_process.poll() is not None:
            return {"success": False, "error": "Visualization server is not running"}
        
        logger.info("Stopping visualization server...")
        _ui_server_process.terminate()
        
        try:
            _ui_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ui_server_process.kill()
            _ui_server_process.wait()
        
        _ui_server_process = None
        return {"success": True, "message": "Visualization server stopped"}
    
    except Exception as e:
        logger.error(f"Error stopping visualization server: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def get_visualization_server_status() -> Dict[str, Any]:
    """Get the status of the visualization server."""
    global _ui_server_process
    
    if _ui_server_process is None or _ui_server_process.poll() is not None:
        return {"running": False, "message": "Visualization server is not running"}
    
    return {
        "running": True,
        "pid": _ui_server_process.pid,
        "url": "http://localhost:7777",
        "websocket": "ws://localhost:7777/ws"
    }
