"""UI Server management for Pixeltable visualization."""

import sys
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
        ui_dir = Path(__file__).parent.parent / "visualization" / "ui"
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

