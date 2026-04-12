"""
Main entry point for the Pixeltable MCP server.

Activates uvloop (if available) before any other asyncio usage,
then delegates to server.main().
"""

import argparse
import asyncio
import logging
import sys
import os

# Activate uvloop before anything else touches the event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("mcp-server-pixeltable-developer")
    except Exception:
        from mcp_server_pixeltable_stio import __version__

        return __version__


def main():
    """Entry point called by the console script."""
    parser = argparse.ArgumentParser(
        prog="mcp-server-pixeltable-developer",
        description=(
            "Pixeltable developer MCP server: stdio JSON-RPC for MCP clients, "
            "optional canvas UI on http://127.0.0.1:7777/canvas."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    parser.parse_args()

    logger.info("Starting Pixeltable MCP server")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Module path: {__file__}")

    try:
        from mcp_server_pixeltable_stio.server import main as server_main
        logger.info("Calling server main()")
        server_main()
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)

        # Fall back to a minimal MCP server that stays alive
        from mcp.server.fastmcp import FastMCP
        from mcp_server_pixeltable_stio.utils import setup_resilient_process

        setup_resilient_process()
        logger.info("Starting minimal MCP server after error")
        minimal_mcp = FastMCP(name="pixeltable-minimal")
        minimal_mcp.run()


if __name__ == "__main__":
    main()
