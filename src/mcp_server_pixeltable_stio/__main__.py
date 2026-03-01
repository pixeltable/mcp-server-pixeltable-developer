"""
Main entry point for the Pixeltable MCP server.

Activates uvloop (if available) before any other asyncio usage,
then delegates to server.main().
"""

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


def main():
    """Entry point called by the console script."""
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
