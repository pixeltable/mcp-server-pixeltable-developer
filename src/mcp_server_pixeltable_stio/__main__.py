"""
Main entry point for the Pixeltable MCP server.
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add some diagnostics to help debug startup issues
logger.info("Starting Pixeltable MCP server")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Module path: {__file__}")

try:
    # Import server main function
    from mcp_server_pixeltable_stio.server import main
    
    # Run the server
    if __name__ == "__main__":
        logger.info("Calling main()")
        main()
except Exception as e:
    logger.error(f"Error starting server: {e}", exc_info=True)
    # Don't exit - we need to keep process alive for MCP
    # Continue with a minimal server
    import time
    
    # Set up a minimal MCP server that just stays alive
    from mcp.server.fastmcp import FastMCP
    
    # Import utility for resilient process
    from mcp_server_pixeltable_stio.utils import setup_resilient_process
    
    # Make sure we don't exit
    setup_resilient_process()
    
    # Start minimal MCP server
    logger.info("Starting minimal MCP server after error")
    minimal_mcp = FastMCP(name="pixeltable-minimal", version="0.1.0")
    minimal_mcp.run()
