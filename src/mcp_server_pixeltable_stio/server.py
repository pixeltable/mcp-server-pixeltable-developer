"""
Pixeltable MCP Server.

This module provides an MCP server implementation that exposes Pixeltable functionality.
Designed for stateless operation where each function call is independent.
"""

import os
import sys
import json
import logging
from mcp.server.fastmcp import FastMCP

# Import utilities
from mcp_server_pixeltable_stio.utils import setup_resilient_process

# Import core functionality
from mcp_server_pixeltable_stio.core.config import (
    get_default_pixeltable_path,
    get_system_default_pixeltable_path,
    get_effective_pixeltable_path,
    has_user_default_pixeltable
)

from mcp_server_pixeltable_stio.core.pixeltable_functions import (
    pixeltable_init,
    pixeltable_create_table,
    pixeltable_get_table,
    pixeltable_list_tables,
    pixeltable_drop_table,
    pixeltable_create_view,
    pixeltable_create_snapshot,
    pixeltable_create_dir,
    pixeltable_drop_dir,
    pixeltable_list_dirs,
    pixeltable_ls,
    pixeltable_move,
    pixeltable_list_functions,
    pixeltable_configure_logging,
    pixeltable_get_types,
    pixeltable_create_replica,
    pixeltable_query_table,
    pixeltable_get_table_schema,
    pixeltable_get_version,
    pixeltable_insert_data,
    pixeltable_add_computed_column,
    pixeltable_check_dependencies,
    pixeltable_install_yolox,
    pixeltable_install_openai,
    pixeltable_install_huggingface,
    pixeltable_install_all_dependencies,
    pixeltable_smart_install,
    pixeltable_auto_install_for_expression,
    pixeltable_suggest_install_from_error,
    pixeltable_system_diagnostics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Start the Pixeltable MCP server."""
    # Create MCP server
    mcp = FastMCP(name="pixeltable", version="0.1.0")
    
    # Log server initialization
    logger.info("Pixeltable MCP server initializing")
    
    # Try to initialize Pixeltable with appropriate data directory
    try:
        effective_path = get_effective_pixeltable_path()
        logger.info(f"Using Pixeltable data path: {effective_path}")
        
        # Set environment variable if needed
        if not has_user_default_pixeltable():
            os.environ['PIXELTABLE_HOME'] = effective_path
            logger.info(f"Set PIXELTABLE_HOME environment variable to: {effective_path}")
        
        # Set required Pixeltable configuration
        if 'PIXELTABLE_FILE_CACHE_SIZE_G' not in os.environ:
            os.environ['PIXELTABLE_FILE_CACHE_SIZE_G'] = '10'
            logger.info("Set PIXELTABLE_FILE_CACHE_SIZE_G=10")
            
        # Disable Pixeltable console output to prevent JSON parsing issues
        os.environ['PIXELTABLE_DISABLE_STDOUT'] = '1'
        
        # Initialize Pixeltable once at server startup
        import pixeltable as pxt
        
        # Try to force a clean initialization
        try:
            # First try without explicit init to see if auto-init worked
            tables = pxt.list_tables()
            logger.info(f"Pixeltable auto-initialized successfully, found {len(tables)} tables")
        except Exception as e:
            logger.info(f"Auto-init failed: {e}, trying explicit init")
            try:
                pxt.init()
                logger.info("Explicit pxt.init() completed successfully")
                tables = pxt.list_tables()
                logger.info(f"After explicit init, found {len(tables)} tables")
            except Exception as e2:
                logger.error(f"Explicit init also failed: {e2}")
                logger.info("Continuing with partial initialization")
        
        logger.info(f"Pixeltable setup completed with version {pxt.__version__}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Pixeltable: {e}")
        logger.info("Continuing without Pixeltable initialization")
    
    # Configure to never exit on stdin EOF and handle signals
    # original_exit = setup_resilient_process()  # DISABLED - causing restart issues
        
    # Register core table management functions
    mcp.tool()(pixeltable_init)
    mcp.tool()(pixeltable_create_table)
    mcp.tool()(pixeltable_get_table)
    mcp.tool()(pixeltable_list_tables)
    mcp.tool()(pixeltable_drop_table)
    mcp.tool()(pixeltable_create_view)
    mcp.tool()(pixeltable_create_snapshot)
    
    # Register directory management functions
    mcp.tool()(pixeltable_create_dir)
    mcp.tool()(pixeltable_drop_dir)
    mcp.tool()(pixeltable_list_dirs)
    mcp.tool()(pixeltable_ls)
    mcp.tool()(pixeltable_move)
    
    # Register function and utility functions
    mcp.tool()(pixeltable_list_functions)
    mcp.tool()(pixeltable_configure_logging)
    mcp.tool()(pixeltable_get_types)
    mcp.tool()(pixeltable_get_version)
    
    # Register extended operations
    mcp.tool()(pixeltable_create_replica)
    mcp.tool()(pixeltable_query_table)
    mcp.tool()(pixeltable_get_table_schema)
    mcp.tool()(pixeltable_insert_data)
    mcp.tool()(pixeltable_add_computed_column)
    
    # Register smart dependency management
    mcp.tool()(pixeltable_check_dependencies)
    mcp.tool()(pixeltable_install_yolox)
    mcp.tool()(pixeltable_install_openai)
    mcp.tool()(pixeltable_install_huggingface)
    mcp.tool()(pixeltable_install_all_dependencies)
    mcp.tool()(pixeltable_smart_install)
    mcp.tool()(pixeltable_auto_install_for_expression)
    mcp.tool()(pixeltable_suggest_install_from_error)
    mcp.tool()(pixeltable_system_diagnostics)
    
    # Start the server
    logger.info("Pixeltable MCP server starting...")
    mcp.run()


if __name__ == "__main__":
    main()
