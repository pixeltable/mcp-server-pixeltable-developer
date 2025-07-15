"""
Utility functions for the Pixeltable MCP server.

This module provides utility functions for the Pixeltable MCP server.
"""

import os
import sys
import signal
import logging

logger = logging.getLogger(__name__)

def setup_resilient_process():
    """
    Set up the process to be resilient to termination.
    
    This function:
    1. Prevents sys.exit from terminating the process
    2. Sets up signal handlers to ignore termination signals
    3. Ensures unbuffered I/O
    
    Returns:
        The original sys.exit function in case it needs to be restored
    """
    # Override sys.exit to prevent it from being called
    original_exit = sys.exit
    def exit_prevention(code=0):
        logger.warning(f"Exit prevented with code {code}")
    sys.exit = exit_prevention
    
    # Set up signal handlers to prevent termination
    def handle_signal(sig, frame):
        logger.warning(f"Signal {sig} ignored")
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]:
        try:
            signal.signal(sig, handle_signal)
        except Exception:
            pass  # Some signals might not be available on all platforms
    
    # Force unbuffered mode for all IO
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    
    return original_exit
