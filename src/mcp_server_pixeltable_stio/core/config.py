"""
Configuration functions for Pixeltable MCP.

This module provides functions for getting configuration settings.
"""

import logging
import os
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def get_system_default_pixeltable_path() -> str:
    """
    Get the system default Pixeltable data path.
    
    Returns:
        Path to the system default Pixeltable data directory
    """
    return os.path.expanduser("~/.pixeltable")

def get_default_pixeltable_path() -> Optional[str]:
    """
    Get the user default Pixeltable path from environment variables.
    
    Returns:
        Path to the user default Pixeltable data directory, or None if not configured
    """
    env_path = os.environ.get("PIXELTABLE_HOME")
    if env_path:
        # Expand user directory if needed
        if env_path.startswith("~"):
            env_path = os.path.expanduser(env_path)
        return env_path
    return None

def has_user_default_pixeltable() -> bool:
    """
    Check if a user default Pixeltable data directory is configured.
    
    Returns:
        True if a user default is configured, False otherwise
    """
    return get_default_pixeltable_path() is not None

def get_effective_pixeltable_path() -> str:
    """
    Get the effective Pixeltable data path with fallbacks.
    
    Returns:
        The effective path to use for Pixeltable data
    """
    # Try user-configured path first
    user_path = get_default_pixeltable_path()
    if user_path:
        return user_path
    
    # Fall back to system default
    return get_system_default_pixeltable_path()
