"""
Configuration functions for Pixeltable MCP.

This module provides functions for getting and setting configuration settings.
"""

import logging
import os
from typing import Optional, Dict, Any
import json

# Configure logging
logger = logging.getLogger(__name__)

# Config file path for persistent settings
CONFIG_FILE = os.path.expanduser("~/.pixeltable_mcp_config.json")

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
    
    Priority order:
    1. PIXELTABLE_HOME environment variable
    2. Configured datastore path from config file
    3. System default ~/.pixeltable
    
    Returns:
        The effective path to use for Pixeltable data
    """
    # Try environment variable first (highest priority)
    env_path = get_default_pixeltable_path()
    if env_path:
        return env_path
    
    # Try config file setting
    config = load_config()
    if config and 'datastore_path' in config:
        return os.path.expanduser(config['datastore_path'])
    
    # Fall back to system default
    return get_system_default_pixeltable_path()

def load_config() -> Optional[Dict[str, Any]]:
    """
    Load configuration from file.
    
    Returns:
        Configuration dictionary or None if file doesn't exist
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    return None

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save config file: {e}")
        return False

def set_datastore_path(path: str) -> bool:
    """
    Set the Pixeltable datastore path in configuration.
    
    Args:
        path: Path to the datastore directory
        
    Returns:
        True if set successfully, False otherwise
    """
    try:
        # Expand user path
        expanded_path = os.path.expanduser(path)
        
        # Load existing config or create new
        config = load_config() or {}
        
        # Update datastore path
        config['datastore_path'] = expanded_path
        
        # Save config
        if save_config(config):
            logger.info(f"Set datastore path to: {expanded_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to set datastore path: {e}")
        return False

def get_configured_datastore_path() -> Optional[str]:
    """
    Get the configured datastore path from config file.
    
    Returns:
        Configured datastore path or None if not configured
    """
    config = load_config()
    if config and 'datastore_path' in config:
        return os.path.expanduser(config['datastore_path'])
    return None
