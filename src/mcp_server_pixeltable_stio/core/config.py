"""
Simple configuration for Pixeltable MCP using TOML.
"""

import logging
import os
import sys
import toml
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_config_path() -> str:
    """
    Get the path to the config.toml file.

    If running in uv tool environment, use config.toml in the tool directory.
    Otherwise use ~/.pixeltable_mcp/config.toml
    """
    if sys.executable and '.local/share/uv/tools/' in sys.executable:
        # We're in a uv tool environment
        tool_dir = sys.prefix
        return os.path.join(tool_dir, 'config.toml')
    else:
        # Development or other environment
        config_dir = os.path.expanduser("~/.pixeltable_mcp")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, 'config.toml')

def load_config() -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Returns default config if file doesn't exist.
    """
    config_path = get_config_path()

    # Default configuration
    default_config = {
        'storage': {
            'datastore_path': os.path.expanduser('~/.pixeltable')
        },
        'cache': {
            'file_cache_size_gb': 10
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded = toml.load(f)
                # Merge loaded config with defaults
                for section in default_config:
                    if section in loaded:
                        default_config[section].update(loaded[section])
                    else:
                        loaded[section] = default_config[section]
                return loaded
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Config file not found at {config_path}, using defaults")
        # Create the default config file
        save_config(default_config)

    return default_config

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to TOML file.
    """
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            toml.dump(config, f)
        logger.info(f"Saved config to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False

def get_datastore_path() -> str:
    """
    Get the configured datastore path.
    """
    config = load_config()
    path = config.get('storage', {}).get('datastore_path', '~/.pixeltable')
    return os.path.expanduser(path)

def set_datastore_path(path: str) -> bool:
    """
    Update the datastore path in configuration.
    """
    config = load_config()
    if 'storage' not in config:
        config['storage'] = {}
    config['storage']['datastore_path'] = path
    return save_config(config)

# Keep these for backward compatibility
def get_effective_pixeltable_path() -> str:
    """Get the effective Pixeltable data path."""
    return get_datastore_path()

def get_configured_datastore_path() -> Optional[str]:
    """Get the configured datastore path."""
    return get_datastore_path()

def has_user_default_pixeltable() -> bool:
    """Check if a custom datastore path is configured."""
    config = load_config()
    return 'storage' in config and 'datastore_path' in config['storage']

def get_system_default_pixeltable_path() -> str:
    """Get system default path."""
    return os.path.expanduser('~/.pixeltable')

def get_default_pixeltable_path() -> Optional[str]:
    """No longer checks environment variables."""
    return None