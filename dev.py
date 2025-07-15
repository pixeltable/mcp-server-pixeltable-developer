#!/usr/bin/env python3
"""
Development and testing utilities for Pixeltable MCP server.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    try:
        import mcp_server_pixeltable
        print(f"✅ Package imported: {mcp_server_pixeltable.__version__}")
        
        from mcp_server_pixeltable.core.config import get_effective_pixeltable_path
        print(f"✅ Config works, effective path: {get_effective_pixeltable_path()}")
        
        from mcp_server_pixeltable.core.pixeltable_functions import ensure_pixeltable_available
        print("✅ Functions module imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pixeltable_availability():
    """Test if Pixeltable is available."""
    print("\nTesting Pixeltable availability...")
    try:
        import pixeltable as pxt
        print(f"✅ Pixeltable available: {pxt.__version__}")
        return True
    except ImportError:
        print("ℹ️ Pixeltable not installed (install with: pip install pixeltable)")
        return False

def test_mcp_dependencies():
    """Test MCP dependencies."""
    print("\nTesting MCP dependencies...")
    try:
        from mcp.server.fastmcp import FastMCP
        print("✅ MCP FastMCP available")
        
        # Test creating a server instance
        server = FastMCP(name="test", version="0.1.0")
        print("✅ MCP server creation works")
        return True
    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        return False

def test_server_creation():
    """Test server can be created with functions."""
    print("\nTesting server creation...")
    try:
        from mcp_server_pixeltable.server import main
        print("✅ Server module can be imported")
        
        # We can't actually run main() here as it would start the server
        # But we can test that it imports successfully
        return True
    except Exception as e:
        print(f"❌ Server creation test failed: {e}")
        return False

def create_sample_mcp_config():
    """Create a sample MCP configuration."""
    config = {
        "mcpServers": {
            "pixeltable": {
                "command": "uv",
                "args": ["run", "mcp-server-pixeltable"],
                "env": {
                    "PIXELTABLE_HOME": "~/.pixeltable"
                }
            }
        }
    }
    
    config_file = Path(__file__).parent / "sample_mcp_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Sample MCP config created: {config_file}")
    return config_file

def run_server_test():
    """Run a quick server test."""
    print("\nTesting server startup...")
    try:
        # Test that we can import and create the server without errors
        from mcp.server.fastmcp import FastMCP
        from mcp_server_pixeltable.core.pixeltable_functions import pixeltable_init
        
        # Create test server
        mcp = FastMCP(name="test-pixeltable", version="0.1.0")
        mcp.tool()(pixeltable_init)
        
        print("✅ Server can be created with tools")
        return True
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def show_development_info():
    """Show development information."""
    print("\n" + "="*60)
    print("Development Information")
    print("="*60)
    
    print("\nProject structure:")
    print("├── src/mcp_server_pixeltable/")
    print("│   ├── __init__.py")
    print("│   ├── __main__.py")
    print("│   ├── server.py")
    print("│   ├── utils.py")
    print("│   └── core/")
    print("│       ├── __init__.py")
    print("│       ├── config.py")
    print("│       └── pixeltable_functions.py")
    print("├── pyproject.toml")
    print("├── README.md")
    print("└── install.py")
    
    print("\nDevelopment commands:")
    print("  python dev.py                    # Run this script")
    print("  python install.py               # Setup dependencies")
    print("  uv run mcp-server-pixeltable    # Run server with uv")
    print("  uv sync                         # Sync dependencies")
    
    print("\nTesting with MCP inspector:")
    print("  npx @modelcontextprotocol/inspector uv run mcp-server-pixeltable")

def main():
    """Run development tests."""
    print("Pixeltable MCP Server - Development Testing")
    print("="*50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Pixeltable Availability", test_pixeltable_availability),
        ("MCP Dependencies", test_mcp_dependencies),
        ("Server Creation", test_server_creation),
        ("Server Test", run_server_test),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append(False)
    
    # Create sample config
    create_sample_mcp_config()
    
    # Show summary
    print(f"\n--- Summary ---")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Server is ready for use.")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    # Show development info
    show_development_info()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
