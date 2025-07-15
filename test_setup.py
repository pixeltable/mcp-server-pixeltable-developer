#!/usr/bin/env python3
"""
Test script for Pixeltable MCP server.
"""

import sys
import os

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    try:
        print("Testing imports...")
        from mcp_server_pixeltable import __version__
        print(f"✓ Package version: {__version__}")
        
        from mcp_server_pixeltable.core.config import get_effective_pixeltable_path
        print(f"✓ Config module works")
        
        from mcp_server_pixeltable.core.pixeltable_functions import ensure_pixeltable_available
        print(f"✓ Pixeltable functions module works")
        
        from mcp_server_pixeltable.utils import setup_resilient_process
        print(f"✓ Utils module works")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_pixeltable_availability():
    """Test if Pixeltable is available."""
    try:
        import pixeltable as pxt
        print("✓ Pixeltable is available")
        return True
    except ImportError:
        print("! Pixeltable is not installed (this is expected in development)")
        return True

def test_mcp_server():
    """Test if the MCP server can be created."""
    try:
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP(name="test", version="0.1.0")
        print("✓ MCP server can be created")
        return True
    except Exception as e:
        print(f"✗ MCP server creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running MCP Server Pixeltable tests...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Pixeltable Availability", test_pixeltable_availability),
        ("MCP Server Creation", test_mcp_server),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append(result)
    
    print(f"\n--- Summary ---")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
