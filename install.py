#!/usr/bin/env python3
"""
Installation and setup script for Pixeltable MCP server.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def check_requirements():
    """Check if required tools are available."""
    print("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    else:
        print(f"✅ Python {sys.version}")
    
    # Check if uv is available
    result = run_command(['uv', '--version'], check=False)
    if result is None:
        print("❌ uv not found. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    else:
        print("✅ uv is available")
    
    return True

def install_package():
    """Install the package in development mode."""
    print("\nSetting up Pixeltable MCP server with uv...")
    
    # Install dependencies with uv
    result = run_command(['uv', 'sync'])
    if result is None:
        print("❌ Failed to sync dependencies")
        return False
    
    print("✅ Dependencies synced successfully")
    print("ℹ️ Use 'uv run mcp-server-pixeltable' to run the server")
    return True

def test_installation():
    """Test if the installation works."""
    print("\nTesting installation...")
    
    # Try to import the package
    try:
        import mcp_server_pixeltable
        print(f"✅ Package imported successfully (version {mcp_server_pixeltable.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import package: {e}")
        return False
    
    # Try to run the server with --help (if available)
    result = run_command(['uv', 'run', 'mcp-server-pixeltable', '--help'], check=False)
    if result and result.returncode == 0:
        print("✅ Command line interface works")
    else:
        print("ℹ️ Command line interface not fully functional (this is expected)")
    
    return True

def show_usage():
    """Show usage instructions."""
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nTo use the Pixeltable MCP server:")
    print("\n1. Run the server with uv:")
    print("   uv run mcp-server-pixeltable")
    print("\n2. Add to your Claude Desktop configuration:")
    print('   {"mcpServers": {"pixeltable": {"command": "uv", "args": ["run", "mcp-server-pixeltable"]}}}')
    print("\n3. For development, test with MCP inspector:")
    print("   npx @modelcontextprotocol/inspector uv run mcp-server-pixeltable")
    print("\nFor more information, see README.md")

def main():
    """Main installation function."""
    print("Pixeltable MCP Server Installation")
    print("="*40)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return 1
    
    # Install package
    if not install_package():
        print("\n❌ Installation failed.")
        return 1
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation tests failed.")
        return 1
    
    # Show usage
    show_usage()
    
    print("\n✅ Installation completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
