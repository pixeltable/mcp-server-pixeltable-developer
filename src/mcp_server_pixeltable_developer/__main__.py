"""Command-line entry point for the stdio MCP server."""

from __future__ import annotations

import argparse
import logging

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mcp-server-pixeltable-developer",
        description="Pixeltable developer MCP server over stdio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .server import main as run_server

    run_server()


if __name__ == "__main__":
    main()
