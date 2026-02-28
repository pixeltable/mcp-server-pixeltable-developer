"""
Core functionality for the Pixeltable MCP server.

Modules:
- helpers        – shared utilities (pxt import, output suppression, serialisation, config)
- tables         – table CRUD, views, snapshots, replicas, queries, inserts, computed columns
- directories    – directory CRUD, listing, browsing, moving
- dependencies   – dependency checking, installation, system diagnostics
- udf            – UDF creation, array helpers, type helpers, tool wrappers, MCP connection
- resources      – MCP resource handlers (read-only, return JSON strings)
- repl_functions – REPL session, introspection, bug logging
- canvas_server  – browser canvas display server
"""
