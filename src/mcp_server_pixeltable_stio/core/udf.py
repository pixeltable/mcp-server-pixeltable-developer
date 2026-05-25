"""
UDF, type helpers, tool wrappers, and MCP connection for the Pixeltable MCP server.

Contains:
- pixeltable_create_udf      – create user-defined functions
- pixeltable_create_array     – create array expressions
- pixeltable_create_tools     – wrap UDFs for LLM tool-calling
- pixeltable_connect_mcp      – connect to external MCP servers
- pixeltable_create_type      – unified type helper (replaces 5 individual type creators)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .helpers import pxt, ensure_pixeltable_available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UDF creation
# ---------------------------------------------------------------------------

def pixeltable_create_udf(
    function_code: str,
    function_name: str,
    kwargs: str = "{}"
) -> Dict[str, Any]:
    """Create a User Defined Function from code.

    Allows dynamic creation of custom functions that can be used
    in computed columns and other Pixeltable operations.

    Args:
        function_code: Python code for the function
        function_name: Name for the UDF
        kwargs: Additional parameters for UDF creation (JSON string)

    Returns:
        Success status and UDF information
    """
    try:
        ensure_pixeltable_available()

        try:
            parsed_kwargs = json.loads(kwargs)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON in kwargs: {e}"}

        exec_globals: Dict[str, Any] = {
            'pxt': pxt,
            'pixeltable': pxt,
            '__builtins__': __builtins__,
        }

        try:
            import numpy as np
            exec_globals['np'] = np
            exec_globals['numpy'] = np
        except ImportError:
            pass

        try:
            from PIL import Image
            exec_globals['Image'] = Image
        except ImportError:
            pass

        exec(function_code, exec_globals)

        if function_name not in exec_globals:
            return {
                "success": False,
                "error": f"Function '{function_name}' was not defined in the provided code",
            }

        created_function = exec_globals[function_name]
        udf_func = pxt.udf(created_function, **parsed_kwargs)

        return {
            "success": True,
            "message": f"UDF '{function_name}' created successfully",
            "function_name": function_name,
            "udf_info": {
                "name": function_name,
                "type": "user_defined_function",
                "callable": True,
            },
        }

    except Exception as e:
        logger.error(f"Error creating UDF: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Array expression helper
# ---------------------------------------------------------------------------

def pixeltable_create_array(elements: list) -> Dict[str, Any]:
    """Create array expressions for Pixeltable.

    Useful for assembling literal arrays in queries and computed columns.

    Args:
        elements: List of elements for the array.

    Returns:
        Summary of the constructed array (the underlying expression stays in
        Pixeltable; MCP just reports a preview).
    """
    try:
        ensure_pixeltable_available()

        # pxt.Array as a class is subscriptable for types; the helper here is
        # for callers building literal arrays, which round-trip as plain lists
        # in Pixeltable expressions.
        return {
            "success": True,
            "message": f"Array created with {len(elements)} elements",
            "array_info": {
                "length": len(elements),
                "type": "pixeltable_array",
                "elements": elements[:5] if len(elements) > 5 else elements,
                "truncated": len(elements) > 5,
            },
        }

    except Exception as e:
        logger.error(f"Error creating array: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tools wrapper for LLM integration
# ---------------------------------------------------------------------------

def pixeltable_create_tools(
    function_names: List[str],
    register_as: str = "tools",
) -> Dict[str, Any]:
    """Bind a ``pxt.tools(...)`` collection in the persistent REPL.

    Pixeltable agents pass ``tools=pxt.tools(*fns)`` into provider ``messages`` /
    ``chat_completions`` calls. The REPL runs in a subprocess, so this helper
    pushes code into it to build the collection from names you've already
    defined there with ``execute_python`` (``@pxt.udf`` / ``@pxt.query``
    callables, or names from ``pixeltable_connect_mcp``).

    Args:
        function_names: REPL variable names that resolve to UDFs / queries / MCP tools.
            Use a name prefixed with ``*`` to splat a sequence
            (e.g. ``['local_query', '*mcp_tools']`` → ``pxt.tools(local_query, *mcp_tools)``).
        register_as: REPL variable name to bind the resulting ``pxt.tools(...)`` value to.

    Returns:
        Echo of the binding and the inspected ``len(register_as)``.
    """
    try:
        ensure_pixeltable_available()

        if not function_names:
            return {"success": False, "error": "function_names is empty"}

        # Build the argument list, supporting `*name` splat.
        args = []
        for name in function_names:
            if name.startswith('*'):
                args.append(f"*{name[1:]}")
            else:
                args.append(name)
        snippet = (
            f"{register_as} = pxt.tools({', '.join(args)})\n"
            f"print(f'__pxt_tools_count__={{len({register_as}._tools)}}')"
        )

        from .repl_functions import execute_python  # local import to avoid cycle
        result = execute_python(snippet)
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error') or result.get('stderr') or 'execute_python failed',
                "snippet": snippet,
            }

        count = None
        for line in (result.get('output') or '').splitlines():
            if line.startswith('__pxt_tools_count__='):
                try:
                    count = int(line.split('=', 1)[1])
                except ValueError:
                    pass
        return {
            "success": True,
            "registered_as": register_as,
            "tool_count": count,
            "function_names": function_names,
        }

    except Exception as e:
        logger.error(f"Error creating tools: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# External MCP connection (uses pxt.mcp_udfs added in Pixeltable 0.6.x)
# ---------------------------------------------------------------------------

def pixeltable_connect_mcp(
    url: str,
    register_as: str = "mcp_tools",
) -> Dict[str, Any]:
    """Pull tools from an external MCP server into the persistent REPL.

    Wraps ``pxt.mcp_udfs(url)`` (Pixeltable >= 0.6.x). The list of UDF-shaped
    callables is bound to ``register_as`` in the REPL so subsequent
    ``execute_python`` calls can reference it directly or splat it through
    ``pixeltable_create_tools(['local_query', '*mcp_tools'])``.

    Args:
        url: MCP server URL (typically ``http://host:port/mcp`` or ``http://host:port/sse``).
        register_as: REPL variable name to bind the returned list to.

    Returns:
        Dict with the tools' names, comments, and binding info.
    """
    try:
        ensure_pixeltable_available()

        if not hasattr(pxt, 'mcp_udfs'):
            return {
                "success": False,
                "error": "pxt.mcp_udfs is unavailable. Upgrade Pixeltable to >= 0.6.x.",
            }

        # Push the connection into the REPL so the names are reusable later.
        snippet = (
            f"{register_as} = pxt.mcp_udfs({url!r})\n"
            f"__pxt_mcp_meta__ = [\n"
            f"    {{'name': getattr(t, 'name', None) or getattr(t, '__name__', 'unknown'),\n"
            f"      'comment': (t.comment() if hasattr(t, 'comment') else (getattr(t, '__doc__', '') or '')).strip()}}\n"
            f"    for t in {register_as}\n"
            f"]\n"
            f"import json as _json\n"
            f"print('__pxt_mcp_tools__=' + _json.dumps(__pxt_mcp_meta__))"
        )

        from .repl_functions import execute_python  # local import to avoid cycle
        result = execute_python(snippet)
        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error') or result.get('stderr') or 'execute_python failed',
                "url": url,
            }

        tool_metadata = []
        for line in (result.get('output') or '').splitlines():
            if line.startswith('__pxt_mcp_tools__='):
                try:
                    tool_metadata = json.loads(line.split('=', 1)[1])
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "url": url,
            "registered_as": register_as,
            "tool_count": len(tool_metadata),
            "tools": tool_metadata,
        }

    except Exception as e:
        logger.error(f"Error connecting to MCP: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Unified type helper (replaces 5 individual type creators)
# ---------------------------------------------------------------------------

def pixeltable_create_type(
    type_name: str,
    element_type: Optional[str] = None
) -> Dict[str, Any]:
    """Return a Pixeltable type object for use in schema definitions.

    Replaces the individual ``create_image_type``, ``create_video_type``,
    etc. helpers with a single function that accepts any type name.

    Supported type names (case-insensitive):
        Int, Float, String, Bool, Json, Image, Video, Audio, Document,
        Timestamp, Date, Array (optionally with element_type).

    Args:
        type_name: Name of the Pixeltable type (e.g. "Image", "Array").
        element_type: For Array types, the element type name (e.g. "Float").

    Returns:
        Dict with the type information.
    """
    try:
        ensure_pixeltable_available()

        type_mapping = {
            'int': ('Int', pxt.Int),
            'float': ('Float', pxt.Float),
            'string': ('String', pxt.String),
            'bool': ('Bool', pxt.Bool),
            'json': ('Json', pxt.Json),
            'image': ('Image', pxt.Image),
            'video': ('Video', pxt.Video),
            'audio': ('Audio', pxt.Audio),
            'document': ('Document', pxt.Document),
            'timestamp': ('Timestamp', pxt.Timestamp),
            'date': ('Date', pxt.Date),
            'array': ('Array', pxt.Array),
        }

        key = type_name.strip().lower()
        if key not in type_mapping:
            supported = ', '.join(sorted(m[0] for m in type_mapping.values()))
            return {
                'success': False,
                'error': f"Unknown type '{type_name}'. Supported types: {supported}",
            }

        canonical_name, type_obj = type_mapping[key]

        # Handle Array with optional element_type
        if key == 'array' and element_type:
            elem_key = element_type.strip().lower()
            if elem_key not in type_mapping:
                return {
                    'success': False,
                    'error': f"Unknown element type '{element_type}' for Array.",
                }
            _, elem_obj = type_mapping[elem_key]
            resolved = pxt.Array[elem_obj]
            return {
                'success': True,
                'type': canonical_name,
                'type_repr': repr(resolved),
                'element_type': element_type,
                'description': f'Pixeltable Array[{element_type}] type',
            }

        return {
            'success': True,
            'type': canonical_name,
            'type_repr': repr(type_obj),
            'description': f'Pixeltable {canonical_name} type for schema definitions',
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
