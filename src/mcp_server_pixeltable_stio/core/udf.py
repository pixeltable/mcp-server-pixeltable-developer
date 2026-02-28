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

def pixeltable_create_array(elements: list, kwargs: str = "{}") -> Dict[str, Any]:
    """Create array expressions for Pixeltable.

    Useful for creating complex data structures and expressions
    that can be used in queries and computed columns.

    Args:
        elements: List of elements for the array
        kwargs: Additional parameters for array creation (JSON string)

    Returns:
        Array expression result
    """
    try:
        ensure_pixeltable_available()

        array_expr = pxt.Array(elements)

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

def pixeltable_create_tools(udfs: str, kwargs: str = "{}") -> Dict[str, Any]:
    """Create tools collection for LLM integration.

    Wraps UDFs for use with language models and tool-calling APIs.
    Enables integration between Pixeltable functions and AI models.

    Args:
        udfs: UDF functions to wrap as tools
        kwargs: Additional parameters for tool creation (JSON string)

    Returns:
        Tools collection information
    """
    try:
        ensure_pixeltable_available()

        tools = []
        for udf in udfs:
            if callable(udf):
                tool_info = {
                    "name": getattr(udf, '__name__', 'unknown'),
                    "type": "udf_tool",
                    "callable": True,
                    "function": udf,
                }
                tools.append(tool_info)
            else:
                return {
                    "success": False,
                    "error": f"Invalid UDF provided: {udf} is not callable",
                }

        return {
            "success": True,
            "message": f"Created tools collection with {len(tools)} tools",
            "tools": tools,
            "tool_count": len(tools),
        }

    except Exception as e:
        logger.error(f"Error creating tools: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# External MCP connection
# ---------------------------------------------------------------------------

def pixeltable_connect_mcp(url: str, kwargs: str = "{}") -> Dict[str, Any]:
    """Connect to external MCP server and import functions.

    This enables research dataset sharing and function import capability.
    Can be used to pull in functions from academic papers,
    other research groups, or external AI services.

    Args:
        url: URL of the MCP server to connect to
        kwargs: Additional connection parameters (JSON string)

    Returns:
        Connection status and available functions
    """
    try:
        ensure_pixeltable_available()

        return {
            "success": False,
            "error": "MCP connection not yet implemented",
            "message": "This feature is planned for future release",
            "url": url,
            "status": "not_implemented",
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
            type_obj = pxt.Array(elem_obj)
            return {
                'success': True,
                'type': canonical_name,
                'type_object': type_obj,
                'element_type': element_type,
                'description': f'Pixeltable Array[{element_type}] type',
            }

        return {
            'success': True,
            'type': canonical_name,
            'type_object': type_obj,
            'description': f'Pixeltable {canonical_name} type for schema definitions',
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
