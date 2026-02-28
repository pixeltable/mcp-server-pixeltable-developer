"""
Shared utilities for the Pixeltable MCP server.

Contains common helpers used across all core modules:
- Output suppression for clean JSON transport
- Pixeltable availability checks
- Result serialization
- Configuration, version, type info, and documentation utilities
"""

import logging
import json
import sys
import io
import os
from typing import Any, Dict, List, Optional, Union
from functools import wraps

# Import pixeltable -- shared by every module via `from .helpers import pxt`
try:
    import pixeltable as pxt
except ImportError as e:
    logging.error(f"Failed to import pixeltable: {e}")
    pxt = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core shared helpers
# ---------------------------------------------------------------------------

def suppress_pixeltable_output(func):
    """Decorator to suppress stdout/stderr during Pixeltable operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        captured_errors = io.StringIO()

        try:
            sys.stdout = captured_output
            sys.stderr = captured_errors
            result = func(*args, **kwargs)
            return result
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            captured = captured_output.getvalue()
            captured_err = captured_errors.getvalue()
            if captured.strip():
                logger.debug(f"Suppressed stdout from {func.__name__}: {captured.strip()}")
            if captured_err.strip():
                logger.debug(f"Suppressed stderr from {func.__name__}: {captured_err.strip()}")

    return wrapper


def ensure_pixeltable_available():
    """Ensure Pixeltable is available and raise an error if not."""
    if pxt is None:
        raise ValueError("Pixeltable is not available. Please install it first.")


def serialize_result(result: Any) -> Dict[str, Any]:
    """Serialize Pixeltable results for JSON transport."""
    try:
        if hasattr(result, 'to_dict'):
            return {"type": "object", "data": result.to_dict()}
        elif hasattr(result, '__dict__'):
            return {"type": "object", "data": {k: str(v) for k, v in result.__dict__.items()}}
        elif isinstance(result, (str, int, float, bool, type(None))):
            return {"type": "primitive", "data": result}
        elif isinstance(result, (list, tuple)):
            return {"type": "list", "data": [serialize_result(item)["data"] for item in result]}
        elif isinstance(result, dict):
            return {"type": "dict", "data": {k: serialize_result(v)["data"] for k, v in result.items()}}
        else:
            return {"type": "string", "data": str(result)}
    except Exception as e:
        logger.warning(f"Failed to serialize result {type(result)}: {e}")
        return {"type": "string", "data": str(result)}


# ---------------------------------------------------------------------------
# Configuration utilities
# ---------------------------------------------------------------------------

def pixeltable_configure_logging(
    to_stdout: Optional[bool] = None,
    level: Optional[int] = None,
    add: Optional[str] = None,
    remove: Optional[str] = None
) -> Dict[str, Any]:
    """Configure logging."""
    try:
        ensure_pixeltable_available()
        pxt.configure_logging(to_stdout=to_stdout, level=level, add=add, remove=remove)
        return {"success": True, "message": "Logging configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring logging: {e}")
        raise ValueError(f"Failed to configure logging: {e}")


@suppress_pixeltable_output
def pixeltable_get_version() -> Dict[str, Any]:
    """Get Pixeltable version information."""
    try:
        ensure_pixeltable_available()
        return {
            "success": True,
            "pixeltable_version": pxt.__version__,
            "mcp_server_version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Error getting version: {e}")
        raise ValueError(f"Failed to get version: {e}")


def pixeltable_get_types() -> Dict[str, Any]:
    """Get available Pixeltable data types."""
    try:
        ensure_pixeltable_available()

        types_info = {
            "basic_types": {
                "Int": "Integer type",
                "Float": "Floating point type",
                "String": "String/text type",
                "Bool": "Boolean type",
                "Json": "JSON object type"
            },
            "temporal_types": {
                "Timestamp": "Timestamp type",
                "Date": "Date type"
            },
            "media_types": {
                "Image": "Image file type",
                "Video": "Video file type",
                "Audio": "Audio file type",
                "Document": "Document file type"
            },
            "complex_types": {
                "Array": "Array type (requires element type)"
            }
        }

        return {"success": True, "types": types_info}
    except Exception as e:
        logger.error(f"Error getting types: {e}")
        raise ValueError(f"Failed to get types: {e}")


# ---------------------------------------------------------------------------
# Datastore configuration
# ---------------------------------------------------------------------------

def pixeltable_set_datastore(path: str) -> Dict[str, Any]:
    """Change the Pixeltable datastore location.

    Updates config.toml and calls pxt.init() to switch to the new location.

    Args:
        path: Path to the datastore directory
    """
    try:
        from mcp_server_pixeltable_stio.core.config import set_datastore_path

        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            os.makedirs(expanded_path, exist_ok=True)

        set_datastore_path(expanded_path)
        pxt.init({'pixeltable.home': expanded_path})
        tables = pxt.list_tables()

        return {
            "success": True,
            "message": f"Switched to datastore: {expanded_path}",
            "path": expanded_path,
            "tables": tables,
            "table_count": len(tables)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def pixeltable_get_datastore() -> Dict[str, Any]:
    """Get the current Pixeltable datastore path configuration."""
    try:
        from mcp_server_pixeltable_stio.core.config import load_config, get_config_path

        config = load_config()
        config_path = get_config_path()
        datastore_path = config.get('storage', {}).get('datastore_path', '~/.pixeltable')
        datastore_path = os.path.expanduser(datastore_path)

        env_path = os.environ.get('PIXELTABLE_HOME')
        currently_active = env_path if env_path else datastore_path
        exists = os.path.exists(currently_active)

        try:
            tables = pxt.list_tables()
        except Exception:
            tables = []

        return {
            "success": True,
            "config_file": config_path,
            "configured_path": datastore_path,
            "currently_active": currently_active,
            "exists": exists,
            "tables": tables,
            "table_count": len(tables),
            "cache_size_gb": config.get('cache', {}).get('file_cache_size_gb', 10)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Informational / meta utilities
# ---------------------------------------------------------------------------

def pixeltable_list_functions() -> Dict[str, Any]:
    """List all registered functions."""
    try:
        ensure_pixeltable_available()
        styled_df = pxt.list_functions()
        df = styled_df.data
        functions = df.to_dict('records')
        return {"success": True, "functions": functions, "count": len(functions)}
    except Exception as e:
        logger.error(f"Error listing functions: {e}")
        raise ValueError(f"Failed to list functions: {e}")


def pixeltable_get_help() -> Dict[str, Any]:
    """Get comprehensive help and overview of Pixeltable concepts and workflows."""
    return {
        "success": True,
        "overview": {
            "what_is_pixeltable": (
                "Pixeltable is a Python framework for multimodal AI applications that provides "
                "incremental storage, transformation, indexing, and orchestration of data. "
                "It's built on PostgreSQL but extends it with native support for images, video, "
                "audio, and documents alongside structured data."
            ),
            "key_innovation": (
                "Unlike traditional databases, Pixeltable uses 'computed columns' that automatically "
                "run AI models and transformations on your data. When you add new data, all dependent "
                "computations update automatically - this is called 'incremental computation'."
            ),
            "architecture": {
                "storage": "Data stored in ~/.pixeltable with PostgreSQL for metadata and flat files for media",
                "computation": "Declarative computed columns that auto-update when data changes",
                "versioning": "Automatic versioning and time-travel queries for all data changes"
            }
        },
        "core_concepts": {
            "tables": (
                "Tables store your multimodal data. Unlike SQL tables, they natively handle "
                "images, video, audio, and documents. Create with pixeltable_create_table()."
            ),
            "computed_columns": (
                "Columns that automatically compute values using Python expressions or AI models. "
                "Example: Add object detection that runs on every image automatically. "
                "Use pixeltable_add_computed_column()."
            ),
            "views": (
                "Filtered or transformed views of tables. Like SQL views but with full "
                "multimodal support. Changes to base table automatically propagate."
            ),
            "snapshots": (
                "Immutable copies of a table at a point in time. Useful for reproducibility "
                "and comparing model outputs across versions."
            ),
            "incremental_updates": (
                "When you add data or change a computed column, Pixeltable only recomputes "
                "what's necessary, saving time and compute costs."
            )
        },
        "data_types": {
            "multimodal": ["pxt.Image", "pxt.Video", "pxt.Audio", "pxt.Document"],
            "structured": ["String", "Int", "Float", "Bool", "Json", "Array"],
            "special": ["Embeddings (for vector search)", "File paths and URLs"]
        },
        "typical_workflows": {
            "1_basic_flow": [
                "Create table: pixeltable_create_table() with schema",
                "Insert data: pixeltable_insert_data() with images/videos/text",
                "Add AI: pixeltable_add_computed_column() with model inference",
                "Query: pixeltable_query_table() to get results"
            ],
            "2_computer_vision": [
                "Create table with pxt.Image column",
                "Insert images from directory or URLs",
                "Add YOLOX for object detection: 'yolox.yolox(image, threshold=0.5)'",
                "Add OpenAI Vision: 'openai.vision(prompt, image)'",
                "Query results with filters"
            ],
            "3_rag_pipeline": [
                "Create table with pxt.Document column",
                "Insert documents (PDFs, text files)",
                "Add embedding column: 'sentence_transformers.embed(text)'",
                "Create embedding index for vector search",
                "Query with similarity search"
            ],
            "4_video_analysis": [
                "Create table with pxt.Video column",
                "Extract frames as computed column",
                "Run models on frames (object detection, classification)",
                "Aggregate results across frames"
            ]
        },
        "ai_integrations": {
            "cloud_providers": ["OpenAI (GPT, DALL-E, Whisper)", "Anthropic (Claude)", "Google (Gemini)", "Fireworks"],
            "local_models": ["Ollama", "Sentence Transformers", "YOLOX", "Hugging Face models"],
            "custom_models": "Create UDFs with pixeltable_create_udf() for any Python function"
        },
        "best_practices": [
            "Use computed columns instead of manual processing - they auto-update",
            "Leverage incremental computation - only new data gets processed",
            "Create views for different use cases instead of duplicating data",
            "Use snapshots before major changes for rollback capability",
            "Set appropriate num_retained_versions to manage storage"
        ],
        "common_patterns": {
            "batch_inference": "Add computed column → Pixeltable handles batching automatically",
            "model_comparison": "Add multiple computed columns with different models → Query to compare",
            "data_validation": "Use computed columns with Python expressions for validation rules",
            "feature_engineering": "Chain computed columns for complex transformations"
        },
        "getting_started": {
            "simple_example": {
                "description": "Analyze images with AI",
                "steps": [
                    "pixeltable_create_table('images', {'image': 'Image', 'label': 'String'})",
                    "pixeltable_insert_data('images', [{'image': 'cat.jpg', 'label': 'cat'}])",
                    "pixeltable_add_computed_column('images', 'objects', 'yolox.yolox(image)')",
                    "pixeltable_query_table('images') - See detected objects"
                ]
            }
        },
        "tips": [
            "Check dependencies before adding AI columns: pixeltable_check_dependencies()",
            "Use the pixeltable://tools resource to see all available operations",
            "Set custom datastore path: pixeltable_set_datastore('/my/path')",
            "Use execute_python() for interactive exploration with pxt pre-loaded"
        ]
    }


def pixeltable_list_tools() -> Dict[str, Any]:
    """List all available Pixeltable MCP tools with their descriptions."""
    try:
        import inspect
        import mcp_server_pixeltable_stio.core.pixeltable_functions as pf
        import mcp_server_pixeltable_stio.core.repl_functions as rf

        pixeltable_funcs = [
            (name, func) for name, func in inspect.getmembers(pf)
            if name.startswith('pixeltable_') and callable(func)
        ]

        repl_funcs = [
            (name, func) for name, func in inspect.getmembers(rf)
            if callable(func) and not name.startswith('_')
        ]

        categories = {
            "Table Management": ["create_table", "get_table", "list_tables", "drop_table", "query_table", "get_table_schema"],
            "Data Operations": ["insert_data", "add_computed_column", "create_view", "create_snapshot", "create_replica"],
            "Directory Management": ["create_dir", "drop_dir", "list_dirs", "ls", "move"],
            "Configuration": ["set_datastore", "get_datastore", "configure_logging", "get_version", "list_tools"],
            "AI/ML Integration": ["create_udf", "create_array", "create_tools", "connect_mcp", "query"],
            "Data Types": ["create_type"],
            "Dependencies": ["check_dependencies", "install_dependency"],
            "REPL & Debug": ["execute_python", "introspect_function", "list_available_functions", "log_bug", "log_missing_feature", "generate_bug_report"],
            "Utilities": ["list_functions", "get_types", "system_diagnostics", "search_docs"],
        }

        categorized: Dict[str, list] = {cat: [] for cat in categories}
        categorized["Other"] = []

        all_funcs = pixeltable_funcs + repl_funcs

        for name, func in all_funcs:
            doc = ""
            if func.__doc__:
                lines = func.__doc__.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        doc = line
                        break

            found = False
            for cat, keywords in categories.items():
                if any(kw in name.lower() for kw in keywords):
                    categorized[cat].append({
                        "name": name,
                        "description": doc or "No description available"
                    })
                    found = True
                    break

            if not found:
                categorized["Other"].append({
                    "name": name,
                    "description": doc or "No description available"
                })

        result = {}
        total_count = 0
        for cat, tools in categorized.items():
            if tools:
                result[cat] = sorted(tools, key=lambda x: x["name"])
                total_count += len(tools)

        return {
            "success": True,
            "total_tools": total_count,
            "categories": result,
            "message": f"Found {total_count} tools across {len(result)} categories"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def pixeltable_search_docs(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search Pixeltable documentation using Mintlify's MCP endpoint.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    """
    try:
        import requests

        response = requests.post(
            "https://docs.pixeltable.com/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json"
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "SearchPixeltableDocumentation",
                    "arguments": {"query": query}
                }
            },
            timeout=10
        )

        results = []
        for line in response.text.split('\n'):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'result' in data and 'content' in data['result']:
                    for item in data['result']['content'][:max_results]:
                        text = item.get('text', '')
                        lines = text.split('\n')
                        title = ''
                        link = ''
                        content = ''

                        for i, line_text in enumerate(lines):
                            if line_text.startswith('Title: '):
                                title = line_text[7:]
                            elif line_text.startswith('Link: '):
                                link = line_text[6:]
                            elif line_text.startswith('Content: '):
                                content = '\n'.join(lines[i:]).replace('Content: ', '', 1)
                                break

                        if title and link:
                            results.append({
                                'title': title,
                                'link': link,
                                'snippet': content[:500] if content else 'No content available'
                            })
                    break

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
            "message": f"Found {len(results)} results for '{query}'"
        }

    except Exception as req_err:
        # Catch both requests exceptions and general exceptions
        return {
            "success": False,
            "error": f"Error searching docs: {str(req_err)}"
        }
