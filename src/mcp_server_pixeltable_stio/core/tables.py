"""
Table management for the Pixeltable MCP server.

Covers table CRUD, views, snapshots, replicas, queries, inserts,
computed columns, and schema introspection.
"""

import logging
import sys
import io
import os
from typing import Any, Dict, List, Optional, Union

from .helpers import pxt, ensure_pixeltable_available, suppress_pixeltable_output, serialize_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema type resolution
# ---------------------------------------------------------------------------

_BASE_PXT_TYPES = (
    'Int', 'String', 'Float', 'Bool', 'Json',
    'Image', 'Video', 'Audio', 'Document',
    'Timestamp', 'Date', 'UUID', 'Binary',
)


def _resolve_base(name: str):
    """Map a Pixeltable type name (case-insensitive) to its `pxt.<Type>` object."""
    key = name.strip()
    for candidate in _BASE_PXT_TYPES:
        if candidate.lower() == key.lower():
            return getattr(pxt, candidate)
    raise ValueError(f"Unknown Pixeltable type: {name!r}")


def _resolve_pxt_type(spec):
    """Resolve a schema type spec to a real Pixeltable type object.

    Supported shapes:
      - actual Pixeltable type / annotated object (passed through)
      - "String", "Image", "Float" (case-insensitive)
      - "Required[String]"
      - "Array[Float]", "Array[Int]", "Array[String]"
      - "Required[Array[Float]]"
      - {"type": "String", "required": true}
      - {"type": "Array", "element_type": "Float"}
      - {"type": "Array", "element_type": "Float", "required": true}

    Raises ValueError for any unknown / malformed spec (no silent fallback).
    """
    if spec is None:
        raise ValueError('Column type cannot be None')

    if isinstance(spec, dict):
        type_name = spec.get('type')
        if not isinstance(type_name, str):
            raise ValueError(f"Schema dict must include string 'type', got: {spec!r}")
        if type_name.lower() == 'array':
            element_name = spec.get('element_type') or spec.get('element') or 'Float'
            element = _resolve_base(element_name)
            inner = pxt.Array[element]
        else:
            inner = _resolve_base(type_name)
        if spec.get('required'):
            inner = pxt.Required[inner]
        return inner

    if not isinstance(spec, str):
        # actual pxt type / annotated; passed through.
        return spec

    text = spec.strip()
    if not text:
        raise ValueError('Empty type spec')

    lower = text.lower()
    if lower.startswith('required[') and text.endswith(']'):
        inner_spec = text[len('Required['): -1]
        return pxt.Required[_resolve_pxt_type(inner_spec)]
    if lower.startswith('array[') and text.endswith(']'):
        element_spec = text[len('Array['): -1].strip()
        # Only base types are allowed as Array element (no nested Array / Required).
        element = _resolve_base(element_spec)
        return pxt.Array[element]
    return _resolve_base(text)


# ---------------------------------------------------------------------------
# Init / health-check
# ---------------------------------------------------------------------------

def pixeltable_init(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check Pixeltable initialization and recover from circular-init failures.

    Args:
        config_overrides: Optional dict passed straight to ``pxt.init(config_overrides=...)``
            (0.6.x signature).
    """
    try:
        ensure_pixeltable_available()

        logger.info(f"Checking Pixeltable status with PIXELTABLE_HOME={os.environ.get('PIXELTABLE_HOME')}")

        try:
            tables = pxt.list_tables()
            return {
                "success": True,
                "message": "Pixeltable is initialized and working",
                "version": pxt.__version__,
                "table_count": len(tables)
            }
        except Exception as e:
            error_msg = str(e)
            if "Circular env initialization detected" not in error_msg:
                return {
                    "success": False,
                    "message": f"Pixeltable is imported but not fully functional: {e}",
                    "version": pxt.__version__
                }

            try:
                logger.info("Attempting to reset Pixeltable's circular initialization flag")
                from pixeltable.env import Env
                from pixeltable.config import Config

                if hasattr(Env, '_Env__initializing'):
                    Env._Env__initializing = False
                if hasattr(Env, '_instance') and Env._instance is not None:
                    Env._instance = None
                if hasattr(Config, '_Config__instance') and Config._Config__instance is not None:
                    Config._Config__instance = None

                current_home = os.environ.get('PIXELTABLE_HOME')
                if current_home and current_home.startswith('~'):
                    os.environ['PIXELTABLE_HOME'] = os.path.expanduser(current_home)
                elif not current_home:
                    os.environ['PIXELTABLE_HOME'] = os.path.expanduser('~/.pixeltable')

                logger.info("Attempting fresh initialization after reset")
                pxt.init(config_overrides=config_overrides)

                tables = pxt.list_tables()
                return {
                    "success": True,
                    "message": "Pixeltable recovered from circular initialization after reset",
                    "version": pxt.__version__,
                    "table_count": len(tables)
                }
            except Exception as e2:
                logger.error(f"Recovery attempt failed: {e2}")
                return {
                    "success": False,
                    "message": f"Circular initialization detected and recovery failed: {e2}",
                    "version": pxt.__version__,
                    "original_error": error_msg
                }

    except Exception as e:
        logger.error(f"Error checking Pixeltable: {e}")
        raise ValueError(f"Failed to check Pixeltable: {e}")


# ---------------------------------------------------------------------------
# Table CRUD
# ---------------------------------------------------------------------------

def pixeltable_create_table(
    path: str,
    schema: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    source_format: Optional[str] = None,
    schema_overrides: Optional[Dict[str, Any]] = None,
    on_error: str = 'abort',
    primary_key: Optional[Union[str, List[str]]] = None,
    create_default_idxs: bool = True,
    comment: Optional[str] = None,
    custom_metadata: Optional[Any] = None,
    media_validation: str = 'on_write',
    if_exists: str = 'error',
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new base table.

    Schema column types accept string forms ("Image", "Required[String]", "Array[Float]")
    or dict forms ({"type": "String", "required": True}). Unknown types raise an error
    instead of silently falling back to String. Per pixeltable-skill anti-patterns, prefer
    `pxt.Required[T]` (or `Required[T]` strings) for primary keys.
    """
    try:
        ensure_pixeltable_available()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            if schema:
                converted_schema = {}
                for col_name, col_type in schema.items():
                    try:
                        converted_schema[col_name] = _resolve_pxt_type(col_type)
                    except ValueError as e:
                        return {
                            "success": False,
                            "error": f"Column '{col_name}': {e}",
                        }
                schema = converted_schema

            table = pxt.create_table(
                path,
                schema=schema,
                source=source,
                source_format=source_format,
                schema_overrides=schema_overrides,
                on_error=on_error,
                primary_key=primary_key,
                create_default_idxs=create_default_idxs,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                if_exists=if_exists,
                extra_args=extra_args,
            )

            return {"success": True, "table_path": path}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        return {"success": False, "error": str(e)}


def pixeltable_get_table(path: str) -> Dict[str, Any]:
    """Get a handle to an existing table, view, or snapshot."""
    try:
        ensure_pixeltable_available()
        table = pxt.get_table(path)
        return {
            "success": True,
            "message": f"Table '{path}' retrieved successfully",
            "table_path": str(table._path()),
            "table_info": serialize_result(table)
        }
    except Exception as e:
        logger.error(f"Error getting table: {e}")
        raise ValueError(f"Failed to get table: {e}")


@suppress_pixeltable_output
def pixeltable_list_tables(dir_path: str = '', recursive: bool = True) -> Dict[str, Any]:
    """List tables in a directory."""
    try:
        ensure_pixeltable_available()
        tables = pxt.list_tables(dir_path, recursive=recursive)
        return {"success": True, "tables": tables, "count": len(tables)}
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise ValueError(f"Failed to list tables: {e}")


def pixeltable_drop_table(
    table: str,
    force: bool = False,
    if_not_exists: str = 'error'
) -> Dict[str, Any]:
    """Drop a table, view, or snapshot."""
    try:
        ensure_pixeltable_available()
        pxt.drop_table(table, force=force, if_not_exists=if_not_exists)
        return {"success": True, "message": f"Table '{table}' dropped successfully"}
    except Exception as e:
        logger.error(f"Error dropping table: {e}")
        raise ValueError(f"Failed to drop table: {e}")


# ---------------------------------------------------------------------------
# Views & Snapshots
# ---------------------------------------------------------------------------

def _build_iterator(base_table, iterator_kind: str, kwargs: Dict[str, Any]):
    """Build a GeneratingFunctionCall iterator for create_view / create_snapshot.

    `iterator_kind` selects the iterator module/function; column-reference kwargs
    (those whose value starts with 'table.') are resolved against `base_table`.

    Supported kinds (skill workflows.md / providers.md):
      - 'frame_iterator'      : video frames     (kwargs: video=table.video, fps=...)
      - 'document_splitter'   : document chunks  (kwargs: document=table.document,
                                                          separators='token_limit',
                                                          limit=300)
      - 'audio_splitter'      : audio chunks     (kwargs: audio=table.audio, duration=30.0)
      - 'string_splitter'     : sentence chunks  (kwargs: text=table.text,
                                                          separators='sentence')
    """
    if iterator_kind == 'frame_iterator':
        from pixeltable.functions.video import frame_iterator as iter_fn
    elif iterator_kind == 'document_splitter':
        from pixeltable.functions.document import document_splitter as iter_fn
    elif iterator_kind == 'audio_splitter':
        from pixeltable.functions.audio import audio_splitter as iter_fn
    elif iterator_kind == 'string_splitter':
        from pixeltable.functions.string import string_splitter as iter_fn
    else:
        raise ValueError(
            f"Unknown iterator '{iterator_kind}'. "
            "Expected one of: frame_iterator, document_splitter, audio_splitter, string_splitter."
        )

    resolved = {}
    for key, value in (kwargs or {}).items():
        if isinstance(value, str) and value.startswith('table.'):
            attr_name = value[len('table.'):]
            if not hasattr(base_table, attr_name):
                raise ValueError(f"Iterator kwarg '{key}': base table has no column '{attr_name}'")
            resolved[key] = getattr(base_table, attr_name)
        else:
            resolved[key] = value
    return iter_fn(**resolved)


def pixeltable_create_view(
    path: str,
    base_table_path: str,
    additional_columns: Optional[Dict[str, Any]] = None,
    is_snapshot: bool = False,
    create_default_idxs: bool = False,
    iterator: Optional[str] = None,
    iterator_kwargs: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
    custom_metadata: Optional[Any] = None,
    media_validation: str = 'on_write',
    if_exists: str = 'error',
) -> Dict[str, Any]:
    """Create a view of an existing table.

    Pass `iterator` ('frame_iterator', 'document_splitter', 'audio_splitter',
    'string_splitter') with `iterator_kwargs` to build chunked or sampled views
    without dropping into `execute_python`. Column-reference values are written
    as strings like 'table.video' / 'table.document' / 'table.audio' / 'table.text'
    and resolved against the base table.

    Example:
        pixeltable_create_view(
            path='proj.frames', base_table_path='proj.videos',
            iterator='frame_iterator',
            iterator_kwargs={'video': 'table.video', 'fps': 1.0},
            if_exists='ignore',
        )
    """
    try:
        ensure_pixeltable_available()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            base_table = pxt.get_table(base_table_path)
            iterator_call = None
            if iterator:
                iterator_call = _build_iterator(base_table, iterator, iterator_kwargs or {})

            pxt.create_view(
                path=path,
                base=base_table,
                additional_columns=additional_columns,
                is_snapshot=is_snapshot,
                create_default_idxs=create_default_idxs,
                iterator=iterator_call,
                comment=comment,
                custom_metadata=custom_metadata,
                media_validation=media_validation,
                if_exists=if_exists,
            )
            return {"success": True, "view_path": path}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        return {"success": False, "error": str(e)}


def pixeltable_create_snapshot(
    path: str,
    base_table_path: str,
    additional_columns: Optional[Dict[str, Any]] = None,
    iterator: Optional[str] = None,
    iterator_kwargs: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
    custom_metadata: Optional[Any] = None,
    media_validation: str = 'on_write',
    if_exists: str = 'error',
) -> Dict[str, Any]:
    """Create a snapshot of an existing table.

    Accepts the same iterator + iterator_kwargs as create_view.
    """
    try:
        ensure_pixeltable_available()
        base_table = pxt.get_table(base_table_path)
        iterator_call = None
        if iterator:
            iterator_call = _build_iterator(base_table, iterator, iterator_kwargs or {})

        snapshot = pxt.create_snapshot(
            path_str=path,
            base=base_table,
            additional_columns=additional_columns,
            iterator=iterator_call,
            comment=comment,
            custom_metadata=custom_metadata,
            media_validation=media_validation,
            if_exists=if_exists,
        )
        return {
            "success": True,
            "message": f"Snapshot '{path}' created successfully",
            "snapshot_path": str(snapshot._path()) if snapshot else path,
        }
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        raise ValueError(f"Failed to create snapshot: {e}")


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def pixeltable_get_table_schema(table_path: str) -> Dict[str, Any]:
    """Get the schema of a table."""
    try:
        ensure_pixeltable_available()
        table = pxt.get_table(table_path)
        columns = []
        for col in table._tbl_version_path.columns():
            columns.append({
                "name": col.name,
                "type": str(col.col_type),
                "nullable": col.col_type.nullable
            })
        return {
            "success": True,
            "table_path": table_path,
            "columns": columns,
            "column_count": len(columns)
        }
    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        raise ValueError(f"Failed to get table schema: {e}")


# ---------------------------------------------------------------------------
# Replicas
# ---------------------------------------------------------------------------

def pixeltable_create_replica(destination: str, source: str) -> Dict[str, Any]:
    """Replicate or publish a table (0.6.x: replaces the old create_replica API).

    - If `source` looks like a remote URI (pxt://, http://, https://, pxtfs://),
      pulls it into the local catalog at `destination` via `pxt.replicate(source, destination)`.
    - Otherwise, `source` is a local table path and `destination` is a remote URI,
      so we use `pxt.publish(source, destination)`.
    """
    try:
        ensure_pixeltable_available()
        remote_prefixes = ('pxt://', 'pxtfs://', 'http://', 'https://', 's3://', 'gs://', 'az://')

        if source.startswith(remote_prefixes):
            pxt.replicate(remote_uri=source, local_path=destination)
            return {
                "success": True,
                "message": f"Replicated remote '{source}' into local '{destination}'",
            }

        if destination.startswith(remote_prefixes):
            source_table = pxt.get_table(source)
            pxt.publish(source=source_table, destination_uri=destination)
            return {
                "success": True,
                "message": f"Published local '{source}' to remote '{destination}'",
            }

        return {
            "success": False,
            "error": (
                "Either source or destination must be a remote URI "
                "(pxt://, pxtfs://, http(s)://, s3://, gs://, az://)."
            ),
        }
    except Exception as e:
        logger.error(f"Error creating replica: {e}")
        raise ValueError(f"Failed to create replica: {e}")


# ---------------------------------------------------------------------------
# Queries & inserts
# ---------------------------------------------------------------------------

def pixeltable_query_table(table_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Execute a simple query on a table."""
    try:
        ensure_pixeltable_available()
        table = pxt.get_table(table_path)
        df = table.select()
        if limit:
            df = df.limit(limit)
        result_set = df.collect()

        # ResultSet rows are dicts: {'col': value, ...}
        rows_raw = [dict(row) for row in result_set]
        col_names = list(rows_raw[0].keys()) if rows_raw else []

        def _safe(v: Any) -> Any:
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            return str(v)

        return {
            "success": True,
            "columns": col_names,
            "data": [[_safe(row[c]) for c in col_names] for row in rows_raw],
            "row_count": len(rows_raw),
        }
    except Exception as e:
        logger.error(f"Error querying table: {e}")
        raise ValueError(f"Failed to query table: {e}")


def pixeltable_insert_data(table_path: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Insert data into a table."""
    try:
        ensure_pixeltable_available()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            table = pxt.get_table(table_path)
            table.insert(data)
            return {"success": True, "rows_inserted": len(data)}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        return {"success": False, "error": str(e)}


def pixeltable_query(
    table_path: str,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generic query interface for Pixeltable.

    Args:
        table_path: Path to the table to query
        limit: Maximum number of rows to return
        columns: List of columns to select (if None, selects all)
    """
    try:
        ensure_pixeltable_available()
        table = pxt.get_table(table_path)

        # Resolve column name strings to actual column references
        if columns:
            col_refs = []
            for col_name in columns:
                if hasattr(table, col_name):
                    col_refs.append(getattr(table, col_name))
                else:
                    return {
                        "success": False,
                        "error": f"Column '{col_name}' not found in table '{table_path}'",
                    }
            result = table.select(*col_refs)
        else:
            result = table.select()

        if limit:
            result = result.limit(limit)
        result_set = result.collect()

        # ResultSet rows are dicts: {'col': value, ...}
        rows_raw = [dict(row) for row in result_set]
        col_names = list(rows_raw[0].keys()) if rows_raw else []

        def _safe(v: Any) -> Any:
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            return str(v)

        return {
            "success": True,
            "columns": col_names,
            "data": [[_safe(row[c]) for c in col_names] for row in rows_raw],
            "row_count": len(rows_raw),
            "table_path": table_path,
            "query_info": {"columns": columns, "limit": limit}
        }
    except Exception as e:
        logger.error(f"Error in pixeltable_query: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Computed columns
# ---------------------------------------------------------------------------

def pixeltable_add_computed_column(
    table_path: str,
    column_name: str,
    expression: str,
    if_exists: str = 'error',
    auto_install: bool = False
) -> Dict[str, Any]:
    """Add a computed column to an existing table with smart dependency management.

    Args:
        table_path: Path to the table
        column_name: Name of the new computed column
        expression: Python expression string defining the computation
        if_exists: What to do if column exists ('error', 'replace', 'ignore')
        auto_install: Whether to automatically install missing dependencies

    Example expressions:
        - "yolox.yolox(table.image, model_id='yolox_s', threshold=0.5)"
        - "openai.chat_completions(...)" with image content blocks for vision (not openai.vision)
        - "image.width(table.image)"
    """
    try:
        ensure_pixeltable_available()

        # Import dependency helpers (avoids circular import at module level)
        from .dependencies import (
            check_dependencies,
            import_pixeltable_yolox_module,
            pixeltable_auto_install_for_expression,
        )

        deps = check_dependencies(expression)

        if not deps['all_satisfied'] and not auto_install:
            missing_info = []
            for dep in deps['missing']:
                missing_info.append(f"• {dep['name']}: {dep['description']} ({dep['size']}, {dep['time']})")
            suggestion = "\n".join([
                "Install missing dependencies first:",
                *[f"  pixeltable_install_dependency('{dep['name']}')" for dep in deps['missing']],
                "",
                "Or run with auto_install=True to install automatically"
            ])
            return {
                "success": False,
                "error": "Missing dependencies",
                "missing_dependencies": deps['missing'],
                "details": "\n".join(missing_info),
                "suggestion": suggestion
            }

        if not deps['all_satisfied'] and auto_install:
            logger.info("Auto-installing missing dependencies...")
            install_result = pixeltable_auto_install_for_expression(expression)
            if not install_result.get('success', False):
                return {
                    "success": False,
                    "error": "Failed to auto-install dependencies",
                    "install_error": install_result.get('error', 'Unknown error'),
                    "failed_dependencies": install_result.get('failed', [])
                }

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            table = pxt.get_table(table_path)

            eval_context = {
                'table': table,
                'pxt': pxt,
                'pixeltable': pxt,
            }

            yolox_mod = import_pixeltable_yolox_module()
            if yolox_mod is not None:
                eval_context['yolox'] = yolox_mod

            # Best-effort import of every provider/iterator module the skill references.
            # Each name lives under pixeltable.functions.<name>; missing optional deps
            # (e.g. anthropic, whisper) are skipped silently so expressions that don't
            # use them still evaluate.
            _provider_modules = [
                # LLM / chat providers (skill providers.md)
                'openai', 'anthropic', 'gemini', 'together', 'fireworks',
                'ollama', 'mistralai', 'groq', 'deepseek', 'openrouter',
                'replicate', 'bedrock', 'fabric', 'llama_cpp',
                # Embeddings / multimodal
                'huggingface', 'voyageai', 'jina', 'twelvelabs',
                # Image / video generation
                'bfl', 'runwayml', 'fal', 'reve',
                # Audio
                'whisper', 'whisperx',
                # Iterators and utilities
                'video', 'document', 'audio', 'string', 'image', 'math', 'uuid',
            ]
            for _name in _provider_modules:
                try:
                    _mod = __import__(f'pixeltable.functions.{_name}', fromlist=[_name])
                    eval_context.setdefault(_name, _mod)
                except ImportError:
                    continue

            # uuid7() is the common entrypoint from pixeltable.functions.uuid
            try:
                from pixeltable.functions.uuid import uuid7  # type: ignore
                eval_context['uuid7'] = uuid7
            except ImportError:
                pass

            try:
                computed_expr = eval(expression, eval_context)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to evaluate expression '{expression}': {e}"
                }

            kwargs = {column_name: computed_expr}
            if if_exists != 'error':
                kwargs['if_exists'] = if_exists
            table.add_computed_column(**kwargs)

            columns = []
            for col in table._tbl_version_path.columns():
                columns.append({
                    "name": col.name,
                    "type": str(col.col_type),
                    "nullable": col.col_type.nullable
                })

            return {
                "success": True,
                "message": f"Computed column '{column_name}' added successfully",
                "table_path": table_path,
                "column_name": column_name,
                "expression": expression,
                "dependencies_used": deps['available'],
                "updated_schema": columns
            }

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        return {"success": False, "error": str(e)}
