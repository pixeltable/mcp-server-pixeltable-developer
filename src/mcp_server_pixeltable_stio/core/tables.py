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
# Init / health-check
# ---------------------------------------------------------------------------

def pixeltable_init(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check Pixeltable initialization status and try to resolve issues."""
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
            if "Circular env initialization detected" in error_msg:
                try:
                    logger.info("Attempting to reset Pixeltable's circular initialization flag")
                    from pixeltable.env import Env
                    from pixeltable.config import Config

                    if hasattr(Env, '_Env__initializing'):
                        logger.info(f"Current __initializing state: {Env._Env__initializing}")
                        Env._Env__initializing = False
                        logger.info("Reset __initializing flag to False")

                    if hasattr(Env, '_instance') and Env._instance is not None:
                        logger.info("Found existing Env instance, clearing it")
                        Env._instance = None

                    if hasattr(Config, '_Config__instance') and Config._Config__instance is not None:
                        logger.info("Found existing Config instance, clearing it")
                        Config._Config__instance = None

                    os.environ['PIXELTABLE_FILE_CACHE_SIZE_G'] = '100'
                    logger.info("Set PIXELTABLE_FILE_CACHE_SIZE_G=100")

                    current_home = os.environ.get('PIXELTABLE_HOME')
                    logger.info(f"Current PIXELTABLE_HOME: {current_home}")
                    if current_home and current_home.startswith('~'):
                        expanded_home = os.path.expanduser(current_home)
                        os.environ['PIXELTABLE_HOME'] = expanded_home
                        logger.info(f"Expanded PIXELTABLE_HOME from {current_home} to {expanded_home}")
                    elif not current_home:
                        default_home = os.path.expanduser('~/.pixeltable')
                        os.environ['PIXELTABLE_HOME'] = default_home
                        logger.info(f"Set PIXELTABLE_HOME to default: {default_home}")

                    logger.info("Attempting fresh initialization after reset")
                    if config_overrides:
                        pxt.init(**config_overrides)
                    else:
                        pxt.init()

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
            else:
                return {
                    "success": False,
                    "message": f"Pixeltable is imported but not fully functional: {e}",
                    "version": pxt.__version__
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
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error',
    extra_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new base table."""
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
                    if isinstance(col_type, str):
                        type_mapping = {
                            'int': pxt.Int, 'Int': pxt.Int,
                            'string': pxt.String, 'String': pxt.String,
                            'float': pxt.Float, 'Float': pxt.Float,
                            'bool': pxt.Bool, 'Bool': pxt.Bool,
                            'json': pxt.Json, 'Json': pxt.Json,
                            'image': pxt.Image, 'Image': pxt.Image,
                            'video': pxt.Video, 'Video': pxt.Video,
                            'audio': pxt.Audio, 'Audio': pxt.Audio,
                            'document': pxt.Document, 'Document': pxt.Document,
                            'timestamp': pxt.Timestamp, 'Timestamp': pxt.Timestamp,
                            'date': pxt.Date, 'Date': pxt.Date,
                        }
                        converted_schema[col_name] = type_mapping.get(col_type, pxt.String)
                    else:
                        converted_schema[col_name] = col_type
                schema = converted_schema

            table = pxt.create_table(
                path,
                schema=schema,
                source=source,
                source_format=source_format,
                schema_overrides=schema_overrides,
                on_error=on_error,
                primary_key=primary_key,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
                if_exists=if_exists,
                extra_args=extra_args
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

def pixeltable_create_view(
    path: str,
    base_table_path: str,
    additional_columns: Optional[Dict[str, Any]] = None,
    is_snapshot: bool = False,
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error'
) -> Dict[str, Any]:
    """Create a view of an existing table."""
    try:
        ensure_pixeltable_available()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            base_table = pxt.get_table(base_table_path)
            view = pxt.create_view(
                path=path,
                base=base_table,
                additional_columns=additional_columns,
                is_snapshot=is_snapshot,
                num_retained_versions=num_retained_versions,
                comment=comment,
                media_validation=media_validation,
                if_exists=if_exists
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
    num_retained_versions: int = 10,
    comment: str = '',
    media_validation: str = 'on_write',
    if_exists: str = 'error'
) -> Dict[str, Any]:
    """Create a snapshot of an existing table."""
    try:
        ensure_pixeltable_available()
        base_table = pxt.get_table(base_table_path)
        snapshot = pxt.create_snapshot(
            path_str=path,
            base=base_table,
            additional_columns=additional_columns,
            num_retained_versions=num_retained_versions,
            comment=comment,
            media_validation=media_validation,
            if_exists=if_exists
        )
        return {
            "success": True,
            "message": f"Snapshot '{path}' created successfully",
            "snapshot_path": str(snapshot._path()) if snapshot else path
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
    """Create a replica of a table."""
    try:
        ensure_pixeltable_available()
        if not source.startswith('pxt://'):
            source_table = pxt.get_table(source)
            result = pxt.create_replica(destination, source_table)
        else:
            result = pxt.create_replica(destination, source)
        return {
            "success": True,
            "message": f"Replica created from '{source}' to '{destination}'"
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
        result_data = df.collect()
        return {
            "success": True,
            "data": serialize_result(result_data)["data"],
            "row_count": len(result_data) if isinstance(result_data, list) else 1
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
        result = table.select(*columns) if columns else table.select()
        if limit:
            result = result.limit(limit)
        data = result.collect()
        return {
            "success": True,
            "data": serialize_result(data)["data"],
            "row_count": len(data) if isinstance(data, list) else 1,
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
        - "openai.vision('Describe this image', table.image, model='gpt-4o-mini')"
        - "image.width(table.image)"
    """
    try:
        ensure_pixeltable_available()

        # Import dependency helpers (avoids circular import at module level)
        from .dependencies import check_dependencies, pixeltable_auto_install_for_expression

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

            try:
                from pixeltable.ext.functions import yolox
                eval_context['yolox'] = yolox
            except ImportError:
                pass

            try:
                from pixeltable.functions import openai, image, string, math
                eval_context.update({
                    'openai': openai, 'image': image,
                    'string': string, 'math': math
                })
            except ImportError:
                pass

            try:
                from pixeltable.functions import huggingface
                eval_context['huggingface'] = huggingface
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
