"""Worker functions that run in isolated processes with Pixeltable access."""

import json
from typing import Dict, Any, List


def get_tables() -> List[str]:
    """Get list of all Pixeltable tables.

    This runs in an isolated worker process.
    """
    import pixeltable as pxt
    tables = pxt.list_tables()
    return tables


def get_table_info(table_path: str) -> Dict[str, Any]:
    """Get detailed information about a table.

    This runs in an isolated worker process.

    Args:
        table_path: Path to the table

    Returns:
        Dictionary with table information
    """
    import pixeltable as pxt

    try:
        table = pxt.get_table(table_path)
        schema = {}

        for col_name, col_type in table._schema.items():
            schema[col_name] = {
                'type': str(col_type),
                'computed': hasattr(table, '_computed_columns') and col_name in table._computed_columns
            }

        return {
            'name': table_path,
            'schema': schema,
            'row_count': table.count()
        }
    except Exception as e:
        return {
            'error': str(e),
            'name': table_path
        }
