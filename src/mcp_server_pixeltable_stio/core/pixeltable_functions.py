"""
Core Pixeltable functions for MCP server – **re-export hub**.

All implementation has been moved to specialised modules:

- ``tables.py``        – table CRUD, views, snapshots, replicas, queries, inserts, computed columns
- ``directories.py``   – directory CRUD, listing, browsing, moving
- ``dependencies.py``  – dependency checking, installation, diagnostics
- ``udf.py``           – UDF creation, array helpers, type helpers, tool wrappers, MCP connection
- ``helpers.py``       – shared utilities, configuration, version, types, docs, list_tools

This file re-exports every public symbol so that existing ``from … pixeltable_functions import X``
statements continue to work without modification during the migration period.
"""

# ---- tables.py ----
from .tables import (  # noqa: F401
    pixeltable_init,
    pixeltable_create_table,
    pixeltable_get_table,
    pixeltable_list_tables,
    pixeltable_drop_table,
    pixeltable_create_view,
    pixeltable_create_snapshot,
    pixeltable_get_table_schema,
    pixeltable_create_replica,
    pixeltable_query_table,
    pixeltable_insert_data,
    pixeltable_query,
    pixeltable_add_computed_column,
)

# ---- directories.py ----
from .directories import (  # noqa: F401
    pixeltable_create_dir,
    pixeltable_drop_dir,
    pixeltable_list_dirs,
    pixeltable_ls,
    pixeltable_move,
)

# ---- dependencies.py ----
from .dependencies import (  # noqa: F401
    check_dependencies,
    pixeltable_check_dependencies,
    pixeltable_install_dependency,
    pixeltable_auto_install_for_expression,
    pixeltable_system_diagnostics,
)

# ---- udf.py ----
from .udf import (  # noqa: F401
    pixeltable_create_udf,
    pixeltable_create_array,
    pixeltable_create_tools,
    pixeltable_connect_mcp,
    pixeltable_create_type,
)

# ---- helpers.py ----
from .helpers import (  # noqa: F401
    pxt,
    suppress_pixeltable_output,
    ensure_pixeltable_available,
    serialize_result,
    pixeltable_configure_logging,
    pixeltable_get_version,
    pixeltable_get_types,
    pixeltable_set_datastore,
    pixeltable_get_datastore,
    pixeltable_list_functions,
    pixeltable_get_help,
    pixeltable_list_tools,
    pixeltable_search_docs,
)
