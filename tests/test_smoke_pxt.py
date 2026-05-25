"""End-to-end smoke test against a real Pixeltable catalog (--run-slow only)."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore:Field name \"schema\".*shadows.*:UserWarning"),
]


def test_create_insert_query_drop(isolated_pixeltable_home):
    """Full lifecycle: create dir + table -> insert -> query -> drop.

    Uses the isolated_pixeltable_home fixture so the test never touches user data.
    """
    from mcp_server_pixeltable_stio.core.directories import pixeltable_create_dir
    from mcp_server_pixeltable_stio.core.tables import (
        pixeltable_create_table,
        pixeltable_drop_table,
        pixeltable_insert_data,
        pixeltable_query_table,
    )

    dir_r = pixeltable_create_dir("smoke", if_exists="ignore")
    assert dir_r["success"], dir_r

    create_r = pixeltable_create_table(
        "smoke.items",
        schema={"id": "Required[String]", "value": "Int"},
        if_exists="replace",
    )
    assert create_r["success"], create_r

    insert_r = pixeltable_insert_data(
        "smoke.items",
        [{"id": "a", "value": 1}, {"id": "b", "value": 2}],
    )
    assert insert_r["success"], insert_r
    assert insert_r["rows_inserted"] == 2

    query_r = pixeltable_query_table("smoke.items", limit=5)
    assert query_r["success"], query_r
    assert query_r["row_count"] == 2
    assert set(query_r["columns"]) == {"id", "value"}

    drop_r = pixeltable_drop_table("smoke.items")
    assert drop_r["success"], drop_r


def test_create_view_with_iterator(isolated_pixeltable_home):
    """Validate that the iterator wiring on pixeltable_create_view works end-to-end."""
    from mcp_server_pixeltable_stio.core.directories import pixeltable_create_dir
    from mcp_server_pixeltable_stio.core.tables import (
        pixeltable_create_table,
        pixeltable_create_view,
    )

    pixeltable_create_dir("smoke", if_exists="ignore")
    pixeltable_create_table(
        "smoke.docs",
        schema={"document": "Document", "title": "String"},
        if_exists="replace",
    )

    # Use paragraph separators -- token_limit requires tiktoken which may be absent in CI.
    view_r = pixeltable_create_view(
        path="smoke.chunks",
        base_table_path="smoke.docs",
        iterator="document_splitter",
        iterator_kwargs={"document": "table.document", "separators": "paragraph"},
        if_exists="replace",
    )
    assert view_r["success"], view_r


def test_unknown_type_fails_loudly(isolated_pixeltable_home):
    from mcp_server_pixeltable_stio.core.directories import pixeltable_create_dir
    from mcp_server_pixeltable_stio.core.tables import pixeltable_create_table

    pixeltable_create_dir("smoke", if_exists="ignore")
    r = pixeltable_create_table(
        "smoke.bad",
        schema={"mystery": "NotAPxtType"},
        if_exists="replace",
    )
    assert r["success"] is False
    assert "NotAPxtType" in r["error"]
