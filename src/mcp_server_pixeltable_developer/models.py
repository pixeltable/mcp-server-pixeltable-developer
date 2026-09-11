"""Typed MCP inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictModel(BaseModel):
    """Base model for stable public result contracts."""

    model_config = ConfigDict(extra="forbid")


class CatalogEntry(BaseModel):
    """One object in the Pixeltable catalog tree."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    path: str
    kind: str | None = None
    rows: int | None = None
    columns: int | None = None
    version: int | None = None
    error_count: int | None = None
    base: str | None = None
    entries: list[CatalogEntry] = Field(default_factory=list)


class CatalogResult(StrictModel):
    """Catalog entries below a requested path."""

    path: str
    entries: list[CatalogEntry]
    tree: CatalogEntry | None = None


class ColumnDescription(BaseModel):
    """Relevant metadata for a table column."""

    model_config = ConfigDict(extra="allow")

    name: str
    type_: str
    is_stored: bool | None = None
    is_primary_key: bool | None = None
    is_computed: bool | None = None
    computed_with: str | None = None
    depends_on: list[list[str]] | None = None
    comment: str | None = None


class TableDescription(BaseModel):
    """Schema and metadata reported by ``pxt describe``."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    path: str
    kind: str
    columns: dict[str, ColumnDescription]
    indexes: dict[str, Any] = Field(default_factory=dict)
    has_default_idxs: bool = False
    is_view: bool = False
    is_snapshot: bool = False
    version: int
    primary_key: list[str] | str | None = None


class RowsResult(StrictModel):
    """A bounded table preview."""

    path: str
    rows: list[dict[str, JsonValue]]
    count: int


class RowResult(StrictModel):
    """One row selected by primary key."""

    path: str
    row: dict[str, JsonValue]


class ErrorsResult(StrictModel):
    """Failed computed-column values for a table."""

    path: str
    column: str | None = None
    errors: list[dict[str, JsonValue]]
    count: int


class InsertResult(StrictModel):
    """Pixeltable insertion statistics."""

    path: str
    num_rows: int
    num_computed_values: int
    num_exceptions: int
    columns_with_exceptions: list[str]
    updated_columns: list[str]


class CheckResult(StrictModel):
    """Static validation result for a schema or service file."""

    kind: Literal["schema", "service"]
    file: str
    valid: bool
    errors: list[str]
    warnings: list[str]


class ReconcileResult(StrictModel):
    """A schema or service reconciliation result."""

    kind: Literal["schema", "service"]
    operation: Literal["diff", "update", "prune"]
    file: str
    target: str
    pending: bool
    details: JsonValue


class ServiceInfo(BaseModel):
    """Stable service fields with forward-compatible CLI metadata."""

    model_config = ConfigDict(extra="allow")

    name: str
    catalog_path: str | None = None
    endpoint: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    state: str | None = None
    error: str | None = None
    app_module: str | None = None
    spec: JsonValue = None
    pid: int | None = None
    process_started_at: float | None = None


class ServiceListResult(StrictModel):
    """Services known to the Pixeltable daemon."""

    target: str | None = None
    services: list[ServiceInfo]


class ServiceStopResult(StrictModel):
    """Services stopped by a request."""

    names: list[str]
    details: JsonValue


class ScaffoldResult(StrictModel):
    """A newly generated application or schema file."""

    project: str
    kind: Literal["service", "schema"]
    file: str
    created: bool


class RecomputeResult(StrictModel):
    """A computed-column recovery plan or execution result."""

    path: str
    columns: list[str]
    errors_only: bool
    dry_run: bool
    pending: bool
    details: JsonValue


class StatusResource(StrictModel):
    """Redacted server and runtime status."""

    server_name: str
    server_version: str
    mcp_version: str
    pixeltable_version: str
    project: str
    transport: str
    unsafe_tools_enabled: bool


JsonObject = dict[str, JsonValue]
