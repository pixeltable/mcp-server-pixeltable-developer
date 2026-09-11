"""Safe, typed Pixeltable tools exposed by the MCP server."""

from __future__ import annotations

from contextlib import suppress
from typing import Annotated, Any, Literal, TypeVar
from urllib.parse import urlsplit

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, JsonValue, ValidationError

from .models import (
    CatalogEntry,
    CatalogResult,
    CheckResult,
    ErrorsResult,
    InsertResult,
    RecomputeResult,
    ReconcileResult,
    RowResult,
    RowsResult,
    ScaffoldResult,
    ServiceInfo,
    ServiceListResult,
    ServiceStopResult,
    TableDescription,
)
from .runtime import CommandRunner, ProcessResult, ServerConfig

READ_ONLY_LOCAL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
READ_ONLY_REMOTE = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
INSERT_ACTION = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
RECOMPUTE_ACTION = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
SCAFFOLD_ACTION = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
RECONCILE_ACTION = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=True,
)
DESTRUCTIVE_ACTION = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=True,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _model(model: type[ModelT], payload: Any, *, operation: str) -> ModelT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ToolError(f"Pixeltable returned an unexpected {operation} result") from exc


def _dict(result: ProcessResult, *, operation: str) -> dict[str, Any]:
    if not isinstance(result.data, dict):
        raise ToolError(f"Pixeltable returned an unexpected {operation} result")
    return result.data


def _list(result: ProcessResult, *, operation: str) -> list[Any]:
    if not isinstance(result.data, list):
        raise ToolError(f"Pixeltable returned an unexpected {operation} result")
    return result.data


def _project_label(config: ServerConfig, file_path: str) -> str:
    path = config.resolve_project_file(file_path)
    return path.relative_to(config.project_root).as_posix()


def _validate_columns(config: ServerConfig, columns: list[str] | None) -> list[str]:
    if columns is None:
        return []
    return [config.validate_identifier(column, label="column name") for column in columns]


def _validate_service_reference(value: str) -> str:
    value = value.strip()
    if not value or value.startswith("-") or "\x00" in value or any(part == ".." for part in value.split("/")):
        raise ToolError(f"Invalid service name: {value!r}")
    return value


def _service_info(payload: Any) -> ServiceInfo:
    if not isinstance(payload, dict):
        raise ToolError("Pixeltable returned an unexpected service listing")
    normalized = dict(payload)
    endpoint = normalized.get("endpoint")
    if normalized.get("port") is None and isinstance(endpoint, str):
        with suppress(ValueError):
            normalized["port"] = urlsplit(endpoint).port
    return _model(ServiceInfo, normalized, operation="service listing")


def register_default_tools(
    server: MCPServer[Any],
    config: ServerConfig,
    runner: CommandRunner,
) -> None:
    """Register the supported 0.2 tool contract."""

    @server.tool(
        name="pixeltable_list_catalog",
        description="List tables, views, and directories in the local Pixeltable catalog.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def list_catalog(
        path: Annotated[str, Field(description="Local catalog directory, or empty for the root.")] = "",
        include_counts: Annotated[
            bool,
            Field(description="Run count queries and include row counts. This can be slower on large tables."),
        ] = False,
    ) -> CatalogResult:
        catalog_path = config.validate_catalog_path(path, allow_empty=True)
        arguments = ["ls"]
        if catalog_path:
            arguments.append(catalog_path)
        arguments.extend(["--tree", "--json"])
        if include_counts:
            arguments.append("--counts")
        result = await runner.pxt(arguments)
        payload = _dict(result, operation="catalog listing")
        tree_payload = payload.get("tree")
        entries_payload = payload.get("entries", [])
        if isinstance(tree_payload, dict):
            entries_payload = tree_payload.get("entries", entries_payload)
        if not isinstance(entries_payload, list):
            raise ToolError("Pixeltable returned an unexpected catalog listing")
        tree = _model(CatalogEntry, tree_payload, operation="catalog tree") if tree_payload is not None else None
        entries = [_model(CatalogEntry, entry, operation="catalog entry") for entry in entries_payload]
        return CatalogResult(path=catalog_path, entries=entries, tree=tree)

    @server.tool(
        name="pixeltable_describe",
        description="Return schema, computed-column, index, and version metadata for a local table or view.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def describe(
        path: Annotated[str, Field(description="Local table or view path.")],
    ) -> TableDescription:
        table_path = config.validate_catalog_path(path)
        result = await runner.pxt(["describe", table_path, "--json"])
        return _model(TableDescription, _dict(result, operation="table description"), operation="table description")

    @server.tool(
        name="pixeltable_rows",
        description="Preview up to 100 rows without evaluating unstored computed columns unless explicitly requested.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def rows(
        path: Annotated[str, Field(description="Local table or view path.")],
        limit: Annotated[int, Field(ge=1, le=100, description="Maximum rows to return.")] = 10,
        columns: Annotated[
            list[str] | None,
            Field(max_length=100, description="Optional column subset. Explicit unstored columns may be evaluated."),
        ] = None,
    ) -> RowsResult:
        table_path = config.validate_catalog_path(path)
        selected_columns = _validate_columns(config, columns)
        arguments = ["rows", table_path, "-n", str(limit), "--json"]
        if selected_columns:
            arguments.extend(["--cols", ",".join(selected_columns)])
        result = await runner.pxt(arguments)
        raw_rows = _list(result, operation="row preview")
        if any(not isinstance(row, dict) for row in raw_rows):
            raise ToolError("Pixeltable returned an unexpected row preview")
        return RowsResult(path=table_path, rows=raw_rows, count=len(raw_rows))

    @server.tool(
        name="pixeltable_get_row",
        description="Look up one local table row using primary-key values in declared key order.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def get_row(
        path: Annotated[str, Field(description="Local table path with a primary key.")],
        primary_key: Annotated[
            list[str],
            Field(min_length=1, max_length=32, description="Primary-key values in the order shown by describe."),
        ],
        columns: Annotated[list[str] | None, Field(max_length=100, description="Optional column subset.")] = None,
    ) -> RowResult:
        table_path = config.validate_catalog_path(path)
        selected_columns = _validate_columns(config, columns)
        arguments = ["get", "--json"]
        if selected_columns:
            arguments.extend(["--cols", ",".join(selected_columns)])
        arguments.extend(["--", table_path, *primary_key])
        result = await runner.pxt(arguments)
        payload = _dict(result, operation="primary-key lookup")
        row = payload.get("row")
        if not isinstance(row, dict):
            raise ToolError("Pixeltable returned an unexpected primary-key lookup")
        return RowResult(path=table_path, row=row)

    @server.tool(
        name="pixeltable_errors",
        description="List failed computed-column values for a local table that has a primary key.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def errors(
        path: Annotated[str, Field(description="Local table path with a primary key.")],
        column: Annotated[str | None, Field(description="Optional computed column to filter by.")] = None,
    ) -> ErrorsResult:
        table_path = config.validate_catalog_path(path)
        arguments = ["errors", table_path, "--json"]
        if column is not None:
            column = config.validate_identifier(column, label="column name")
            arguments.extend(["--col", column])
        result = await runner.pxt(arguments)
        raw_errors = _list(result, operation="computed-column errors")
        if any(not isinstance(error, dict) for error in raw_errors):
            raise ToolError("Pixeltable returned an unexpected error listing")
        return ErrorsResult(path=table_path, column=column, errors=raw_errors, count=len(raw_errors))

    @server.tool(
        name="pixeltable_insert_rows",
        description="Insert 1 to 1,000 JSON rows into a local table and run its stored computed columns.",
        annotations=INSERT_ACTION,
        structured_output=True,
    )
    async def insert_rows(
        path: Annotated[str, Field(description="Local base-table path.")],
        rows: Annotated[
            list[dict[str, JsonValue]],
            Field(min_length=1, max_length=1000, description="Rows keyed by declared column name."),
        ],
        on_error: Annotated[
            Literal["abort", "ignore"],
            Field(description="Abort the batch on a computation error, or retain rows and record errors."),
        ] = "abort",
    ) -> InsertResult:
        table_path = config.validate_catalog_path(path)
        result = await runner.worker("insert", {"path": table_path, "rows": rows, "on_error": on_error})
        return _model(InsertResult, _dict(result, operation="insert"), operation="insert")

    @server.tool(
        name="pixeltable_recompute",
        description="Preview or run computed-column recovery. Defaults to failed rows only and a dry run.",
        annotations=RECOMPUTE_ACTION,
        structured_output=True,
    )
    async def recompute(
        path: Annotated[str, Field(description="Local table or view path.")],
        columns: Annotated[
            list[str],
            Field(min_length=1, max_length=100, description="Computed columns to recompute."),
        ],
        errors_only: Annotated[bool, Field(description="Only recompute rows whose selected column failed.")] = True,
        dry_run: Annotated[bool, Field(description="Return the plan without changing stored values.")] = True,
        cascade: Annotated[bool, Field(description="Also recompute dependent computed columns.")] = True,
    ) -> RecomputeResult:
        table_path = config.validate_catalog_path(path)
        selected_columns = _validate_columns(config, columns)
        if errors_only and len(selected_columns) != 1:
            raise ToolError("errors_only requires exactly one computed column")
        arguments = ["recompute", table_path, *selected_columns, "--json"]
        if errors_only:
            arguments.append("--errors-only")
        if not cascade:
            arguments.append("--no-cascade")
        arguments.append("-n" if dry_run else "-f")
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        payload = _dict(result, operation="recompute")
        pending = result.pending
        if dry_run:
            table_rows = payload.get("table_rows", 0)
            pending = isinstance(table_rows, int) and table_rows > 0
        return RecomputeResult(
            path=table_path,
            columns=selected_columns,
            errors_only=errors_only,
            dry_run=dry_run,
            pending=pending,
            details=payload,
        )

    @server.tool(
        name="pixeltable_scaffold_app",
        description="Initialize the project and write one Pixeltable 0.7 application or brief schema example.",
        annotations=SCAFFOLD_ACTION,
        structured_output=True,
    )
    async def scaffold_app(
        kind: Annotated[
            Literal["service", "schema"],
            Field(description="Generate an HTTP service application or a brief schema-only application."),
        ] = "service",
        output_path: Annotated[
            str | None,
            Field(description="New project-relative Python file. Defaults to app.py or schema.py."),
        ] = None,
    ) -> ScaffoldResult:
        chosen_output = output_path or ("app.py" if kind == "service" else "schema.py")
        output = config.resolve_scaffold_output(chosen_output)
        await runner.pxt(["init", "--json"])
        arguments = [kind, "example", "--out", str(output)]
        if kind == "schema":
            arguments.append("--brief")
        await runner.pxt(arguments, parse_json=False)
        return ScaffoldResult(
            project=config.project_root.name,
            kind=kind,
            file=output.relative_to(config.project_root).as_posix(),
            created=output.is_file(),
        )

    @server.tool(
        name="pixeltable_schema_check",
        description="Validate a TableModel schema file without reading or changing a catalog.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def schema_check(
        schema_file: Annotated[str, Field(description="Project-contained Python schema file.")],
    ) -> CheckResult:
        file_label = _project_label(config, schema_file)
        result = await runner.pxt(["schema", "check", file_label, "--json"])
        payload = _dict(result, operation="schema check")
        return CheckResult(
            kind="schema",
            file=file_label,
            valid=bool(payload.get("valid")),
            errors=[str(value) for value in payload.get("errors", [])],
            warnings=[str(value) for value in payload.get("warnings", [])],
        )

    @server.tool(
        name="pixeltable_schema_diff",
        description="Read the migration plan between a TableModel schema file and a local or hosted target.",
        annotations=READ_ONLY_REMOTE,
        structured_output=True,
    )
    async def schema_diff(
        schema_file: Annotated[str, Field(description="Project-contained Python schema file.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
    ) -> ReconcileResult:
        file_label = _project_label(config, schema_file)
        target = config.validate_target(target)
        result = await runner.pxt(
            ["schema", "diff", file_label, target, "--json"], allowed_exit_codes=frozenset({0, 2})
        )
        return ReconcileResult(
            kind="schema", operation="diff", file=file_label, target=target, pending=result.pending, details=result.data
        )

    @server.tool(
        name="pixeltable_schema_update",
        description="Reconcile a TableModel schema against a target. Destructive plans require explicit permission.",
        annotations=RECONCILE_ACTION,
        structured_output=True,
    )
    async def schema_update(
        schema_file: Annotated[str, Field(description="Project-contained Python schema file.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
        allow_destructive: Annotated[
            bool,
            Field(description="Permit column or index drops. Unsupported expression/type changes remain refused."),
        ] = False,
        dry_run: Annotated[bool, Field(description="Return the plan without changing the target.")] = False,
    ) -> ReconcileResult:
        file_label = _project_label(config, schema_file)
        target = config.validate_target(target)
        arguments = ["schema", "update", file_label, target, "--json", "-f"]
        if allow_destructive:
            arguments.append("--allow-destructive")
        if dry_run:
            arguments.append("-n")
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        return ReconcileResult(
            kind="schema",
            operation="update",
            file=file_label,
            target=target,
            pending=result.pending,
            details=result.data,
        )

    @server.tool(
        name="pixeltable_schema_prune",
        description="Preview or remove target tables that are absent from the TableModel schema.",
        annotations=DESTRUCTIVE_ACTION,
        structured_output=True,
    )
    async def schema_prune(
        schema_file: Annotated[str, Field(description="Project-contained Python schema file.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
        dry_run: Annotated[bool, Field(description="List removals without dropping tables.")] = True,
    ) -> ReconcileResult:
        file_label = _project_label(config, schema_file)
        target = config.validate_target(target)
        arguments = ["schema", "prune", file_label, target, "--json", "-n" if dry_run else "-f"]
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        return ReconcileResult(
            kind="schema",
            operation="prune",
            file=file_label,
            target=target,
            pending=result.pending,
            details=result.data,
        )

    @server.tool(
        name="pixeltable_service_check",
        description="Validate a FastAPIRouter application without changing a catalog or starting a service.",
        annotations=READ_ONLY_LOCAL,
        structured_output=True,
    )
    async def service_check(
        app_file: Annotated[str, Field(description="Project-contained Python service application.")],
    ) -> CheckResult:
        file_label = _project_label(config, app_file)
        result = await runner.pxt(["service", "check", file_label, "--json"])
        payload = _dict(result, operation="service check")
        return CheckResult(
            kind="service",
            file=file_label,
            valid=bool(payload.get("valid")),
            errors=[str(value) for value in payload.get("errors", [])],
            warnings=[str(value) for value in payload.get("warnings", [])],
        )

    @server.tool(
        name="pixeltable_service_diff",
        description="Compare declared FastAPI services with a local or hosted target without changing either.",
        annotations=READ_ONLY_REMOTE,
        structured_output=True,
    )
    async def service_diff(
        app_file: Annotated[str, Field(description="Project-contained Python service application.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
        service: Annotated[str | None, Field(description="Optional declared service name.")] = None,
        enable_otel: Annotated[bool, Field(description="Include OpenTelemetry state in the comparison.")] = False,
    ) -> ReconcileResult:
        file_label = _project_label(config, app_file)
        target = config.validate_target(target)
        arguments = ["service", "diff", file_label, target]
        if service is not None:
            arguments.append(config.validate_identifier(service, label="service name"))
        arguments.append("--json")
        if enable_otel:
            arguments.append("--otel")
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        return ReconcileResult(
            kind="service",
            operation="diff",
            file=file_label,
            target=target,
            pending=result.pending,
            details=result.data,
        )

    @server.tool(
        name="pixeltable_service_update",
        description="Start or reconcile declared services against a local or hosted target.",
        annotations=RECONCILE_ACTION,
        structured_output=True,
    )
    async def service_update(
        app_file: Annotated[str, Field(description="Project-contained Python service application.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
        service: Annotated[str | None, Field(description="Optional declared service name.")] = None,
        allow_destructive: Annotated[
            bool,
            Field(description="Permit route removals or changes that stop serving existing routes."),
        ] = False,
        dry_run: Annotated[bool, Field(description="Return the plan without starting or restarting services.")] = False,
        enable_otel: Annotated[bool, Field(description="Start services with OpenTelemetry instrumentation.")] = False,
        port: Annotated[int | None, Field(ge=1, le=65535, description="Port for one named service.")] = None,
    ) -> ReconcileResult:
        file_label = _project_label(config, app_file)
        target = config.validate_target(target)
        if port is not None and service is None:
            raise ToolError("port requires one named service")
        arguments = ["service", "update", file_label, target]
        if service is not None:
            arguments.append(config.validate_identifier(service, label="service name"))
        arguments.extend(["--json", "-f"])
        if allow_destructive:
            arguments.append("--allow-destructive")
        if dry_run:
            arguments.append("-n")
        if enable_otel:
            arguments.append("--otel")
        if port is not None:
            arguments.extend(["--port", str(port)])
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        return ReconcileResult(
            kind="service",
            operation="update",
            file=file_label,
            target=target,
            pending=result.pending,
            details=result.data,
        )

    @server.tool(
        name="pixeltable_service_list",
        description="List services known locally or for one explicit local/hosted target.",
        annotations=READ_ONLY_REMOTE,
        structured_output=True,
    )
    async def service_list(
        target: Annotated[
            str | None,
            Field(description="Optional local catalog directory or pxt://org:database/path URI."),
        ] = None,
    ) -> ServiceListResult:
        arguments = ["service", "list"]
        if target is not None:
            target = config.validate_target(target)
            arguments.append(target)
        arguments.append("--json")
        result = await runner.pxt(arguments)
        services = [_service_info(payload) for payload in _list(result, operation="service listing")]
        return ServiceListResult(target=target, services=services)

    @server.tool(
        name="pixeltable_service_stop",
        description="Stop one or more named services without removing their configuration.",
        annotations=DESTRUCTIVE_ACTION,
        structured_output=True,
    )
    async def service_stop(
        names: Annotated[
            list[str],
            Field(min_length=1, max_length=100, description="Service names or TARGET/NAME references."),
        ],
    ) -> ServiceStopResult:
        validated_names = [_validate_service_reference(name) for name in names]
        result = await runner.pxt(["service", "stop", "--json", "--", *validated_names])
        return ServiceStopResult(names=validated_names, details=result.data)

    @server.tool(
        name="pixeltable_service_prune",
        description="Preview or remove running services absent from the application file.",
        annotations=DESTRUCTIVE_ACTION,
        structured_output=True,
    )
    async def service_prune(
        app_file: Annotated[str, Field(description="Project-contained Python service application.")],
        target: Annotated[str, Field(description="Local catalog directory or pxt://org:database/path URI.")],
        dry_run: Annotated[bool, Field(description="List removals without stopping services.")] = True,
    ) -> ReconcileResult:
        file_label = _project_label(config, app_file)
        target = config.validate_target(target)
        arguments = ["service", "prune", file_label, target, "--json", "-n" if dry_run else "-f"]
        result = await runner.pxt(arguments, allowed_exit_codes=frozenset({0, 2}))
        return ReconcileResult(
            kind="service",
            operation="prune",
            file=file_label,
            target=target,
            pending=result.pending,
            details=result.data,
        )
