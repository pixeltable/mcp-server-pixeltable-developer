# Changelog

## 0.2.1 (2026-10-10)

First release on PyPI. Install with `uvx mcp-server-pixeltable-developer`.

**Dependencies**
- Locked Pixeltable 0.7.16 and MCP Python SDK 2.3.0. The minimum Pixeltable is now 0.7.10.

**Guidance**
- Changing a computed expression: Pixeltable 0.7.10 alters the column in place and
  keeps the stored values. The app guidance, the debug prompt, and the docs now say to
  run `pxt recompute TABLE COLUMN -f` afterwards. A type change is still unsupported.
- Cloud: the guidance told agents to put credentials under `secrets` in the project
  config, which makes the database entry fail to load. Keys now go through
  `pxt secret set`, and sign-in through `pxt login`.
- The missing-primary-key error suggests the uuid7 key that `pxt service example` generates.

**Distribution**
- Published to PyPI through trusted publishing, and to the official MCP Registry as
  `io.github.pixeltable/mcp-server-pixeltable-developer` through GitHub OIDC. A `v*`
  tag publishes to both.
- A Claude plugin in `plugin/` runs the server for the open project.
- The Docker example keeps the catalog in a named volume. A catalog bind-mounted from
  a macOS host fails Postgres `initdb`.
- `smithery.yaml` installs the PyPI package. The PyPI author is Pixeltable.

**Reliability**
- The README documents `PXT_PORT`: a `pxt` command from another project restarts the
  shared daemon and fails the server's in-flight calls.
- Tests run each catalog on a private daemon port and stop its daemon and Postgres on
  teardown.

**Release process**
- Tags no longer wait on the 48-trial agent evaluation, which is pinned to Pixeltable
  0.7.6 and errored in its harness. The evaluation-artifact check still runs.

## 0.2.0

Rebuilt on the MCP Python SDK 2 and Pixeltable 0.7: an application-first tool set
(scaffold, schema and service lifecycle, inspection, recovery), typed results, and
opt-in unsafe tools. See the [migration guide](docs/migration-0.1-to-0.2.md).
