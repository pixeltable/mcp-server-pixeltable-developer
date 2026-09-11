# Pixeltable Developer MCP 0.2 implementation review

**Review date:** 2026-09-10
**Repository:** `pixeltable/mcp-server-pixeltable-developer`
**Upgrade base:** [`5ae8063e07ea6ad1aa625039e20475986d7a7750`](https://github.com/pixeltable/mcp-server-pixeltable-developer/tree/5ae8063e07ea6ad1aa625039e20475986d7a7750)
**Candidate package:** 0.2.0 on branch `codex/mcp-0.2-upgrade`
**Comparison targets:** Pixeltable 0.7.6, pixeltable-skill 2.8.3, MCP Python SDK 2.2.0, MCP specification 2026-07-28

## Verdict

The 0.2 implementation fixes the two release-blocking compatibility failures in
0.1 and replaces its broad, unsafe procedural API with the requested app-first
MCP 2 surface. In the reviewed working tree, the production server exposes
exactly 18 typed tools, four resources, and four prompts; the three unsafe tools
are absent by default and restricted to an exact environment opt-in over
`stdio`. Local integration tests exercise real Pixeltable 0.7.6 schema and
service lifecycles, deterministic document/image/video/audio fixtures, mocked
tool calling, additive and destructive migrations, rejection of unsupported
computed-expression changes, and failed-computation recovery.

The MCP protocol result is also strong. The repository's test-only Streamable
HTTP endpoint passed all 30 applicable stateless server checks in the official
conformance runner. The frozen 2026-07-28 requirement run passed all 37 scored
server scenarios. Tasks are intentionally unimplemented.

**Release verdict: hold the 0.2.0 tag until the 48 agent trials are complete.**
The runtime implementation is a viable release candidate, but agent parity with
pixeltable-skill 2.8.3 and the two website documents has not been demonstrated.
Two repetitions per condition are a regression screen, not evidence of
statistical superiority. Provider mocks prove wiring only, and Cloud guidance
was reviewed but not live-tested.

No edit to the pixeltable-skill repository is required or included in this
upgrade.

## Reproducibility record

| Item | Recorded value |
|---|---|
| Upgrade base | `5ae8063e07ea6ad1aa625039e20475986d7a7750`, 0.1.0 |
| Candidate | 0.2.0 working tree on `codex/mcp-0.2-upgrade` |
| Pixeltable | 0.7.6, Python `>=3.11`, uploaded 2026-09-09 17:02:40 UTC |
| Pixeltable wheel SHA-256 | `9e6cdbe54f042b31786bede4a4cd4d68b361b31d75f34ea4e9c415146c901114` |
| Pixeltable source | [`pixeltable/v0.7.6`](https://github.com/pixeltable/pixeltable/tree/v0.7.6) |
| Pixeltable skill | 2.8.3 at [`f550e6ed757b48635e4f53900840f4e9a1fb4c93`](https://github.com/pixeltable/pixeltable-skill/tree/f550e6ed757b48635e4f53900840f4e9a1fb4c93) |
| MCP Python SDK | 2.2.0, Python `>=3.10`, uploaded 2026-09-07 16:06 UTC |
| MCP wheel SHA-256 | `bde982589473a060ae145e3406e9a5333fe538c97229ba841f5a7f92be004f81` |
| MCP source archive SHA-256 | `2dc37ecb1974becdcebdbf7561e7c15a07dbbf20ba21ba16c3593b3038b3afbd` |
| MCP SDK tag commit | [`9972c21aa42054fb1450c5fc614761ed11847ec6`](https://github.com/modelcontextprotocol/python-sdk/tree/9972c21aa42054fb1450c5fc614761ed11847ec6) |
| MCP specification | [2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) |
| Conformance runner | `@modelcontextprotocol/conformance@0.2.0-alpha.11` |
| Conformance repository snapshot | [`a983ba93c91e0bb31d0b6849eeb52f0ad1083107`](https://github.com/modelcontextprotocol/conformance/tree/a983ba93c91e0bb31d0b6849eeb52f0ad1083107) |
| `get-started.md` snapshot | 7,832 bytes; SHA-256 `7d03873526a31940552adbd4ea3b1be79b308d058a6d4b64d437ac97d14e4578` |
| `llms.txt` snapshot | 27,381 bytes; SHA-256 `7b326e37ac26468d1553a86fafe03680a61ddd0216e44c46f1931e168e3b72d8` |
| Evidence retrieval/recheck date | 2026-09-10 |

Version and artifact metadata were rechecked against the official
[Pixeltable 0.7.6 release](https://pypi.org/project/pixeltable/0.7.6/),
[MCP 2.2.0 release](https://pypi.org/project/mcp/2.2.0/),
[MCP SDK release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.2.0),
and [MCP conformance repository](https://github.com/modelcontextprotocol/conformance).
The exact wheel hashes are also locked at
[`uv.lock:1231-1256`](../uv.lock#L1231-L1256) and
[`uv.lock:1691-1732`](../uv.lock#L1691-L1732).

The review inspected all candidate source modules, package metadata, lock file,
README, migration guide, tests, CI, canvas, and conformance harness. It did not
make paid provider calls, create Cloud resources, deploy to Pixeltable Cloud,
publish a package, tag a release, or modify pixeltable-skill.

## What changed from 0.1

The baseline review in [`review-0.1.0.md`](review-0.1.0.md) reproduced three
material failures:

1. The 0.1 lock passed 38 tests with Pixeltable 0.6.3 and MCP 1.11, but the open
   dependency range allowed incompatible current packages.
2. Pixeltable 0.7.6 with MCP held below 2 produced 36 passing tests and two real
   catalog failures because `create_default_idxs` was removed in favor of
   `has_default_idxs`.
3. Pixeltable 0.7.6 with MCP 2.2 failed during import at
   `mcp.server.fastmcp`.

The candidate dispositions are:

| 0.1 defect | 0.2 disposition | Evidence |
|---|---|---|
| Fresh resolution can select an incompatible MCP major | Constrain to `mcp>=2.2,<3`; lock 2.2.0 | [`pyproject.toml:26-31`](../pyproject.toml#L26-L31), [`uv.lock:1231-1256`](../uv.lock#L1231-L1256) |
| Pixeltable 0.7.6 requires Python 3.11+ | Require Python 3.11 and advertise 3.11-3.14 | [`pyproject.toml:10-24`](../pyproject.toml#L10-L24) |
| MCP 1 `FastMCP` import fails under MCP 2 | Construct `MCPServer` with version, lifespan, and cache metadata | [`server.py:38-76`](../src/mcp_server_pixeltable_developer/server.py#L38-L76) |
| Procedural create wrappers call removed arguments | Replace them with application scaffolding and `pxt schema`/`pxt service` reconciliation | [`tools.py:313-436`](../src/mcp_server_pixeltable_developer/tools.py#L313-L436), [`tools.py:438-591`](../src/mcp_server_pixeltable_developer/tools.py#L438-L591) |
| Replica/publish targets removed Pixeltable behavior | Remove it from the public surface; use current Cloud database/schema/service ordering in guidance | [`resources.py:30-57`](../src/mcp_server_pixeltable_developer/resources.py#L30-L57) |
| Failures are returned in successful `{"success": false}` envelopes | Return domain models on success and raise `ToolError` for expected failures | [`models.py:10-192`](../src/mcp_server_pixeltable_developer/models.py#L10-L192), [`tools.py:79-95`](../src/mcp_server_pixeltable_developer/tools.py#L79-L95) |
| Package/module name is misspelled | Use `mcp_server_pixeltable_developer`; ship a one-release stderr-only compatibility shim | [`pyproject.toml:51-58`](../pyproject.toml#L51-L58), [`mcp_server_pixeltable_stio/__init__.py`](../src/mcp_server_pixeltable_stio/__init__.py) |
| Startup falls back to an empty server | Build the real server at import and allow configuration/startup failures to exit nonzero | [`server.py:112-119`](../src/mcp_server_pixeltable_developer/server.py#L112-L119), [`test_stdio.py:94-114`](../tests/test_stdio.py#L94-L114) |
| Runtime catalog switching resets Pixeltable private state | Fix project root and `PIXELTABLE_HOME` for the process lifetime | [`runtime.py:51-103`](../src/mcp_server_pixeltable_developer/runtime.py#L51-L103) |
| Arbitrary execution and file display are normal tools | Remove them from the default contract; isolate three explicit unsafe tools | [`server.py:77-85`](../src/mcp_server_pixeltable_developer/server.py#L77-L85), [`unsafe.py:285-320`](../src/mcp_server_pixeltable_developer/unsafe.py#L285-L320) |

## Public contract assessment

### Packaging and process model

The candidate has the intended package identity and bounds: version 0.2.0,
Python 3.11+, Pixeltable serve support within 0.7, and MCP within 2.x.
`mcp[cli]` appears only in the test extra; `requests`, `toml`, and `uvloop` are
absent from runtime dependencies. The console command remains
`mcp-server-pixeltable-developer`, while the canonical import package is
`mcp_server_pixeltable_developer`.

`ServerConfig` resolves and fixes the project root, Pixeltable catalog,
transport, executable, timeout, and output limit at startup. Project files are
resolved before use and must remain under the configured root; symlink escapes
are rejected. Hosted targets must use the explicit `pxt://` syntax
([`runtime.py:51-160`](../src/mcp_server_pixeltable_developer/runtime.py#L51-L160)).

Every `pxt` call goes through an asynchronous argument-array runner. It captures
and redacts both streams, applies byte and time limits, starts a separate POSIX
process group, kills the group on cancellation or failure, and maps disallowed
exit codes to `ToolError` without invoking a shell
([`runtime.py:175-324`](../src/mcp_server_pixeltable_developer/runtime.py#L175-L324)).
Schema/service diff-style operations explicitly accept exit code 2 as a valid
pending result; other exit codes are errors.

Production remains `stdio`. The Streamable HTTP app is a test-only factory that
forces unsafe mode off and registers conformance diagnostic fixtures only in
that factory ([`server.py:89-109`](../src/mcp_server_pixeltable_developer/server.py#L89-L109)).

### Default surface

Discovery returns exactly this surface:

| Area | Tools |
|---|---|
| Catalog reads | `pixeltable_list_catalog`, `pixeltable_describe`, `pixeltable_rows`, `pixeltable_get_row`, `pixeltable_errors` |
| Data actions | `pixeltable_insert_rows`, `pixeltable_recompute` |
| App creation | `pixeltable_scaffold_app` |
| Schema lifecycle | `pixeltable_schema_check`, `pixeltable_schema_diff`, `pixeltable_schema_update`, `pixeltable_schema_prune` |
| Service lifecycle | `pixeltable_service_check`, `pixeltable_service_diff`, `pixeltable_service_update`, `pixeltable_service_list`, `pixeltable_service_stop`, `pixeltable_service_prune` |

All tools use `structured_output=True` and Pydantic result models. The result
contracts contain domain fields instead of a synthetic success flag. Input
schemas enforce the 100-row read cap, 1,000-row insert cap, two allowed insert
error modes, dry-run defaults, and the single-column constraint for
errors-only recomputation
([`tools.py:127-341`](../src/mcp_server_pixeltable_developer/tools.py#L127-L341),
[`test_contract.py:89-102`](../tests/test_contract.py#L89-L102)).

Annotations match the expected effect model: local catalog reads/checks are
read-only and closed-world; remote-capable diff/list operations are read-only
and open-world; insert/recompute are non-idempotent and open-world; schema and
service mutations are destructive, idempotent, and open-world; scaffold is a
local non-destructive, non-idempotent action
([`tools.py:32-73`](../src/mcp_server_pixeltable_developer/tools.py#L32-L73),
[`test_contract.py:105-138`](../tests/test_contract.py#L105-L138)).

The four resources have stable MIME types:

| Resource | MIME type | Purpose |
|---|---|---|
| `pixeltable://status` | `application/json` | Redacted server, MCP, Pixeltable, project, transport, and unsafe-mode status |
| `pixeltable://catalog` | `application/json` | Current local catalog inventory |
| `pixeltable://guidance/app` | `text/markdown` | App-first Pixeltable 0.7.6 workflow |
| `pixeltable://guidance/cloud` | `text/markdown` | Cloud preparation order, explicitly labeled not live-tested |

The four prompts are `pixeltable_build_app`, `pixeltable_build_rag`,
`pixeltable_build_agent`, and `pixeltable_debug_computation`. Resources and
prompts are registered through MCP 2 public APIs; no private manager inspection
remains ([`resources.py:60-127`](../src/mcp_server_pixeltable_developer/resources.py#L60-L127),
[`prompts.py:11-96`](../src/mcp_server_pixeltable_developer/prompts.py#L11-L96)).

### Pixeltable workflow accuracy

The embedded application guidance follows pixeltable-skill 2.8.3 and released
Pixeltable 0.7.6:

- begin with `pxt init`, then generate `pxt service example --out app.py` or a
  brief schema example;
- declare tables through `TableModel`, computed columns through assignments,
  and HTTP routes on `FastAPIRouter` imported from `pixeltable.serving`;
- check, diff, and update schema before service reconciliation;
- use non-nullable `T` and optional `T | None`, not `pxt.Required[T]`;
- use `has_default_idxs=False`, not `create_default_idxs`;
- treat an in-place computed-expression change as unsupported even with
  `--allow-destructive`; rename the column or drop and re-add it;
- recover failed stored values with `pxt recompute`, normally previewing
  `--errors-only` before applying.

These claims are enforced in the shipped-guidance scan and contract tests
([`test_documentation.py:16-83`](../tests/test_documentation.py#L16-L83),
[`test_contract.py:141-191`](../tests/test_contract.py#L141-L191)). Advanced
expressions stay in source control instead of crossing the MCP boundary as
arbitrary strings.

| Guidance claim | Classification | Released/source evidence | Executable evidence |
|---|---|---|---|
| Use `TableModel` declarations and computed assignments | **Supported** | Pixeltable 0.7.6 application examples and skill 2.8.3 | Schema/service and recovery applications compile and reconcile in [`test_pixeltable_integration.py`](../tests/test_pixeltable_integration.py) |
| Import `FastAPIRouter` from `pixeltable.serving` | **Supported** | Pixeltable 0.7.6 application workflow; skill 2.8.3 | A generated app is started and receives a real HTTP POST in [`test_pixeltable_integration.py:66-171`](../tests/test_pixeltable_integration.py#L66-L171) |
| Prefer `pxt init` plus `pxt service example`/`pxt schema example` | **Supported** | Released 0.7.6 CLI and current `get-started.md` | `pixeltable_scaffold_app` runs the released commands and refuses overwrite in [`test_pixeltable_integration.py:66-77`](../tests/test_pixeltable_integration.py#L66-L77) |
| The index option is `has_default_idxs`, default false | **Supported** | [`globals.py:64-80`](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/globals.py#L64-L80) | `describe` returns false in [`test_pixeltable_integration.py:94-98`](../tests/test_pixeltable_integration.py#L94-L98) |
| Stored types are non-nullable; use `T | None` for optional values | **Supported** | [`type_system.py:1858-1863`](https://github.com/pixeltable/pixeltable/blob/v0.7.6/pixeltable/type_system.py#L1858-L1863); skill 2.8.3 | Generated app accepts an optional body and schema evolution adds an optional column |
| Apply schema and services through check -> diff -> update | **Supported** | Released 0.7.6 CLI and current website/skill workflow | Complete real lifecycle in [`test_pixeltable_integration.py:79-168`](../tests/test_pixeltable_integration.py#L79-L168) |
| Computed-expression replacement is unsupported in place | **Supported** | Skill 2.8.3 and 0.7.6 migration behavior | Diff classifies it unsupported and update refuses it even with destructive permission in [`test_pixeltable_integration.py:310-327`](../tests/test_pixeltable_integration.py#L310-L327) |
| Recover stored failures with errors-only recompute | **Supported** | Released `pxt errors`/`pxt recompute` CLI | Failure, preview, apply, and clearance are asserted in [`test_pixeltable_integration.py:214-270`](../tests/test_pixeltable_integration.py#L214-L270) |
| Use current modality iterators and preserve multimodal data in Pixeltable | **Supported for local deterministic fixtures** | Skill 2.8.3 and Pixeltable 0.7.6 functions packages | Document, video, and audio iterators plus image rotation execute in [`test_pixeltable_integration.py:330-541`](../tests/test_pixeltable_integration.py#L330-L541) |
| Declare an embedding index and use a named modality argument for similarity | **Supported for local deterministic retrieval** | Skill 2.8.3 and Pixeltable 0.7.6 | A local 3D embedding UDF/index and `similarity(string=...)` query rank the expected chunk first in [`test_pixeltable_integration.py:360-402`](../tests/test_pixeltable_integration.py#L360-L402) and [`test_pixeltable_integration.py:498-522`](../tests/test_pixeltable_integration.py#L498-L522) |
| Use `pxt.tools()` and `invoke_tools()` for tool dispatch | **Supported for wiring** | Skill 2.8.3 and Pixeltable 0.7.6 provider functions | Deterministic mocked call in [`test_pixeltable_integration.py:338-401`](../tests/test_pixeltable_integration.py#L338-L401) and [`test_pixeltable_integration.py:498-505`](../tests/test_pixeltable_integration.py#L498-L505) |
| Update Cloud database, then schema, then service | **Source-reviewed; unverified live** | Current skill and Cloud CLI guidance | No hosted deployment was authorized or run |

### Unsafe mode

Only the exact value `PIXELTABLE_MCP_ENABLE_UNSAFE=1` enables unsafe mode, and
configuration rejects it on Streamable HTTP
([`runtime.py:63-91`](../src/mcp_server_pixeltable_developer/runtime.py#L63-L91)).
The opt-in surface contains only:

- `pixeltable_unsafe_execute_python`: one bounded, non-persistent subprocess in
  the configured project;
- `pixeltable_unsafe_install_package`: one validated PEP 508 index requirement,
  passed as an argument array with a fixed timeout;
- `pixeltable_unsafe_display`: a loopback canvas with a random bearer token,
  same-origin enforcement, and media restricted to configured roots.

Tests cover default absence, exact opt-in, HTTP rejection, timeouts, output
limits, secret redaction, PEP 508 validation, argument arrays, bearer auth,
origin policy, traversal, symlink escape, media containment, and removal of the
raw HTML renderer
([`test_runtime.py:24-102`](../tests/test_runtime.py#L24-L102),
[`test_unsafe.py:29-198`](../tests/test_unsafe.py#L29-L198)). The capability is
still intentionally high-risk: enabled Python or package installation can make
network calls and mutate the environment. Its open-world/destructive
annotations and `stdio` restriction accurately advertise that fact.

## Executable results

### Python and MCP tests

The slow-inclusive local suite completed with **57 passed** on the reviewed
Python environment:

```bash
PIXELTABLE_DISABLE_STDOUT=1 uv run pytest -q --run-slow
```

The suite uses `Client(server, raise_exceptions=True)` for discovery and schema
contracts, and explicit `raise_exceptions=False` calls to verify protocol
`is_error=true` behavior. It checks all tool input/output schemas, annotations,
cache metadata, resource MIME types, prompt rendering, expected tool errors,
unknown-resource errors, and sanitized unexpected exceptions
([`test_contract.py`](../tests/test_contract.py)).

A real subprocess test confirms that concurrent discovery/prompt calls work,
stdout contains only JSON-RPC frames, closing stdin produces prompt EOF shutdown,
and an invalid startup root exits nonzero without protocol output
([`test_stdio.py:26-114`](../tests/test_stdio.py#L26-L114)). Cancellation of a
running child command is tested separately
([`test_runtime.py:164-199`](../tests/test_runtime.py#L164-L199)).

The integration suite allocates a fresh project and `PIXELTABLE_HOME`, then runs
real Pixeltable 0.7.6 commands. Provider behavior is mocked; no paid or hosted
call occurs.

### MCP conformance

The official conformance runner was pinned because the stable npm release
`0.1.16` did not yet recognize specification 2026-07-28. The candidate uses
`@modelcontextprotocol/conformance@0.2.0-alpha.11` and pins the MCP SDK fixture to
tag `v2.2.0` at commit
`9972c21aa42054fb1450c5fc614761ed11847ec6`
([`run-conformance.sh:4-10`](../scripts/run-conformance.sh#L4-L10)).

Observed results:

| Run | Applicable result | Other output | Exit |
|---|---:|---:|---:|
| Candidate test-only app, `server-stateless`, spec 2026-07-28 | **30/30 passed** | 0 failures, 0 warnings | 0 |
| Official MCP 2.2 everything-server fixture, frozen requirements 2026-07-28 | **37/37 scored server scenarios passed** | 167 passes and 26 failures overall; all 26 belong to unscored Tasks-extension or pending cases | 0 |

The second run validates the frozen requirement set independently of the
domain-specific tool surface. It does not turn the unscored Tasks extension
into a product requirement. The candidate deliberately omits Tasks, as planned.
The executable harness starts both servers on ephemeral loopback ports, checks
the SDK fixture commit, and stores results separately
([`run-conformance.sh:29-102`](../scripts/run-conformance.sh#L29-L102)).

The 2026-07-28 result applies to the test-only, stateless per-request HTTP
revision. MCP 2.2 production `stdio` initialization negotiates the SDK's latest
supported handshake, 2025-11-25; the raw wire test sends that version
([`test_stdio.py:62-90`](../tests/test_stdio.py#L62-L90)). The report does not
claim that production `stdio` negotiates 2026-07-28.

## Scenario-level Pixeltable evidence

| Required scenario | Result | Executable evidence | Boundary |
|---|---|---|---|
| Initial HTTP app | **Pass, live local** | Scaffold service app, check/diff/update schema and service, POST to the actual route, update again on the same port, stop and prune ([`test_pixeltable_integration.py:66-171`](../tests/test_pixeltable_integration.py#L66-L171)) | Local loopback only |
| Document retrieval | **Pass, live local** | HTML is split into two deterministic chunks; a declared local embedding index and named-argument similarity query are served through `FastAPIRouter`; querying `alpha` ranks the alpha chunk above beta ([`test_pixeltable_integration.py:360-402`](../tests/test_pixeltable_integration.py#L360-L402), [`test_pixeltable_integration.py:498-522`](../tests/test_pixeltable_integration.py#L498-L522)) | Deterministic local embedding; no provider call |
| Image/video processing | **Pass, live local** | Real generated PNG rotation and MP4 frame iteration ([`test_pixeltable_integration.py:431-530`](../tests/test_pixeltable_integration.py#L431-L530)) | Deterministic local media only |
| Audio processing | **Pass, live local** | Generated WAV splits into four bounded segments with the expected end time ([`test_pixeltable_integration.py:437-443`](../tests/test_pixeltable_integration.py#L437-L443), [`test_pixeltable_integration.py:531-541`](../tests/test_pixeltable_integration.py#L531-L541)) | No transcription provider call |
| Tool calling | **Pass for wiring** | `pxt.tools()` and `invoke_tools()` execute a deterministic mocked model call and return 5 ([`test_pixeltable_integration.py:330-428`](../tests/test_pixeltable_integration.py#L330-L428), [`test_pixeltable_integration.py:543-550`](../tests/test_pixeltable_integration.py#L543-L550)) | Mock proves wiring only |
| Schema evolution | **Pass, live local** | Additive update succeeds; destructive update requires opt-in; computed-expression replacement remains rejected even with opt-in ([`test_pixeltable_integration.py:272-327`](../tests/test_pixeltable_integration.py#L272-L327)) | Local catalog only |
| Failed-computation recovery | **Pass, live local** | Two recorded failures are listed, dry-run reports pending work, errors-only recompute clears them after fixture repair ([`test_pixeltable_integration.py:174-270`](../tests/test_pixeltable_integration.py#L174-L270)) | Deterministic local UDF |
| Cloud deployment preparation | **Source-reviewed only** | Guidance preserves database -> schema -> service ordering and explicit `pxt://` targets ([`resources.py:30-57`](../src/mcp_server_pixeltable_developer/resources.py#L30-L57)) | Not live-tested; no Cloud resource created |

The table supports local runtime correctness for the behavior actually
executed. It does not substitute for the agent trial matrix.

## Agent evaluation status

The acceptance design requires eight scenarios under three guidance conditions,
each repeated twice with identical model, settings, prompts, tool access, and
limits:

| Guidance condition | Planned trials | Completed and passed | Execution errors | Comparative conclusion |
|---|---:|---:|---:|---|
| Upgraded MCP plus pixeltable-skill 2.8.3 | 16 | 2 | 14 | Incomplete |
| pixeltable-skill 2.8.3 alone | 16 | 2 | 14 | Incomplete |
| `get-started.md` plus `llms.txt` | 16 | 1 | 15 | Incomplete |
| **Total** | **48** | **5** | **43** | **Parity not demonstrated** |

Five initial-HTTP-app trials completed and passed their independent executable
verifier. The second website trial was interrupted after generating an invalid
`TableModel` declaration, and the remaining sessions could not begin because
the runner reported `Your workspace is out of credits. Add credits to
continue.` The matrix retains all 43 interruptions as `error`; it does not score
them as task failures or omit them from the record. The five completed trials
cover only one scenario, so they establish harness operation rather than parity.

The matrix is materialized as 48 explicit result rows in
[`trials.csv`](../evals/trials.csv), with retained evidence for every attempt.
The exact eight scenario definitions and
prompt paths are in [`scenarios.json`](../evals/scenarios.json), and
[`sources.lock.json`](../evals/sources.lock.json) records every skill/reference
and website snapshot with its byte count, checksum, source URL, commit or
retrieval date. The eight user-visible prompts are checked in under
[`evals/prompts`](../evals/prompts). Deterministic document, PNG, MP4, and WAV
fixtures have their own generation script and manifest under
[`evals/fixtures`](../evals/fixtures).

The structural validator currently passes, while its complete-release mode
fails because 43 rows have execution errors. This is an intentional, executable
hold:

```bash
python3 evals/validate_results.py
# OK: structural validation passed for evals/trials.csv

python3 evals/validate_results.py --require-complete
# ERROR: release gate requires all 48 trials; incomplete: ...
```

Every completed trial must retain the prompt, source manifest, model/settings,
fresh workspace and catalog paths, transcript, verifier output, elapsed time,
and these scores: completion, executable correctness, first-attempt success,
unsupported APIs, unnecessary dependencies, recovery, and tool calls. Provider
mocks and Cloud instruction-only trials must carry those labels in the result.

The release criterion is exact: the upgraded MCP condition must match or exceed
both baselines overall, have no repeated scenario-specific regression, and
leave no critical API or workflow defect unresolved. Mixed results must be
reported as mixed. Two repetitions cannot establish statistical superiority.

## Prioritized findings

### P0 — No unresolved critical runtime/API defect was found

The 0.1 import failure, removed Pixeltable keyword, removed replica workflow,
fail-open startup, unsafe default execution, and untyped success envelopes are
closed in the candidate. The tested local server can complete its core app,
data, schema, service, multimodal, and recovery operations on Pixeltable 0.7.6.

**Release condition:** preserve the exact default discovery surface and rerun
the locked suite, clean-wheel check, and conformance harness after every code or
dependency change.

### P1 — The 48-trial agent comparison is incomplete

This is the only direct blocker to the requested parity verdict. Code-level and
protocol tests cannot show whether an agent selects the right workflow, avoids
unsupported APIs, recovers efficiently, or performs as well as the skill and
website baselines.

**Correction:** resume the 43 errored trials from the checked-in evaluation
manifest when execution credits are available. Keep the MCP server's default
safe surface for the MCP condition and evaluate plugin hooks separately so they
are not credited to the skill text.

**Verification:** validate a complete results artifact; independently inspect
every failure and unsupported-API score; calculate scenario and overall
comparisons; refuse the tag if either baseline wins overall, a scenario repeats
a regression, or a critical defect remains.

### P1 — Release CI is wired; incomplete evaluation intentionally blocks tags

The candidate CI covers Python 3.11-3.14 with the frozen lock, exact minimums
Pixeltable 0.7.6/MCP 2.2.0, a fresh unpinned resolution within declared bounds,
Ruff format/check, Pyright, wheel/sdist build, clean-wheel install, real-stdio
integration through the test suite, and official conformance. Third-party
actions are pinned by commit, and conformance output is retained as an artifact
([`ci.yml:18-171`](../.github/workflows/ci.yml#L18-L171)).

Ordinary CI validates the evaluation artifact structure. A tag-only job depends
on every deterministic job and invokes the validator with
`--require-complete`; it will fail while any trial is incomplete or errored
([`ci.yml:173-195`](../.github/workflows/ci.yml#L173-L195)). That is the correct
release posture for the current evidence.

**Correction:** rerun and audit the 43 errored results. Do not weaken or bypass
the tag condition.

**Verification:** the ordinary artifact validator passes for the declared
matrix, the complete validator fails now, and it passes only after all trials
contain valid evidence and scores.

### P2 — The conformance runner for the target spec is a prerelease

The stable conformance package available on the review date does not accept
2026-07-28, so the harness pins `0.2.0-alpha.11`. The pin makes the current
result reproducible but carries normal prerelease-runner risk.

**Correction:** retain the exact pin for this candidate. When a stable runner
supports 2026-07-28, review its changelog, update the pin in a dedicated change,
and rerun both domain and frozen-requirement tests.

**Verification:** require the same 30/30 domain result and all scored frozen
server requirements; investigate any scenario reclassification instead of
blindly accepting a changed count.

### P2 — Website guidance still contains a separate legacy-workflow defect

At retrieval on 2026-09-10,
[`get-started.md`](https://www.pixeltable.com/get-started.md) was consistently
app-first. [`llms.txt`](https://www.pixeltable.com/llms.txt) contradicted itself:
its current scaffolding sections direct readers to `pxt service example`, while
later sections still promote `uvx pixeltable-new` and starter-kit templates.
The later material conflicts with pixeltable-skill 2.8.3 and the released 0.7.6
application-file workflow. This is a website-document defect; it is not a
reason to add the legacy workflow to the MCP server.

**Correction:** track the website correction outside this repository. Do not
copy the legacy starter-kit material into MCP prompts or resources.

**Verification:** when website trials run, record which sections were retrieved
and classify failures caused by the conflicting legacy section as website
defects, not MCP regressions.

### P2 — Provider and hosted behavior remain bounded evidence

The mocked tool-calling scenario establishes computed-column and tool-dispatch
wiring. It cannot establish provider authentication, rate-limit behavior,
response drift, or hosted service reconciliation. Cloud instructions have not
been applied to a live account.

**Correction:** keep these boundaries explicit in 0.2. A later, separately
authorized integration run may test one provider sandbox and one disposable
Cloud database; it must not become a prerequisite for deterministic pull-request
tests.

**Verification:** preserve provider-mock and not-live-tested labels in reports,
resources, and prompts until such runs exist.

## Decision-complete remediation and release plan

1. **Freeze the evaluation inputs.** Record the eight prompts, exact skill
   commit, website retrieval hashes/dates, model and reasoning settings, tool
   access, timeouts, deterministic fixtures, and per-scenario verifier.
2. **Complete 48 isolated trials.** Resume the 43 execution-error rows. Execute
   each condition twice for every scenario without exposing repository
   instructions or another Pixeltable skill to the baseline sessions. Capture
   retrievals and elapsed/tool-call data automatically.
3. **Score and audit the trials.** Produce the scenario comparison and overall
   result. Manually review all unsupported-API, recovery, and first-attempt
   failures. State mixed evidence directly.
4. **Enforce the release gate in CI.** Run locked, minimum-bound, and fresh
   resolutions; full tests on Python 3.11-3.14; Ruff; Pyright; build; clean-wheel
   install; real stdio; conformance; banned-pattern scan; and complete trial
   validation. Retain reports as artifacts.
5. **Make the tag decision.** Tag 0.2.0 only if the upgraded MCP condition
   matches or exceeds both baselines overall, no scenario repeats a regression,
   no critical defect remains, and every deterministic CI check passes. Do not
   publish or deploy as part of this review.
6. **Track follow-up work separately.** Website cleanup, a stable conformance
   runner update, live provider testing, and Cloud testing should each be
   explicit later changes. They do not justify weakening the deterministic 0.2
   gate.

## Reproduction commands

From a clean checkout of the candidate working tree with `uv`, Python 3.11+
and Node/npm available:

```bash
uv lock --check
uv sync --frozen --extra test --extra canvas
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
PIXELTABLE_DISABLE_STDOUT=1 uv run --frozen pytest -q --run-slow
uv build
./scripts/run-conformance.sh
```

Clean-wheel smoke test:

```bash
uv venv --python 3.12 .venv-wheel
uv pip install --python .venv-wheel/bin/python dist/*.whl
.venv-wheel/bin/mcp-server-pixeltable-developer --version
.venv-wheel/bin/python -c 'import mcp_server_pixeltable_developer'
.venv-wheel/bin/python -c 'import mcp_server_pixeltable_stio'
```

Default and unsafe discovery can be inspected without private MCP managers:

```bash
uv run python list_tools.py
PIXELTABLE_MCP_ENABLE_UNSAFE=1 uv run python list_tools.py
```

Prepare the pinned guidance snapshots, validate the matrix, and enforce the
release gate with:

```bash
uv run python evals/prepare_sources.py --output-dir /tmp/pixeltable-eval-sources --strict-mutable
uv run python evals/validate_results.py
uv run python evals/validate_results.py --require-complete
```

The last command is expected to fail until the 43 interrupted trials have been
rerun, scored, and linked to retained evidence.

## Limitations

- Local execution was performed on one host; CI is responsible for the full
  Python 3.11-3.14 matrix.
- Provider mocks prove wiring only. No paid provider request was made.
- Cloud preparation is source-reviewed and explicitly not live-tested. No
  hosted resource was created or changed.
- Semantic retrieval uses a deterministic local embedding; it does not prove a
  hosted embedding provider's quality or availability.
- Streamable HTTP exists solely for conformance. Production support is `stdio`.
- Tasks-extension failures from the official everything-server run are
  unscored and the extension is intentionally out of scope.
- The conformance runner is a pinned prerelease because the stable runner did
  not support the target specification on the review date.
- Five of 48 trials completed, all for the initial HTTP scenario; 43 were
  interrupted by exhausted execution credits. Until the matrix is complete,
  agent parity remains inconclusive.
- Two repetitions can expose repeatable regressions but cannot establish
  statistical superiority.
