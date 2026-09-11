# Pixeltable agent evaluation

This directory defines the release-gating comparison for version 0.2. It uses
eight tasks, three guidance conditions, and two repetitions: 48 trials total.

| Condition | Guidance | Additional capability |
|---|---|---|
| `mcp_skill` | Pixeltable skill 2.8.3 and all five references | Safe 0.2 MCP server |
| `skill_only` | The same skill snapshot and references | None |
| `website_docs` | `get-started.md` and `llms.txt` | None |

The skill source is fixed at commit
`f550e6ed757b48635e4f53900840f4e9a1fb4c93`. The website documents are
mutable, so a run fails if their bytes differ from `sources.lock.json` unless
the operator explicitly allows drift. Review and update the source lock before
using changed website text for a release decision.

## Isolation and controls

`run_trials.py` creates a new Codex session, workspace, and agent catalog for
each trial. It launches Codex with `--ephemeral`, `--ignore-user-config`,
`--ignore-rules`, and a workspace with no `AGENTS.md` or repository context.
Only the selected condition's guidance and the checked-in deterministic
fixtures are copied into that workspace. The model, reasoning effort, sandbox,
prompt wrapper, time limit, output limit, and tool-call limit are fixed and
recorded. The MCP condition differs only by the capability being evaluated.

Trials run sequentially. Pixeltable 0.7.6 uses a process-global local CLI
daemon, so concurrent trials with separate `PIXELTABLE_HOME` values can collide.
The runner stops that daemon before and after the agent phase; the verifier then
replays the result in a second fresh catalog and stops it again.

The MCP unsafe mode is removed from every trial environment. Provider responses
are mocked and Cloud work is instruction-only. No paid call, hosted deployment,
or Cloud resource is part of this evaluation.

## Run

Use the locked Python environment and inspect the exact selection before model
execution:

```bash
uv sync --frozen --extra test --extra canvas
uv run python evals/fixtures/generate.py --check
uv run python evals/run_trials.py --all --dry-run
```

Run one pilot first:

```bash
uv run python evals/run_trials.py \
  --trial initial_http_app__mcp_skill__r1 \
  --model gpt-6-astra \
  --reasoning-effort low
```

After reviewing the pilot, run every remaining row with the same settings:

```bash
uv run python evals/run_trials.py \
  --all \
  --model gpt-6-astra \
  --reasoning-effort low
```

The runner requires an explicit `--all` or `--trial`; it never starts billable
model work merely because the script was invoked. A completed row is skipped
unless `--rerun` is supplied. Each new invocation uses a new run ID, while the
matrix can reference evidence from several run directories.

Source preparation can also be reproduced independently:

```bash
uv run python evals/prepare_sources.py \
  --output-dir /tmp/pixeltable-eval-sources \
  --strict-mutable
```

## Evidence and scoring

Every trial retains its exact wrapper prompt, final answer, bounded JSONL
transcript, stderr, source retrieval manifest, independent verifier output, and
an evidence envelope with hashes, settings, tool-call counts, and elapsed time.
The raw prompt, workspace, transcript, stderr, final response, and downloaded
source bundle are ignored by Git because they may be large or contain local
paths. The settings, generated summary, verifier result, and hashed evidence
envelope remain visible for audit and can be included in a release review.

`verify_trial.py` copies the output into a fresh project and catalog. It checks
required files, Python syntax, retired APIs, unnecessary frameworks, schema
application, and the scenario's deterministic behavior. Runtime behavior sets
`executable_correctness`; artifact completeness sets `task_completion`. A
trial earns `first_attempt_success=1` only when its single agent session passes
the independent verifier. The two recovery scenarios also receive a recovery
score. Tool calls and elapsed time are measured by the runner.

Validate the matrix structure at any time:

```bash
uv run python evals/validate_results.py
```

The release gate additionally requires all 48 trials, authenticates each row
against its evidence envelope and source lock, rejects critical defects, checks
overall mean quality against both baselines, and rejects a scenario where the
MCP condition loses to one baseline in both repetitions:

```bash
uv run python evals/validate_results.py --require-complete
```

Two repetitions can expose a repeated regression. They cannot establish
statistical superiority, so mixed results must remain labeled mixed.
