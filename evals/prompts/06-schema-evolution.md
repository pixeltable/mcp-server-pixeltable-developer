Demonstrate Pixeltable schema evolution with four project files.

Create `schema_v1.py` with a `TableModel` table named `items`: integer primary key `id`, required string `value`, and computed `normalized` equal to uppercased `value`. Create `schema_v2.py` as the additive version with an optional string column `note`. Create `schema_expression_change.py` by changing only the existing `normalized` expression from uppercase to lowercase. Create `SCHEMA_EVOLUTION.md` with the exact check, diff, and update commands used against target `eval_evolution`.

Initialize the project; apply v1; show that v2 is additive and apply it; then show that removing `note` is destructive and requires explicit destructive authorization. Prove that the in-place computed-expression change is unsupported even when destructive changes are authorized. Explain the supported recovery: rename the computed column, or drop and re-add it in separate updates. Leave all files for independent replay.
