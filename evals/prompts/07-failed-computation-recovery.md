Build and exercise deterministic failed-computation recovery in `app.py`.

Declare a `TableModel` table named `items` with integer primary key `id`, required string `value`, and computed `recovered`. Implement `recovered` with a typed local UDF that raises `ValueError('fixture not ready')` until a file named `recovery.ready` exists beside `app.py`, then returns the uppercased value.

Initialize the project and apply the schema to `eval_recovery`. Insert rows for `one` and `two` while allowing computed errors. List the stored errors. Preview recomputation of exactly the `recovered` column with both dry-run and errors-only behavior, create `recovery.ready`, then apply errors-only recomputation and prove the errors are gone and row 2 contains `TWO`. Write the replay commands and observations to `RECOVERY.md`.
