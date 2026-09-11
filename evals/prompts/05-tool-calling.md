Build a deterministic tool-calling application in `app.py` without making a paid provider call.

Declare a typed `@pxt.udf` named `add_numbers(a: int, b: int) -> int`. Declare a second deterministic UDF that returns an OpenAI-compatible response containing a tool call to `add_numbers` with arguments 2 and 3. Create `agent_tools = pxt.tools(add_numbers)`. Declare a `TableModel` table named `agent_runs` with integer primary key `id`, required string `question`, a stored computed response column, and a stored computed `tool_results` column using the current provider module's `invoke_tools` function.

Initialize the project, check and apply the schema to `eval_agent`, insert one row, and prove that the stored tool result is `{"add_numbers": [5]}`. Report executable evidence and label the model response as a wiring mock rather than a live provider test.
