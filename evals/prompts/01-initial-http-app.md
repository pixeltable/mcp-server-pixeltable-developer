Build and run a minimal Pixeltable HTTP application in `app.py`.

The application must declare a `TableModel` table named `notes` with an integer primary key `id`, a required string `text`, and a stored computed string column `upper` that uppercases `text`. Declare a `FastAPIRouter` named `api` with a `POST /notes` insert route that accepts `id` and `text` and returns `upper`.

Initialize the project, check and apply the schema to the local catalog directory `eval_http`, start the service, discover its assigned endpoint, and prove that posting `{"id": 1, "text": "hello"}` returns `{"upper": "HELLO"}`. Do not hard-code a port. Leave the source in place and report the commands, endpoint, and observed response.
