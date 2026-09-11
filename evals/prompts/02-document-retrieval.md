Build a deterministic local document-retrieval application in `app.py` using the provided `fixtures/document.html`.

Declare a `TableModel` table named `docs` with integer primary key `id` and a `pxt.Document` column `document`. Declare a `chunks` iterator view with `pixeltable.functions.document.document_splitter(..., separators='paragraph')`. Add a deterministic lexical retrieval query named `search_documents` that accepts `query_text`, scores or filters chunks without downloading a model or calling a provider, and returns matching text with the best result first. A query for `beta` must return the paragraph containing `beta second paragraph`.

Initialize the project, check and apply the schema to `eval_docs`, insert the fixture, run the retrieval, and report executable evidence. Do not install an embedding model, orchestration framework, or separate vector database for this deterministic fixture.
