Prepare, but do not execute, a Pixeltable Cloud deployment for organization `acme` and database `main`.

Create `app.py` with a small `TableModel` table named `messages` and a `FastAPIRouter` insert route. Create `pixeltable.cloud.toml.example` containing the hosted database declaration for `pxt://acme:main`, using an environment-provided `PIXELTABLE_API_KEY` rather than a literal secret. Create `CLOUD.md` with exact commands in the required order: database image/config update, schema update, then service update. Include check/diff steps where the current CLI supports them and explain how to discover the service endpoint.

Perform only local syntax or schema checks. Do not invoke a hosted target, create a resource, deploy, or make a paid provider call. Label every Cloud and provider assertion as not live-tested.
