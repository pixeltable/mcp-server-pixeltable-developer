FROM python:3.11-slim

LABEL io.modelcontextprotocol.server.name="io.github.pixeltable/mcp-server-pixeltable-developer"

RUN pip install --no-cache-dir --upgrade uv

WORKDIR /app
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PIXELTABLE_DISABLE_STDOUT=1

ENTRYPOINT ["uv", "run", "mcp-server-pixeltable-developer"]
