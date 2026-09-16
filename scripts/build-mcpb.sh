#!/usr/bin/env bash
# Build the MCP Bundle (.mcpb) submitted to the Claude desktop extension directory.
# Staging comes from `git archive`, so only tracked files ship: a plain copy would
# sweep in __pycache__ from deleted code and quadruple the bundle.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "${repo_root}/pyproject.toml" | head -1)"
staging="$(mktemp -d)"
output="${1:-${repo_root}/dist/mcp-server-pixeltable-developer-${version}.mcpb}"

trap 'rm -rf "${staging}"' EXIT

git -C "${repo_root}" archive --format=tar HEAD \
    src pyproject.toml uv.lock README.md LICENSE canvas.html | tar -x -C "${staging}"
cp "${repo_root}/mcpb/manifest.json" "${staging}/manifest.json"

mkdir -p "$(dirname "${output}")"
npx --yes @anthropic-ai/mcpb@latest validate "${staging}/manifest.json"
npx --yes @anthropic-ai/mcpb@latest pack "${staging}" "${output}"
echo "Built ${output}"
