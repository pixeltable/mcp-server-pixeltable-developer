#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
python_scripts_dir="$("${python_bin}" -c 'import os, sys; print(os.path.dirname(sys.executable))')"
pxt_bin="${PIXELTABLE_MCP_PXT:-${python_scripts_dir}/pxt}"
uv_bin="${UV_BIN:-uv}"
conformance_package="@modelcontextprotocol/conformance@0.2.0-alpha.11"
sdk_tag="v2.2.0"
sdk_commit="9972c21aa42054fb1450c5fc614761ed11847ec6"
output_root="${CONFORMANCE_OUTPUT_DIR:-${repo_root}/conformance-results}"
work_dir="$(mktemp -d)"
domain_pid=""
fixture_pid=""

mkdir -p "${output_root}/domain" "${output_root}/requirements" "${work_dir}/project" "${work_dir}/catalog"

if [[ ! -x "${pxt_bin}" ]]; then
  echo "Pixeltable CLI is not executable: ${pxt_bin}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${domain_pid}" ]]; then
    kill "${domain_pid}" 2>/dev/null || true
    wait "${domain_pid}" 2>/dev/null || true
  fi
  if [[ -n "${fixture_pid}" ]]; then
    kill "${fixture_pid}" 2>/dev/null || true
    wait "${fixture_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

pick_port() {
  "${python_bin}" - <<'PY'
import socket

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
}

wait_for_port() {
  local port="$1"
  "${python_bin}" - "${port}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            raise SystemExit(0)
    except OSError:
        time.sleep(0.1)
raise SystemExit(f"server did not listen on 127.0.0.1:{port}")
PY
}

"${python_bin}" - <<'PY'
from importlib.metadata import version

assert version("mcp") == "2.2.0", version("mcp")
assert version("pixeltable") == "0.7.6", version("pixeltable")
PY

domain_port="$(pick_port)"
PIXELTABLE_MCP_PROJECT_ROOT="${work_dir}/project" \
PIXELTABLE_HOME="${work_dir}/catalog" \
PIXELTABLE_MCP_PXT="${pxt_bin}" \
PIXELTABLE_DISABLE_STDOUT=1 \
  "${python_bin}" -m uvicorn \
  mcp_server_pixeltable_developer.server:create_conformance_app \
  --factory --host 127.0.0.1 --port "${domain_port}" \
  >"${output_root}/domain-server.log" 2>&1 &
domain_pid=$!
wait_for_port "${domain_port}"
npx --yes "${conformance_package}" server \
  --url "http://127.0.0.1:${domain_port}/mcp" \
  --scenario server-stateless \
  --spec-version 2026-07-28 \
  --output-dir "${output_root}/domain"
kill "${domain_pid}" 2>/dev/null || true
wait "${domain_pid}" 2>/dev/null || true
domain_pid=""

git clone --quiet --depth 1 --branch "${sdk_tag}" \
  https://github.com/modelcontextprotocol/python-sdk "${work_dir}/python-sdk"
actual_sdk_commit="$(git -C "${work_dir}/python-sdk" rev-parse HEAD)"
if [[ "${actual_sdk_commit}" != "${sdk_commit}" ]]; then
  echo "Unexpected MCP SDK ${sdk_tag} commit: ${actual_sdk_commit}" >&2
  exit 1
fi
"${uv_bin}" pip install --python "${python_bin}" --no-deps \
  "${work_dir}/python-sdk/examples/servers/everything-server"

fixture_port="$(pick_port)"
"${python_scripts_dir}/mcp-everything-server" --port "${fixture_port}" \
  >"${output_root}/requirements-server.log" 2>&1 &
fixture_pid=$!
wait_for_port "${fixture_port}"
npx --yes "${conformance_package}" server \
  --url "http://127.0.0.1:${fixture_port}/mcp" \
  --requirements 2026-07-28 \
  --output-dir "${output_root}/requirements"
