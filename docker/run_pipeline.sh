#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${1:-scripts/retrieval/run_all.sh}"
shift || true

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Pipeline script '${SCRIPT_PATH}' not found." >&2
  exit 1
fi

exec bash "${SCRIPT_PATH}" "$@"
