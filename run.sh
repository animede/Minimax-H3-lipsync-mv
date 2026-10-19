#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec .venv/bin/uvicorn app.main:app --host "${APP_HOST:-127.0.0.1}" --port "${APP_PORT:-8780}"
