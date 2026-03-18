#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/system"
PID_FILE="${LOG_DIR}/agent_service.pid"
OUT_FILE="${LOG_DIR}/agent_service.out"
START_TIMEOUT_SECONDS="${AGENT_SERVICE_START_TIMEOUT_SECONDS:-60}"
HEALTH_PATH_LIVE="/api/health/live"
HEALTH_PATH_READY="/api/health/ready"
ENV_FILE="${AGENT_SERVICE_ENV_FILE:-${PROJECT_ROOT}/.env}"

mkdir -p "${LOG_DIR}"

if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Failed to start Agent service: python interpreter not found (.venv/bin/python or python3)." >&2
  exit 1
fi

is_agent_service_pid() {
  local pid="$1"
  local cmdline=""

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 1
  fi

  cmdline="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  if [ -z "${cmdline}" ]; then
    return 1
  fi

  case "${cmdline}" in
    *"${PROJECT_ROOT}/agent_server.py"*|*" agent_server.py"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

probe_http() {
  local url="$1"

  "${PYTHON_BIN}" - "${url}" <<'PY'
import sys
import urllib.request

url = sys.argv[1]

try:
    with urllib.request.urlopen(url, timeout=2) as response:
        if 200 <= response.getcode() < 300:
            raise SystemExit(0)
except Exception:
    pass

raise SystemExit(1)
PY
}

cleanup_failed_start() {
  local pid="$1"

  if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${PID_FILE}"
}

if [ -f "${PID_FILE}" ]; then
  existing_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [ -n "${existing_pid}" ] && is_agent_service_pid "${existing_pid}"; then
    echo "Agent service is already running (PID ${existing_pid})."
    echo "Log file: ${OUT_FILE}"
    exit 0
  fi

  echo "Removing stale PID file: ${PID_FILE}"
  rm -f "${PID_FILE}"
fi

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "Warning: ${ENV_FILE} not found; using current process environment."
fi

export AGENT_SERVICE_MODE=true
AGENT_SERVICE_HOST="${AGENT_SERVICE_HOST:-0.0.0.0}"
AGENT_SERVICE_PORT="${AGENT_SERVICE_PORT:-8001}"

missing_config=()
if [ -z "${DATABASE_URL:-}" ]; then
  missing_config+=("DATABASE_URL")
fi
if [ -z "${AGENT_SERVICE_AUTH_TOKEN:-}" ]; then
  missing_config+=("AGENT_SERVICE_AUTH_TOKEN")
fi
if [ "${#missing_config[@]}" -gt 0 ]; then
  echo "Failed to start Agent service: missing required service configuration: ${missing_config[*]}." >&2
  exit 1
fi

case "${AGENT_SERVICE_HOST}" in
  0.0.0.0|"::"|"[::]"|"")
    HEALTH_HOST="127.0.0.1"
    ;;
  *)
    HEALTH_HOST="${AGENT_SERVICE_HOST}"
    ;;
esac

HEALTH_BASE_URL="http://${HEALTH_HOST}:${AGENT_SERVICE_PORT}"

cd "${PROJECT_ROOT}"

echo "Starting Agent service on ${AGENT_SERVICE_HOST}:${AGENT_SERVICE_PORT}"
echo "Log file: ${OUT_FILE}"

nohup "${PYTHON_BIN}" "${PROJECT_ROOT}/agent_server.py" >>"${OUT_FILE}" 2>&1 </dev/null &
service_pid="$!"
echo "${service_pid}" > "${PID_FILE}"

deadline=$((SECONDS + START_TIMEOUT_SECONDS))
live_ready=0
ready_ready=0

while [ "${SECONDS}" -lt "${deadline}" ]; do
  if ! is_agent_service_pid "${service_pid}"; then
    echo "Agent service exited before becoming healthy. Recent logs:" >&2
    tail -n 20 "${OUT_FILE}" >&2 || true
    cleanup_failed_start "${service_pid}"
    exit 1
  fi

  if probe_http "${HEALTH_BASE_URL}${HEALTH_PATH_LIVE}"; then
    live_ready=1
  fi

  if probe_http "${HEALTH_BASE_URL}${HEALTH_PATH_READY}"; then
    ready_ready=1
  fi

  if [ "${live_ready}" -eq 1 ] && [ "${ready_ready}" -eq 1 ]; then
    echo "Agent service started successfully (PID ${service_pid})."
    echo "Health endpoints: ${HEALTH_BASE_URL}${HEALTH_PATH_LIVE} , ${HEALTH_BASE_URL}${HEALTH_PATH_READY}"
    exit 0
  fi

  sleep 1
done

echo "Agent service did not become healthy within ${START_TIMEOUT_SECONDS}s. Recent logs:" >&2
tail -n 20 "${OUT_FILE}" >&2 || true
cleanup_failed_start "${service_pid}"
exit 1
