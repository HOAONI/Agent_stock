#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/system"
PID_FILE="${LOG_DIR}/agent_service.pid"
OUT_FILE="${LOG_DIR}/agent_service.out"
STOP_GRACE_SECONDS="${AGENT_SERVICE_STOP_GRACE_SECONDS:-20}"

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

if [ ! -f "${PID_FILE}" ]; then
  echo "Agent service is not running (PID file not found)."
  exit 0
fi

service_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
if [ -z "${service_pid}" ]; then
  echo "Agent service PID file is empty. Removing stale PID file."
  rm -f "${PID_FILE}"
  exit 0
fi

if ! is_agent_service_pid "${service_pid}"; then
  echo "Agent service is not running; removing stale PID file."
  rm -f "${PID_FILE}"
  exit 0
fi

echo "Stopping Agent service (PID ${service_pid})."
kill "${service_pid}" >/dev/null 2>&1 || true

deadline=$((SECONDS + STOP_GRACE_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if ! kill -0 "${service_pid}" >/dev/null 2>&1; then
    rm -f "${PID_FILE}"
    echo "Agent service stopped gracefully."
    exit 0
  fi
  sleep 1
done

if is_agent_service_pid "${service_pid}"; then
  echo "Agent service did not stop within ${STOP_GRACE_SECONDS}s; sending SIGKILL."
  kill -9 "${service_pid}" >/dev/null 2>&1 || true
  sleep 1
fi

if kill -0 "${service_pid}" >/dev/null 2>&1; then
  echo "Failed to stop Agent service (PID ${service_pid}). Check ${OUT_FILE} for details." >&2
  exit 1
fi

rm -f "${PID_FILE}"
echo "Agent service stopped."
