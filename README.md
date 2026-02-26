# Agent_stock (Standalone Microservice)

Standalone Agent project extracted from `daily_stock_analysis`, now upgraded for independent microservice deployment.

## Scope

This project keeps the multi-agent paper-trading workflow with service-friendly boundaries:

- Data -> Signal -> Risk -> Execution pipeline
- Sync + Async run execution APIs
- Polling-based async task lifecycle
- PostgreSQL-first persistence in service mode
- Backend-oriented integration contracts (no default direct notification)

## Runtime Modes

- CLI mode (compatibility): run one-off or realtime cycles from `agent_main.py`
- Service mode (recommended for backend integration): run FastAPI via `agent_server.py`

## Layout

- `agent_main.py`: compatibility CLI entrypoint
- `agent_server.py`: microservice ASGI entrypoint
- `agent_api/`: API layer (routers, schemas, auth middleware)
- `agent_stock/`: core agent domain logic and repositories
- `scripts/migrate_agent_storage.py`: DB migration/bootstrap script
- `docker/`: Dockerfile and compose example
- `tests/`: regression and API tests

## Core API Contract

### 1) Create Run

`POST /api/v1/runs`

Request body:

```json
{
  "stock_codes": ["600519", "000001"],
  "async_mode": true,
  "request_id": "optional-idempotency-key",
  "account_name": "paper-default",
  "runtime_config": {
    "account": {
      "account_name": "user-123",
      "initial_cash": 100000,
      "account_display_name": "User 123"
    },
    "llm": {
      "provider": "openai",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o-mini",
      "api_token": "optional-runtime-token",
      "has_token": true
    },
    "strategy": {
      "position_max_pct": 30,
      "stop_loss_pct": 8,
      "take_profit_pct": 15
    },
    "execution": {
      "mode": "paper",
      "has_ticket": false,
      "credential_ticket": "optional-one-time-ticket",
      "ticket_id": 123,
      "broker_account_id": 88
    }
  }
}
```

Behavior:

- `async_mode=false`: returns run payload (`200`)
- `async_mode=true`: returns task payload (`202`)
- `runtime_config` is optional and applies only to the current request.
- `runtime_config.account.account_name` is used as account isolation key when provided.
- Account name length limit is `128` (top-level `account_name` and `runtime_config.account.account_name`).
- Strategy bounds accept `0..100`:
  - `position_max_pct=0`: disable opening new positions for this run.
  - `stop_loss_pct=0`: disable stop-loss trigger for this run.
  - `take_profit_pct=0`: disable take-profit trigger for this run.
- Stage snapshots include request-scoped observability fields (`duration_ms`, `input`, `output`).
- Risk/signal snapshots include compatibility fields:
  - `risk_snapshot`: `effective_stop_loss`, `effective_take_profit`, `position_cap_pct`, `strategy_applied`.
  - `signal_snapshot`: `resolved_stop_loss`, `resolved_take_profit`.
- `runtime_config.execution` is optional:
  - `mode` accepts `paper | broker`.
  - `mode=broker` requires `credential_ticket`.
  - Broker mode currently performs `ticket exchange + execution-event fallback audit`, then falls back to local paper execution because order-gateway contract is not enabled yet.
- Execution snapshots include compatibility fields:
  - `execution_mode`, `backend_task_id`, `broker_requested`, `executed_via`, `broker_ticket_id`, `fallback_reason`.
- Runtime tokens are never persisted in DB snapshots/task rows and are redacted from logs/errors.

### 2) Poll Task

`GET /api/v1/tasks/{task_id}`

Task status is one of: `pending | processing | completed | failed`.

### 3) Get Run

`GET /api/v1/runs/{run_id}`

Returns persisted run snapshots: data/signal/risk/execution + account snapshot.

### 4) List Runs

`GET /api/v1/runs?limit=20&status=completed&trade_date=2026-02-23`

### 5) Account Snapshot

`GET /api/v1/accounts/{account_name}/snapshot`

### 6) Health

- `GET /api/health/live`
- `GET /api/health/ready`

## Authentication

All endpoints except health checks require:

```http
Authorization: Bearer <AGENT_SERVICE_AUTH_TOKEN>
```

## Configuration

Service mode requires:

- `AGENT_SERVICE_MODE=true`
- `DATABASE_URL=postgresql+psycopg://...`
- `AGENT_SERVICE_AUTH_TOKEN=<token>`

Optional service knobs:

- `AGENT_SERVICE_HOST` (default `0.0.0.0`)
- `AGENT_SERVICE_PORT` (default `8001`)
- `AGENT_TASK_MAX_WORKERS` (default `3`)
- `AGENT_WRITE_LOCAL_REPORTS` (default `false`)
- `AGENT_LEGACY_NOTIFY_ENABLED` (default `false`)
- `AGENT_BACKEND_BASE_URL` (default `http://127.0.0.1:8002`)
- `AGENT_BACKEND_REQUEST_TIMEOUT_MS` (default `5000`)
- `AGENT_BACKEND_RETRY_COUNT` (default `1`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Database Migration

```bash
python scripts/migrate_agent_storage.py
```

## Run Service

```bash
uvicorn agent_server:app --host 0.0.0.0 --port 8001
```

## CLI Compatibility

```bash
python agent_main.py --mode once --stocks 600519
python agent_main.py --mode realtime --stocks 600519,000001 --interval-minutes 5
python agent_main.py --mode once --stocks 600519 --notify
```

Notes:

- CLI keeps compatibility with previous usage.
- Notification is disabled by default; use `--notify` to enable legacy behavior explicitly.

## Docker (Service + PostgreSQL)

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

## Validation

```bash
python scripts/check_import_boundaries.py
python -m py_compile agent_main.py agent_server.py agent_api/**/*.py agent_stock/**/*.py src/**/*.py data_provider/*.py
python -m pytest tests -q
```
