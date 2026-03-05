# Changelog

## 2026-03-01

### Changed
- Converged `runtime_config.execution` to simulation-only semantics:
  - accepted fields: `mode|has_ticket|broker_account_id`
  - removed/rejected fields: `credential_ticket|ticket_id`
- Added optional runtime account context contract:
  - `runtime_config.context.account_snapshot|summary|positions`
  - used as primary risk/execution account input.
- Simplified `ExecutionAgent` runtime flow to local paper engine only (no broker ticket exchange, no broker fallback event posting).
- Switched `ExecutionAgent` to light-state intent mode:
  - no persistence writes to `paper_accounts|paper_positions|paper_orders|paper_trades`
  - output remains compatibility-safe (`execution_mode/executed_via=paper`) with projected account snapshot.
- Changed `/api/v1/accounts/{account_name}/snapshot` compatibility semantics:
  - now sourced from latest persisted run snapshot context, not realtime paper ledger.
- Kept execution snapshot compatibility fields while fixing simulation-only values:
  - `execution_mode='paper'`, `broker_requested=false`, `executed_via='paper'`, `broker_ticket_id=null`, `fallback_reason=null`
- Updated tests and docs to remove broker fallback expectations.
- Clarified integration boundary: third-party simulation order submission is executed by `Backend_stock` worker after analysis, not by Agent.

### Breaking
- `POST /api/v1/runs` now rejects `runtime_config.execution.mode=broker` with schema `422` validation error.

## 2026-02-25

### Changed
- Aligned runtime strategy validation with backend profile settings: `position_max_pct`, `stop_loss_pct`, `take_profit_pct` now accept `0..100`.
- Defined `0` strategy semantics for request-scoped risk control:
  - `position_max_pct=0` blocks new opening exposure for the current run.
  - `stop_loss_pct=0` disables stop-loss trigger for the current run.
  - `take_profit_pct=0` disables take-profit trigger for the current run.
- Expanded account name limit to `128` for API schema and storage models (`paper_accounts.name`, `agent_runs.account_name`, `agent_tasks.account_name`).
- Improved task entrypoint account resolution to honor `runtime_config.account.account_name` even when top-level `account_name` is omitted.
- Enhanced stage observability snapshots with `duration_ms`, `input`, and `output` fields.
- Added compatibility fields for downstream mapping:
  - `signal_snapshot`: `resolved_stop_loss`, `resolved_take_profit`
  - `risk_snapshot`: `effective_stop_loss`, `effective_take_profit`, `position_cap_pct`, `strategy_applied`
- Updated agent analysis context to include historical `raw_data` bars for trend-analysis path reliability.
- Extended runtime contract with optional `runtime_config.execution` (`mode|has_ticket|credential_ticket|ticket_id|broker_account_id`).
- Added request-id passthrough from API/task service to execution stage as `backend_task_id`.
- Extended execution snapshots with `execution_mode`, `backend_task_id`, `broker_requested`, `executed_via`, `broker_ticket_id`, `fallback_reason`.
- Added broker bridge fallback path in `ExecutionAgent`: exchange ticket via backend internal API, then fallback to local paper execution with execution-event audit when broker order contract is unavailable.
- Expanded sensitive redaction to cover credential-ticket patterns in logs/errors.

### Added
- Added tests:
  - `tests/test_agent_stage_observability.py`
  - `tests/test_agent_account_name_length.py`
  - `tests/test_execution_agent_broker_runtime.py`

## 2026-02-24

### Added
- Added optional `runtime_config` payload support for `POST /api/v1/runs` (account + llm + strategy).
- Added request-level runtime passthrough across endpoint -> task service -> async worker -> orchestrator.
- Added runtime validation/regression/isolation test coverage:
  - `tests/test_agent_runtime_config_validation.py`
  - `tests/test_agent_task_runtime_passthrough.py`
  - `tests/test_agent_runtime_isolation.py`

### Changed
- Kept bearer authentication unchanged while making run-time config request-scoped and concurrency-safe.
- Extended risk/execution/signal path to consume request-level strategy/account/llm overrides without mutating global singleton config.
- Added sensitive-token redaction helpers for runtime error/log paths.
- Preserved backward compatibility for requests that do not include `runtime_config`.

## 2026-02-23

### Added
- Added a dedicated Agent microservice API layer under `agent_api/`.
- Added synchronous and asynchronous run APIs with task polling support.
- Added static bearer authentication middleware for non-health endpoints.
- Added account snapshot and run query endpoints for backend integration.
- Added `agent_server.py` as ASGI service entrypoint.
- Added `agent_stock/services/agent_task_service.py` for async task lifecycle.
- Added `agent_tasks` storage model and repository CRUD methods.
- Added migration script `scripts/migrate_agent_storage.py`.
- Added Docker deployment files for service + PostgreSQL.

### Changed
- Refactored `agent_stock/services/agent_service.py` to decouple notifications from default path.
- Updated orchestrator and execution agent to support request-level `account_name`.
- Extended run persistence to store `account_name`, `started_at`, and `ended_at`.
- Extended config with service-specific fields (auth token, DB URL, host/port, task workers).
- Kept CLI compatibility and introduced explicit `--notify` switch.

### Notes
- Service mode expects PostgreSQL via `DATABASE_URL` and a non-empty `AGENT_SERVICE_AUTH_TOKEN`.
- Notification sending is disabled by default in service execution path.
