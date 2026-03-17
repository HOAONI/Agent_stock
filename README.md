# Agent_stock

`Agent_stock` 是一个独立的 Python Agent 服务，负责股票分析、模拟交易、运行时账户管理，以及供 `Backend_stock` 调用的内部回测/运行时接口。

## 项目定位

- 这是一个“分析 + 编排 + 执行 + 回测”一体化服务，不是前端项目。
- CLI 入口是 [`agent_main.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_main.py)，适合本地单次运行或实时轮询。
- 微服务入口是 [`agent_server.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_server.py)，对外暴露 FastAPI 接口。
- 核心业务命名空间是 `agent_stock.*`、`agent_api.*`、`data_provider.*`、`patch.*`。

## 快速上手

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

服务模式至少需要以下配置：

- `AGENT_SERVICE_MODE=true`
- `DATABASE_URL=postgresql+psycopg://...`
- `AGENT_SERVICE_AUTH_TOKEN=<token>`

常用可选配置：

- `AGENT_SERVICE_HOST`
- `AGENT_SERVICE_PORT`
- `AGENT_TASK_MAX_WORKERS`
- `AGENT_WRITE_LOCAL_REPORTS`
- `AGENT_LLM_REQUEST_TIMEOUT_MS`
- `REALTIME_SOURCE_PRIORITY`

## 运行方式

启动服务：

```bash
uvicorn agent_server:app --host 0.0.0.0 --port 8001
```

单次运行：

```bash
python agent_main.py --mode once --stocks 600519
```

实时轮询：

```bash
python agent_main.py --mode realtime --stocks 600519,000001 --interval-minutes 5
```

## 主执行链路

一次典型运行会按下面顺序流转：

1. `agent_main.py` 或 `agent_api` 接收请求，解析 CLI/API 参数与 `runtime_config`。
2. `agent_stock.services.agent_service.AgentService` 负责组装运行上下文，并调用编排器。
3. `agent_stock.agents.orchestrator.AgentOrchestrator` 按顺序执行：
   `DataAgent -> SignalAgent -> RiskAgent -> ExecutionAgent`
4. `agent_stock.repositories.execution_repo.ExecutionRepository` 和 `agent_stock.storage.DatabaseManager`
   负责账户、持仓、任务、运行快照和搜索/分析结果落库。
5. 根据配置，运行结果可以写入本地 Markdown/CSV 报表，也可以通过 API 查询。

## 目录职责

- [`agent_stock/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock)
  核心业务目录，包含配置、分析器、多 Agent 编排、存储、报表、回测服务。
- [`agent_api/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_api)
  FastAPI 应用、路由、Schema、鉴权中间件和依赖注入。
- [`data_provider/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/data_provider)
  多数据源抓取层，统一封装 AkShare、Tushare、Baostock、Pytdx、Yfinance 等接口。
- [`patch/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/patch)
  对第三方请求行为的补丁，例如东方财富反爬请求头补丁。
- [`scripts/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/scripts)
  迁移脚本和工程检查脚本。
- [`tests/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/tests)
  覆盖 API、运行时、回测、异步任务与数据处理的回归测试。

## 推荐阅读顺序

第一次接手这个项目，建议按下面顺序阅读：

1. [`agent_main.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_main.py)
   先理解 CLI 如何启动一次运行或实时循环。
2. [`agent_server.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_server.py) 和 [`agent_api/app.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_api/app.py)
   看清服务模式如何启动和挂路由。
3. [`agent_stock/config.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/config.py) 与 [`agent_stock/storage.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/storage.py)
   理解配置来源、数据库连接和核心表结构。
4. [`agent_stock/agents/contracts.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/agents/contracts.py)
   先熟悉运行过程中各阶段产出的统一数据结构。
5. [`agent_stock/agents/orchestrator.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/agents/orchestrator.py)
   掌握数据、信号、风控、执行是怎样串起来的。
6. `agent_stock/services/*`
   重点看 `agent_service.py`、`agent_task_service.py`、`backtest_service.py`、`strategy_backtest_service.py`、`agent_historical_backtest_service.py`。
7. [`data_provider/base.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/data_provider/base.py) 与各个 `*_fetcher.py`
   理解多数据源优先级、降级策略和实时行情补齐逻辑。
8. [`agent_api/v1/endpoints/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_api/v1/endpoints)
   对照接口了解 Backend_stock 是如何调用本服务的。
9. [`tests/`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/tests)
   从复杂测试场景反推业务边界和历史 bug。

## 常见调试入口

- 编排主链路：[`agent_stock/agents/orchestrator.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/agents/orchestrator.py)
- AI 分析与 LLM 兜底：[`agent_stock/analyzer.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/analyzer.py)
- 新闻搜索与缓存：[`agent_stock/search_service.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/search_service.py)
- 账户、持仓、任务落库：[`agent_stock/repositories/execution_repo.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/repositories/execution_repo.py)
- 历史回放与策略回测：[`agent_stock/services/agent_historical_backtest_service.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/services/agent_historical_backtest_service.py)、[`agent_stock/services/strategy_backtest_service.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/agent_stock/services/strategy_backtest_service.py)
- 数据源切换：[`data_provider/base.py`](/Users/hoaon/Desktop/毕设相关/project/v4/Agent_stock/data_provider/base.py)

## API 概览

对外运行接口：

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/accounts/{account_name}/snapshot`

健康检查：

- `GET /api/health/live`
- `GET /api/health/ready`

供 Backend_stock 调用的内部接口：

- `POST /internal/v1/backtrader/accounts/provision`
- `POST /internal/v1/backtrader/account-summary`
- `POST /internal/v1/backtrader/positions`
- `POST /internal/v1/backtrader/orders`
- `POST /internal/v1/backtrader/trades`
- `POST /internal/v1/backtrader/place-order`
- `POST /internal/v1/backtrader/cancel-order`
- `POST /internal/v1/backtest/run`
- `POST /internal/v1/backtest/summary`
- `POST /internal/v1/backtest/curves`
- `POST /internal/v1/backtest/distribution`
- `POST /internal/v1/backtest/compare`

除健康检查外，所有接口默认都要求：

```http
Authorization: Bearer <AGENT_SERVICE_AUTH_TOKEN>
```

`runtime_config` 支持请求级覆盖 `account`、`llm`、`strategy`、`execution` 和 `context`。运行时密钥会在日志中脱敏，且不会持久化保存。

## 验证命令

```bash
python scripts/check_import_boundaries.py
python -m pytest tests -q
python -c "import agent_main, agent_server, agent_api.app; print('imports ok')"
```

当 `AGENT_WRITE_LOCAL_REPORTS=true` 时，Markdown 与 CSV 报表会落到 `logs/agent_reports/<trade-date>/`。
