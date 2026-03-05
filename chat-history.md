# Agent_stock 项目学习与实操对话全记录

本记录涵盖了对 `Agent_stock` 项目的深度解析、核心代码逻辑定位以及修改指南。

---

## 1. 项目全局架构与生效代码确认

**用户：** 全面学习这个项目，然后帮我教会这个项目。达到我想修改的时候知道去哪修改，修改什么！

**助手：**
项目是一个基于 AI Agent 架构的自动化股票分析/交易系统。
- **生效 Agent 路径**：确认目前系统真正使用的是 `agent_stock/agents/` 而非 `src/agents/`。
- **指挥官 (Orchestrator)**：`agent_stock/agents/orchestrator.py` 是核心，负责串联 Data -> Signal -> Risk -> Execution。
- **业务入口**：`agent_main.py` (CLI) 和 `agent_server.py` (API)。

---

## 2. AI 决策逻辑与 Prompt 定位

**用户：** 调用 llm 的时候发给 llm 的提示词是什么？在哪？

**助手：**
- **提示词定义**：位于 `src/analyzer.py` 中的 `SYSTEM_PROMPT` 变量。
- **内容核心**：
  1. **严进策略**：乖离率 > 5% 严禁追高。
  2. **趋势交易**：必须满足 MA5 > MA10 > MA20 多头排列。
  3. **决策仪表盘**：AI 必须输出结构化的 JSON 结果，包含一句话结论、狙击点位（买入/止损/目标价）。
- **数据包装**：在 `src/analyzer.py` 的 `_format_prompt` 方法中，将量化指标（K线、筹码、均线）和新闻情报拼接到 Prompt 中。

---

## 3. 混合式决策架构：纯 AI 还是量化？

**用户：** 最后的结论是纯 llm 的结论吗？还是说除了 llm，还有多种其他量化策略，最后综合出结果？

**助手：**
系统采用的是 **“量化特征提取 -> AI 逻辑推理 -> 硬性风险过滤”** 的混合架构。
1. **量化初筛** (`src/stock_analyzer.py`)：提取均线、MACD 等硬指标。
2. **AI 决策** (`src/analyzer.py`)：LLM 结合量化指标 + 联网搜索的新闻进行综合判断。
3. **风控拦截** (`agent_stock/agents/risk_agent.py`)：即便 AI 喊买，如果触碰仓位上限或止损红线，Risk Agent 会一票否决。

---

## 4. 全自动挂机与交易能力

**用户：** 目前 Agent 有直接全自动挂机，全自动交易的能力吗？

**助手：**
- **全自动挂机**：具备。通过 `python agent_main.py --mode realtime` 开启实时轮询模式。
- **交易模式**：
  - **模拟盘 (Paper)**：默认开启，通过自研引擎在数据库记录交易。
  - **实盘 (Real)**：架构已预留接口，但需手动接入券商 API（如 easytrader 或 QMT）。

---

## 5. 自研模拟盘 vs. Backtrader

**用户：** 不需要接入实盘，只要接入 backtrader 模拟盘就行。本系统实现了吗？

**助手：**
系统实现了一个 **“自研、类 Backtrader”** 的模拟盘引擎：`agent_stock/services/backtrader_runtime_service.py`。
- **为什么自研？**：原生 Backtrader 是内存运行，无法持久化存档；自研引擎支持数据库存储，关机重启后持仓和余额依然存在。
- **如何修改游戏规则？**：
  - **初始资金**：`.env` 里的 `BACKTRADER_DEFAULT_INITIAL_CAPITAL`。
  - **手续费/滑点**：`.env` 里的 `BACKTRADER_DEFAULT_COMMISSION`。

---

## 6. 收益监控与历史曲线

**用户：** 那这怎么看收益之类的数据呢？而且实时更新收益之类的呢？怎么看历史数据和收益曲线？

**助手：**
- **实时收益**：系统实时通过 `ExecutionRepository.recompute_account_metrics` 计算。
- **计算逻辑**：`总资产 = 现金 + (最新价 * 持仓数量)`。
- **历史记录**：每次运行都会在 `agent_runs` 表存入快照。
- **查看方式**：
  1. **Markdown 报表**：`logs/agent_reports/`。
  2. **API 接口**：`/v1/backtrader/summary`。
  3. **收益曲线**：提取 `agent_runs` 表中的 `total_asset` 时间序列即可绘制。

---

## 7. 回测机制详解

**用户：** 回测的话必须确定一个策略才行吧？目前我的系统如果回测的话，会去调用 llm 吗？

**助手：**
- **双轨制回测**：
  1. **纯量化模式**：关闭 AI，只跑 `stock_analyzer.py` 代码逻辑，0 成本，极速。
  2. **AI 增强模式**：开启 LLM，每一天都会调用一次 AI 分析。
- **缓存机制**：如果数据库里已有那一天的 AI 分析记录（`agent_signal_snapshots`），回测时会直接读缓存，不再重复花钱调 API。
- **逻辑一致性**：回测逻辑与实盘挂机完全一致，保证了“所见即所得”。

---

## 8. 修改速查表（我想改 XX，该去哪？）

1. **改 AI 脑子 (Prompt)** ➔ `src/analyzer.py`
2. **改技术指标 (均线/MACD)** ➔ `src/stock_analyzer.py`
3. **改风控规则 (止损/限仓)** ➔ `agent_stock/agents/risk_agent.py`
4. **改游戏规则 (初始资金/手续费)** ➔ `agent_stock/services/backtrader_runtime_service.py`
5. **改执行流程 (串联逻辑)** ➔ `agent_stock/agents/orchestrator.py`
6. **改数据源 (接入新 API)** ➔ `data_provider/`
