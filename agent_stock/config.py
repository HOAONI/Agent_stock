# -*- coding: utf-8 -*-
"""Agent_stock 的统一配置模型与敏感信息处理工具。"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


from dotenv import dotenv_values, load_dotenv

DEFAULT_STOCK_LIST = ["600519", "000001", "300750"]
DEFAULT_REALTIME_SOURCE_PRIORITY = "tencent,akshare_sina,efinance,akshare_em"
ALLOWED_MARKET_SOURCES = ("tencent", "sina", "efinance", "eastmoney", "tushare")
ALLOWED_AGENT_RUN_MODES = {"once", "realtime"}
ALLOWED_AI_REFRESH_POLICIES = {"daily_once", "always"}


def setup_env(override: bool = False) -> None:
    """从配置的 `.env` 文件加载环境变量。"""
    load_dotenv(dotenv_path=_resolve_env_path(), override=override)


def _resolve_env_path() -> Path:
    """解析 `.env` 文件路径，支持通过环境变量覆盖。"""
    env_file = os.getenv("ENV_FILE")
    if env_file:
        return Path(env_file)
    return Path(__file__).resolve().parent.parent / ".env"


_SENSITIVE_KEY_RE = re.compile(
    r"(token|api[_-]?key|secret|password|authorization|credential[_-]?ticket|\bticket\b)",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGN_RE = re.compile(
    r"(?i)\b(token|api[_-]?key|secret|password|authorization|credential[_-]?ticket|ticket)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_\-\.=:+/]{6,})")
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{8,}")


def mask_secret(value: str | None) -> str:
    """对日志和错误中的敏感值进行脱敏。"""
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 6:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def is_valid_secret(value: str | None) -> bool:
    """判断类令牌值是否看起来已配置。"""
    text = str(value or "").strip()
    return bool(text) and len(text) > 10 and not text.startswith("your_")


def infer_openai_compatible_provider(base_url: str | None) -> str:
    """为单个 OpenAI 兼容端点推断提供方标签。"""
    normalized = str(base_url or "").strip().lower()
    if "deepseek" in normalized:
        return "deepseek"
    if normalized and "openai.com" not in normalized:
        return "custom"
    return "openai"


def redact_sensitive_text(text: str | None) -> str:
    """从纯文本中脱敏 token/key/secret 片段。"""
    raw = str(text or "")
    if not raw:
        return raw

    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)}{mask_secret(m.group(2))}", raw)
    redacted = _OPENAI_KEY_RE.sub(lambda m: mask_secret(m.group(0)), redacted)
    redacted = _SENSITIVE_ASSIGN_RE.sub(lambda m: f"{m.group(1)}={mask_secret(m.group(2))}", redacted)
    return redacted


def redact_sensitive_payload(payload: Any) -> Any:
    """递归脱敏载荷类对象中的敏感值。"""
    if payload is None:
        return None
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                result[key] = mask_secret(value if isinstance(value, str) else str(value))
            else:
                result[key] = redact_sensitive_payload(value)
        return result
    if isinstance(payload, list):
        return [redact_sensitive_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_sensitive_payload(item) for item in payload)
    if isinstance(payload, str):
        return redact_sensitive_text(payload)
    return payload


@dataclass(frozen=True)
class RuntimeAccountConfig:
    """请求级账户覆盖配置。"""

    account_name: str
    initial_cash: float | None = None
    account_display_name: str | None = None


@dataclass(frozen=True)
class RuntimeLlmConfig:
    """请求级 LLM 覆盖配置。"""

    provider: str
    base_url: str | None = None
    model: str | None = None
    api_token: str | None = None
    has_token: bool = False


@dataclass(frozen=True)
class RuntimeLlmDefaultConfig:
    """服务级默认 LLM 的解析结果。"""

    provider: str
    base_url: str
    model: str
    has_token: bool = False


@dataclass(frozen=True)
class RuntimeStrategyConfig:
    """请求级策略覆盖配置。"""

    position_max_pct: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass(frozen=True)
class RuntimeExecutionConfig:
    """请求级执行模式覆盖配置。"""

    mode: str = "paper"
    has_ticket: bool = False
    broker_account_id: int | None = None


@dataclass(frozen=True)
class RuntimeDataSourceConfig:
    """请求级行情源覆盖配置。"""

    market_source: str


@dataclass(frozen=True)
class RuntimeContextConfig:
    """请求级上游账户上下文。"""

    account_snapshot: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    positions: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """请求级运行时覆盖配置集合。"""

    account: RuntimeAccountConfig | None = None
    llm: RuntimeLlmConfig | None = None
    strategy: RuntimeStrategyConfig | None = None
    execution: RuntimeExecutionConfig | None = None
    data_source: RuntimeDataSourceConfig | None = None
    context: RuntimeContextConfig | None = None


@dataclass
class Config:
    """应用运行时配置对象。"""

    stock_list: list[str] = field(default_factory=lambda: DEFAULT_STOCK_LIST.copy())

    tushare_token: str | None = None

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_model_fallback: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.7
    gemini_request_delay: float = 2.0
    gemini_max_retries: int = 5
    gemini_retry_delay: float = 5.0

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_temperature: float = 0.7
    anthropic_max_tokens: int = 8192

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    agent_llm_request_timeout_ms: int = 120000

    bocha_api_keys: list[str] = field(default_factory=list)
    tavily_api_keys: list[str] = field(default_factory=list)
    brave_api_keys: list[str] = field(default_factory=list)
    serpapi_keys: list[str] = field(default_factory=list)

    news_max_age_days: int = 3
    bias_threshold: float = 5.0

    database_path: str = "./data/stock_analysis.db"
    database_url: str | None = None
    save_context_snapshot: bool = True

    agent_run_mode: str = "once"
    agent_poll_interval_minutes: int = 5
    agent_market_timezone: str = "Asia/Shanghai"
    agent_market_sessions: str = "09:30-11:30,13:00-15:00"
    agent_ai_refresh_policy: str = "daily_once"
    agent_account_name: str = "paper-default"
    agent_initial_cash: float = 1_000_000.0
    agent_max_single_position_pct: float = 0.50
    agent_max_total_exposure_pct: float = 0.90
    agent_fee_rate: float = 0.0003
    agent_sell_tax_rate: float = 0.001
    agent_slippage_bps: float = 5.0
    agent_min_trade_lot: int = 100
    agent_weight_strong_buy: float = 0.50
    agent_weight_buy: float = 0.30
    agent_weight_hold: float = 0.20
    agent_weight_wait: float = 0.00
    agent_weight_sell: float = 0.00

    agent_service_mode: bool = False
    agent_service_host: str = "0.0.0.0"
    agent_service_port: int = 8001
    agent_service_auth_token: str | None = None
    agent_task_max_workers: int = 3
    agent_write_local_reports: bool = False

    log_dir: str = "./logs"
    log_level: str = "INFO"

    enable_realtime_quote: bool = True
    enable_chip_distribution: bool = True
    enable_eastmoney_patch: bool = False
    realtime_source_priority: str = DEFAULT_REALTIME_SOURCE_PRIORITY

    _instance: ClassVar[Config | None] = None

    @classmethod
    def get_instance(cls) -> Config:
        """返回配置单例。"""
        if cls._instance is None:
            cls._instance = cls._load_from_env()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置配置单例，供测试或重载使用。"""
        cls._instance = None

    @classmethod
    def _load_from_env(cls) -> Config:
        """从环境变量和 `.env` 文件加载配置。"""
        setup_env()
        cls._apply_proxy_settings()

        return cls(
            stock_list=cls._parse_stock_list(os.getenv("STOCK_LIST", "")),
            tushare_token=os.getenv("TUSHARE_TOKEN") or None,
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            gemini_model_fallback=os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash"),
            gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            gemini_request_delay=float(os.getenv("GEMINI_REQUEST_DELAY", "2.0")),
            gemini_max_retries=max(1, int(os.getenv("GEMINI_MAX_RETRIES", "5"))),
            gemini_retry_delay=float(os.getenv("GEMINI_RETRY_DELAY", "5.0")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            anthropic_temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
            anthropic_max_tokens=max(1, int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            agent_llm_request_timeout_ms=max(1000, int(os.getenv("AGENT_LLM_REQUEST_TIMEOUT_MS", "120000"))),
            bocha_api_keys=cls._split_secret_csv(os.getenv("BOCHA_API_KEYS", "")),
            tavily_api_keys=cls._split_secret_csv(os.getenv("TAVILY_API_KEYS", "")),
            brave_api_keys=cls._split_secret_csv(os.getenv("BRAVE_API_KEYS", "")),
            serpapi_keys=cls._split_secret_csv(os.getenv("SERPAPI_API_KEYS", "")),
            news_max_age_days=max(1, int(os.getenv("NEWS_MAX_AGE_DAYS", "3"))),
            bias_threshold=max(1.0, float(os.getenv("BIAS_THRESHOLD", "5.0"))),
            database_path=os.getenv("DATABASE_PATH", "./data/stock_analysis.db"),
            database_url=os.getenv("DATABASE_URL") or None,
            save_context_snapshot=cls._env_flag("SAVE_CONTEXT_SNAPSHOT", True),
            agent_run_mode=os.getenv("AGENT_RUN_MODE", "once").strip().lower(),
            agent_poll_interval_minutes=max(1, int(os.getenv("AGENT_POLL_INTERVAL_MINUTES", "5"))),
            agent_market_timezone=os.getenv("AGENT_MARKET_TIMEZONE", "Asia/Shanghai"),
            agent_market_sessions=os.getenv("AGENT_MARKET_SESSIONS", "09:30-11:30,13:00-15:00"),
            agent_ai_refresh_policy=os.getenv("AGENT_AI_REFRESH_POLICY", "daily_once").strip().lower(),
            agent_account_name=os.getenv("AGENT_ACCOUNT_NAME", "paper-default"),
            agent_initial_cash=float(os.getenv("AGENT_INITIAL_CASH", "1000000")),
            agent_max_single_position_pct=max(0.0, float(os.getenv("AGENT_MAX_SINGLE_POSITION_PCT", "0.5"))),
            agent_max_total_exposure_pct=max(0.0, float(os.getenv("AGENT_MAX_TOTAL_EXPOSURE_PCT", "0.9"))),
            agent_fee_rate=max(0.0, float(os.getenv("AGENT_FEE_RATE", "0.0003"))),
            agent_sell_tax_rate=max(0.0, float(os.getenv("AGENT_SELL_TAX_RATE", "0.001"))),
            agent_slippage_bps=max(0.0, float(os.getenv("AGENT_SLIPPAGE_BPS", "5"))),
            agent_min_trade_lot=max(1, int(os.getenv("AGENT_MIN_TRADE_LOT", "100"))),
            agent_weight_strong_buy=max(0.0, float(os.getenv("AGENT_WEIGHT_STRONG_BUY", "0.5"))),
            agent_weight_buy=max(0.0, float(os.getenv("AGENT_WEIGHT_BUY", "0.3"))),
            agent_weight_hold=max(0.0, float(os.getenv("AGENT_WEIGHT_HOLD", "0.2"))),
            agent_weight_wait=max(0.0, float(os.getenv("AGENT_WEIGHT_WAIT", "0.0"))),
            agent_weight_sell=max(0.0, float(os.getenv("AGENT_WEIGHT_SELL", "0.0"))),
            agent_service_mode=cls._env_flag("AGENT_SERVICE_MODE", False),
            agent_service_host=os.getenv("AGENT_SERVICE_HOST", "0.0.0.0"),
            agent_service_port=max(1, int(os.getenv("AGENT_SERVICE_PORT", "8001"))),
            agent_service_auth_token=os.getenv("AGENT_SERVICE_AUTH_TOKEN") or None,
            agent_task_max_workers=max(1, int(os.getenv("AGENT_TASK_MAX_WORKERS", "3"))),
            agent_write_local_reports=cls._env_flag("AGENT_WRITE_LOCAL_REPORTS", False),
            log_dir=os.getenv("LOG_DIR", "./logs"),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
            enable_realtime_quote=cls._env_flag("ENABLE_REALTIME_QUOTE", True),
            enable_chip_distribution=cls._env_flag("ENABLE_CHIP_DISTRIBUTION", True),
            enable_eastmoney_patch=cls._env_flag("ENABLE_EASTMONEY_PATCH", False),
            realtime_source_priority=cls._resolve_realtime_source_priority(),
        )

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        """读取布尔型环境变量。"""
        return os.getenv(name, str(default).lower()).strip().lower() == "true"

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        """按逗号拆分并清理字符串列表。"""
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    @staticmethod
    def _split_secret_csv(value: str) -> list[str]:
        """按逗号拆分敏感令牌列表，并过滤空值与占位符。"""
        return [item for item in Config._split_csv(value) if is_valid_secret(item)]

    @classmethod
    def _parse_stock_list(cls, value: str) -> list[str]:
        """解析股票代码列表，缺省时回退到默认值。"""
        stock_list = [item.upper() for item in cls._split_csv(value)]
        return stock_list or DEFAULT_STOCK_LIST.copy()

    @classmethod
    def _apply_proxy_settings(cls) -> None:
        """应用代理配置，并为国内数据源补充 no_proxy 白名单。"""
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        if not http_proxy:
            return

        domestic_domains = [
            "eastmoney.com",
            "sina.com.cn",
            "163.com",
            "tushare.pro",
            "baostock.com",
            "sse.com.cn",
            "szse.cn",
            "csindex.com.cn",
            "cninfo.com.cn",
            "localhost",
            "127.0.0.1",
        ]
        current_no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
        existing_domains = [item.strip() for item in current_no_proxy.split(",") if item.strip()]
        final_domains = sorted(set(existing_domains + domestic_domains))
        final_no_proxy = ",".join(final_domains)

        os.environ["NO_PROXY"] = final_no_proxy
        os.environ["no_proxy"] = final_no_proxy
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["http_proxy"] = http_proxy

        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["https_proxy"] = https_proxy

    @classmethod
    def _resolve_realtime_source_priority(cls) -> str:
        """解析实时行情数据源优先级。"""
        explicit = os.getenv("REALTIME_SOURCE_PRIORITY")
        if explicit:
            return explicit
        if str(os.getenv("TUSHARE_TOKEN") or "").strip():
            return f"tushare,{DEFAULT_REALTIME_SOURCE_PRIORITY}"
        return DEFAULT_REALTIME_SOURCE_PRIORITY

    def clone_with_overrides(self, **overrides: Any) -> Config:
        """构建带请求级覆盖项的独立配置对象。"""
        payload: dict[str, Any] = {}
        for item in fields(self):
            payload[item.name] = copy.deepcopy(getattr(self, item.name))
        payload.update(overrides)
        return Config(**payload)

    def clone_for_runtime_llm(self, runtime_llm: RuntimeLlmConfig | None) -> Config:
        """为单个运行时 LLM 覆盖项构建请求级配置视图。"""
        if runtime_llm is None:
            return self

        provider = str(runtime_llm.provider or "").strip().lower()
        base_url = str(runtime_llm.base_url or "").strip() or None
        model = str(runtime_llm.model or "").strip()
        api_token = str(runtime_llm.api_token or "").strip() or None

        overrides: dict[str, Any] = {
            "gemini_api_key": None,
            "anthropic_api_key": None,
            "openai_api_key": None,
        }

        if provider == "gemini":
            overrides.update(
                {
                    "gemini_api_key": api_token or self.gemini_api_key,
                    "gemini_model": model or self.gemini_model,
                }
            )
        elif provider == "anthropic":
            overrides.update(
                {
                    "anthropic_api_key": api_token or self.anthropic_api_key,
                    "anthropic_model": model or self.anthropic_model,
                }
            )
        elif provider in {"openai", "deepseek", "custom"}:
            overrides.update(
                {
                    "openai_api_key": api_token or self.openai_api_key,
                    "openai_base_url": base_url or self.openai_base_url,
                    "openai_model": model or self.openai_model,
                }
            )
        else:
            return self

        return self.clone_with_overrides(**overrides)

    def resolve_default_runtime_llm(self) -> RuntimeLlmDefaultConfig | None:
        """解析当前生效的内置默认 LLM 元数据。"""
        if is_valid_secret(self.gemini_api_key):
            return RuntimeLlmDefaultConfig(
                provider="gemini",
                base_url="https://generativelanguage.googleapis.com",
                model=self.gemini_model or "gemini-3-flash-preview",
                has_token=True,
            )

        if is_valid_secret(self.anthropic_api_key):
            return RuntimeLlmDefaultConfig(
                provider="anthropic",
                base_url="https://api.anthropic.com",
                model=self.anthropic_model or "claude-3-5-sonnet-20241022",
                has_token=True,
            )

        if is_valid_secret(self.openai_api_key):
            base_url = str(self.openai_base_url or "").strip() or "https://api.openai.com/v1"
            return RuntimeLlmDefaultConfig(
                provider=infer_openai_compatible_provider(base_url),
                base_url=base_url,
                model=self.openai_model or "gpt-4o-mini",
                has_token=True,
            )

        return None

    def refresh_stock_list(self) -> None:
        """从最新的 `.env` 或进程环境刷新 `STOCK_LIST`。"""
        stock_list_raw = ""
        env_path = _resolve_env_path()
        if env_path.exists():
            env_values = dotenv_values(env_path)
            stock_list_raw = str(env_values.get("STOCK_LIST") or "").strip()
        if not stock_list_raw:
            stock_list_raw = os.getenv("STOCK_LIST", "")
        self.stock_list = self._parse_stock_list(stock_list_raw)

    def validate(self) -> list[str]:
        """返回非致命配置告警。"""
        warnings: list[str] = []

        if not self.stock_list:
            warnings.append("Missing STOCK_LIST configuration.")
        if not self.tushare_token:
            warnings.append("TUSHARE_TOKEN is not configured; fallback data sources will be used.")
        if not self.gemini_api_key and not self.anthropic_api_key and not self.openai_api_key:
            warnings.append("No Gemini/Anthropic/OpenAI API key is configured.")
        elif not self.gemini_api_key and not self.anthropic_api_key:
            warnings.append("Gemini/Anthropic are not configured; OpenAI-compatible fallback will be used.")
        if not self.bocha_api_keys and not self.tavily_api_keys and not self.brave_api_keys and not self.serpapi_keys:
            warnings.append("No valid search API key is configured; public news fallback will be used.")
        if self.agent_run_mode not in ALLOWED_AGENT_RUN_MODES:
            warnings.append("AGENT_RUN_MODE must be once or realtime.")
        if self.agent_ai_refresh_policy not in ALLOWED_AI_REFRESH_POLICIES:
            warnings.append("AGENT_AI_REFRESH_POLICY must be daily_once or always.")
        if self.agent_max_single_position_pct > 1.0:
            warnings.append("AGENT_MAX_SINGLE_POSITION_PCT should stay within 0..1.")
        if self.agent_max_total_exposure_pct > 1.0:
            warnings.append("AGENT_MAX_TOTAL_EXPOSURE_PCT should stay within 0..1.")
        if self.agent_service_mode and not self.database_url:
            warnings.append("DATABASE_URL is required when AGENT_SERVICE_MODE=true.")
        if self.agent_service_mode and not self.agent_service_auth_token:
            warnings.append("AGENT_SERVICE_AUTH_TOKEN is required when AGENT_SERVICE_MODE=true.")
        return warnings

    def validate_service_requirements(self) -> None:
        """校验服务模式所需配置。"""
        missing = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        if not self.agent_service_auth_token:
            missing.append("AGENT_SERVICE_AUTH_TOKEN")
        if missing:
            raise ValueError(f"Missing required service configuration: {', '.join(missing)}")

    def get_db_url(self) -> str:
        """返回当前生效的 SQLAlchemy 数据库 URL。"""
        if self.database_url:
            return self.database_url
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.absolute()}"


def get_config() -> Config:
    """返回全局配置单例。"""
    return Config.get_instance()
