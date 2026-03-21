# -*- coding: utf-8 -*-
"""数据源抽象层与调度器。

这里定义了所有行情抓取器的统一接口，并由 `DataFetcherManager` 负责做优先级调度、
失败切换、实时行情补齐和预取优化。排查“为什么这次走了某个数据源”时，优先看
这个文件。
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 配置日志
logger = logging.getLogger(__name__)


# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
FIXED_MARKET_SOURCES = ('tencent', 'sina', 'efinance', 'eastmoney', 'tushare')


def normalize_stock_code(stock_code: str) -> str:
    """
    规范化股票代码，去掉交易所前后缀。

    支持的输入与输出示例：
    - '600519' -> '600519'（本身已规范）
    - 'SH600519' -> '600519'（去掉 SH 前缀）
    - 'SZ000001' -> '000001'（去掉 SZ 前缀）
    - 'sh600519' -> '600519'（大小写不敏感）
    - '600519.SH' -> '600519'（去掉 .SH 后缀）
    - '000001.SZ' -> '000001'（去掉 .SZ 后缀）
    - 'HK00700' -> 'HK00700'（港股保留 HK 前缀）
    - 'AAPL' -> 'AAPL'（美股代码保持不变）

    该函数在 `DataFetcherManager` 层统一调用，确保各个 Fetcher
    接收到的 A 股/ETF 代码是干净的 5 位或 6 位数字格式。
    """
    code = stock_code.strip()
    upper = code.upper()

    # 去掉 SH/SZ 前缀（例如 SH600519 -> 600519）
    if upper.startswith(('SH', 'SZ')) and not upper.startswith('SH.') and not upper.startswith('SZ.'):
        candidate = code[2:]
        # 仅当剩余部分看起来是有效数字代码时才去掉前缀
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # 去掉 .SH/.SZ 后缀（例如 600519.SH -> 600519）
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix.upper() in ('SH', 'SZ', 'SS') and base.isdigit():
            return base

    return code


def canonical_stock_code(code: str) -> str:
    """
    返回股票代码的规范展示形式（统一大写）。

    这与 `normalize_stock_code` 不同：`normalize_stock_code` 负责去掉
    交易所前后缀，而本函数用于在展示和存储层统一大小写，确保
    Bot、Web UI、API 和 CLI 等入口看到的代码格式一致（Issue #355）。

    示例：
        'aapl' -> 'AAPL'
        'AAPL' -> 'AAPL'
        '600519' -> '600519'（纯数字保持不变）
        'hk00700' -> 'HK00700'
    """
    return (code or "").strip().upper()


class DataFetchError(Exception):
    """数据获取异常基类。"""
    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常。"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常。"""
    pass


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 实现通用的技术指标计算
    
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        参数：
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        返回：
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）

        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数实时行情

        参数：
            region: 市场区域，cn=A股 us=美股

        返回：
            List[Dict]: 指数列表，每个元素为字典，包含:
                - code: 指数代码
                - name: 指数名称
                - current: 当前点位
                - change: 涨跌点数
                - change_pct: 涨跌幅(%)
                - volume: 成交量
                - amount: 成交额
        """
        return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计

        返回：
            Dict: 包含:
                - up_count: 上涨家数
                - down_count: 下跌家数
                - flat_count: 平盘家数
                - limit_up_count: 涨停家数
                - limit_down_count: 跌停家数
                - total_amount: 两市成交额
        """
        return None

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        获取板块涨跌榜

        参数：
            n: 返回前n个

        返回：
            Tuple: (领涨板块列表, 领跌板块列表)
        """
        return None

    def get_daily_data(
        self,
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 计算技术指标
        
        参数：
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）
            
        返回：
            标准化的 DataFrame，包含技术指标
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")
        
        try:
            # 第 1 步：从具体数据源拉取原始行情。
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # 第 2 步：标准化列名
            df = self._normalize_data(raw_df, stock_code)
            
            # 第 3 步：数据清洗
            df = self._clean_data(df)
            
            # 第 4 步：补齐均线、量比等下游分析依赖的技术指标。
            df = self._calculate_indicators(df)
            
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        计算指标：
        - MA5, MA10, MA20: 移动平均线
        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）
        """
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 量比：当日成交量 / 5日平均成交量
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # 保留2位小数
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    数据源策略管理器
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    
    切换策略：
    - 优先使用高优先级数据源
    - 失败后自动切换到下一个
    - 所有数据源都失败时抛出异常
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        参数：
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
    
    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表

        优先级动态调整逻辑：
        - 如果配置了 TUSHARE_TOKEN：Tushare 优先级提升为 0（最高）
        - 否则按默认优先级：
          0. EfinanceFetcher（优先级 0） - 最高优先级
          1. AkshareFetcher（优先级 1）
          2. PytdxFetcher（优先级 2） - 通达信
          2. TushareFetcher（优先级 2）
          3. BaostockFetcher（优先级 3）
          4. YfinanceFetcher（优先级 4）
        """
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher
        from agent_stock.config import get_config

        config = get_config()

        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）
        efinance = EfinanceFetcher()
        akshare = AkshareFetcher()
        tushare = TushareFetcher()  # 会根据 Token 配置自动调整优先级
        pytdx = PytdxFetcher()      # 通达信数据源（可配 PYTDX_HOST/PYTDX_PORT）
        baostock = BaostockFetcher()
        yfinance = YfinanceFetcher()

        # 初始化数据源列表
        self._fetchers = [
            efinance,
            akshare,
            tushare,
            pytdx,
            baostock,
            yfinance,
        ]

        # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 0）
        self._fetchers.sort(key=lambda f: f.priority)

        # 构建优先级说明
        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")
    
    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)

    @staticmethod
    def _normalize_market_source(source: str) -> str:
        normalized = str(source or "").strip().lower()
        if normalized not in FIXED_MARKET_SOURCES:
            allowed = ", ".join(FIXED_MARKET_SOURCES)
            raise DataSourceUnavailableError(f"unsupported market source: {normalized or '<empty>'}. allowed={allowed}")
        return normalized

    def _find_fetcher(self, fetcher_name: str) -> Optional[BaseFetcher]:
        for fetcher in self._fetchers:
            if fetcher.name == fetcher_name:
                return fetcher
        return None

    def _get_daily_data_fixed_source(
        self,
        stock_code: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
        fixed_source: str,
    ) -> Tuple[pd.DataFrame, str]:
        source = self._normalize_market_source(fixed_source)

        if source == "efinance":
            fetcher = self._find_fetcher("EfinanceFetcher")
            if fetcher is None:
                raise DataSourceUnavailableError("efinance fetcher is not available")
            try:
                df = fetcher.get_daily_data(stock_code=stock_code, start_date=start_date, end_date=end_date, days=days)
            except Exception as exc:
                raise DataSourceUnavailableError(f"efinance daily data failed: {exc}") from exc
        elif source == "tushare":
            from agent_stock.config import get_config

            if not str(get_config().tushare_token or "").strip():
                raise DataSourceUnavailableError("tushare is unavailable: TUSHARE_TOKEN is not configured")
            fetcher = self._find_fetcher("TushareFetcher")
            if fetcher is None:
                raise DataSourceUnavailableError("tushare fetcher is not available")
            try:
                df = fetcher.get_daily_data(stock_code=stock_code, start_date=start_date, end_date=end_date, days=days)
            except Exception as exc:
                raise DataSourceUnavailableError(f"tushare daily data failed: {exc}") from exc
        else:
            fetcher = self._find_fetcher("AkshareFetcher")
            if fetcher is None or not hasattr(fetcher, "get_daily_data_by_source"):
                raise DataSourceUnavailableError("akshare fixed-source daily data is not available")
            sub_source = "em" if source == "eastmoney" else source
            try:
                df = fetcher.get_daily_data_by_source(
                    stock_code=stock_code,
                    source=sub_source,
                    start_date=start_date,
                    end_date=end_date,
                    days=days,
                )
            except Exception as exc:
                raise DataSourceUnavailableError(f"{source} daily data failed: {exc}") from exc

        if df is None or df.empty:
            raise DataSourceUnavailableError(f"{source} daily data returned no rows for {stock_code}")
        return df, source
    
    def get_daily_data(
        self, 
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
        fixed_source: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源）
        
        故障切换策略：
        1. 美股指数/美股股票直接路由到 YfinanceFetcher
        2. 其他代码从最高优先级数据源开始尝试
        3. 捕获异常后自动切换到下一个
        4. 记录每个数据源的失败原因
        5. 所有数据源失败后抛出详细异常
        
        参数：
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            
        返回：
            Tuple[DataFrame, str]: (数据, 成功的数据源名称)
            
        异常：
            DataFetchError: 所有数据源都失败时抛出
        """
        from .us_index_mapping import is_us_index_code, is_us_stock_code

        # 规范化代码（去掉 SH/SZ 前缀等）
        stock_code = normalize_stock_code(stock_code)

        if fixed_source:
            return self._get_daily_data_fixed_source(
                stock_code,
                start_date=start_date,
                end_date=end_date,
                days=days,
                fixed_source=fixed_source,
            )

        errors = []

        # 快速路径：美股指数与美股股票直接路由到 YfinanceFetcher
        if is_us_index_code(stock_code) or is_us_stock_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    try:
                        logger.info(f"[{fetcher.name}] 美股/美股指数 {stock_code} 直接路由...")
                        df = fetcher.get_daily_data(
                            stock_code=stock_code,
                            start_date=start_date,
                            end_date=end_date,
                            days=days,
                        )
                        if df is not None and not df.empty:
                            logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                            return df, fetcher.name
                    except Exception as e:
                        error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                    break
            # YfinanceFetcher 不存在或初始化失败时，统一走美股错误汇总逻辑。
            error_summary = f"美股/美股指数 {stock_code} 获取失败:\n" + "\n".join(errors)
            logger.error(error_summary)
            raise DataFetchError(error_summary)

        for fetcher in self._fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )
                
                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name
                    
            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # 继续尝试下一个数据源
                continue
        
        # 所有数据源都失败
        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)
    
    @property
    def available_fetchers(self) -> List[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]
    
    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        批量预取实时行情数据（在分析开始前调用）
        
        策略：
        1. 检查优先级中是否包含全量拉取数据源（efinance/akshare_em）
        2. 如果不包含，跳过预取（新浪/腾讯是单股票查询，无需预取）
        3. 如果自选股数量 >= 5 且使用全量数据源，则预取填充缓存
        
        这样做的好处：
        - 使用新浪/腾讯时：每只股票独立查询，无全量拉取问题
        - 使用 efinance/东财时：预取一次，后续缓存命中
        
        参数：
            stock_codes: 待分析的股票代码列表
            
        返回：
            预取的股票数量（0 表示跳过预取）
        """
        # 统一规范化所有代码
        stock_codes = [normalize_stock_code(c) for c in stock_codes]

        from agent_stock.config import get_config
        
        config = get_config()
        
        # 如果实时行情被禁用，跳过预取
        if not config.enable_realtime_quote:
            logger.debug("[预取] 实时行情功能已禁用，跳过预取")
            return 0
        
        # 检查优先级中是否包含全量拉取数据源
        # 注意：新增全量接口（如 tushare_realtime）时需同步更新此列表
        # 全量接口特征：一次 API 调用拉取全市场 5000+ 股票数据
        priority = config.realtime_source_priority.lower()
        bulk_sources = ['efinance', 'akshare_em', 'tushare']  # 全量接口列表
        
        # 如果优先级中前两个都不是全量数据源，跳过预取
        # 因为新浪/腾讯是单股票查询，不需要预取
        priority_list = [s.strip() for s in priority.split(',')]
        first_bulk_source_index = None
        for i, source in enumerate(priority_list):
            if source in bulk_sources:
                first_bulk_source_index = i
                break
        
        # 如果没有全量数据源，或者全量数据源排在第 3 位之后，跳过预取
        if first_bulk_source_index is None or first_bulk_source_index >= 2:
            logger.info(f"[预取] 当前优先级使用轻量级数据源(sina/tencent)，无需预取")
            return 0
        
        # 如果股票数量少于 5 个，不进行批量预取（逐个查询更高效）
        if len(stock_codes) < 5:
            logger.info(f"[预取] 股票数量 {len(stock_codes)} < 5，跳过批量预取")
            return 0
        
        logger.info(f"[预取] 开始批量预取实时行情，共 {len(stock_codes)} 只股票...")
        
        # 尝试通过 efinance 或 akshare 预取
        # 只需要调用一次 get_realtime_quote，缓存机制会自动拉取全市场数据
        try:
            # 用第一只股票触发全量拉取
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)
            
            if quote:
                logger.info(f"[预取] 批量预取完成，缓存已填充")
                return len(stock_codes)
            else:
                logger.warning(f"[预取] 批量预取失败，将使用逐个查询模式")
                return 0
                
        except Exception as e:
            logger.error(f"[预取] 批量预取异常: {e}")
            return 0

    def _get_realtime_quote_fixed_source(self, stock_code: str, *, fixed_source: str):
        source = self._normalize_market_source(fixed_source)

        if source == "efinance":
            fetcher = self._find_fetcher("EfinanceFetcher")
            if fetcher is None or not hasattr(fetcher, "get_realtime_quote"):
                raise DataSourceUnavailableError("efinance realtime fetcher is not available")
            try:
                quote = fetcher.get_realtime_quote(stock_code)
            except Exception as exc:
                raise DataSourceUnavailableError(f"efinance realtime quote failed: {exc}") from exc
        elif source == "tushare":
            from agent_stock.config import get_config

            if not str(get_config().tushare_token or "").strip():
                raise DataSourceUnavailableError("tushare is unavailable: TUSHARE_TOKEN is not configured")
            fetcher = self._find_fetcher("TushareFetcher")
            if fetcher is None or not hasattr(fetcher, "get_realtime_quote"):
                raise DataSourceUnavailableError("tushare realtime fetcher is not available")
            try:
                quote = fetcher.get_realtime_quote(stock_code)
            except Exception as exc:
                raise DataSourceUnavailableError(f"tushare realtime quote failed: {exc}") from exc
        else:
            fetcher = self._find_fetcher("AkshareFetcher")
            if fetcher is None or not hasattr(fetcher, "get_realtime_quote"):
                raise DataSourceUnavailableError("akshare realtime fetcher is not available")
            sub_source = "em" if source == "eastmoney" else source
            try:
                quote = fetcher.get_realtime_quote(stock_code, source=sub_source)
            except Exception as exc:
                raise DataSourceUnavailableError(f"{source} realtime quote failed: {exc}") from exc

        if quote is None or not quote.has_basic_data():
            raise DataSourceUnavailableError(f"{source} realtime quote returned no usable data for {stock_code}")
        return quote

    def get_realtime_quote(self, stock_code: str, fixed_source: Optional[str] = None):
        """
        获取实时行情数据（自动故障切换）
        
        故障切换策略（按配置的优先级）：
        1. 美股：使用 YfinanceFetcher.get_realtime_quote()
        2. EfinanceFetcher.get_realtime_quote()
        3. AkshareFetcher.get_realtime_quote(source="em") - 东财
        4. AkshareFetcher.get_realtime_quote(source="sina") - 新浪
        5. AkshareFetcher.get_realtime_quote(source="tencent") - 腾讯
        6. 返回 None（降级兜底）
        
        参数：
            stock_code: 股票代码
            
        返回：
            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None
        """
        # 规范化代码（去掉 SH/SZ 前缀等）
        stock_code = normalize_stock_code(stock_code)

        from .realtime_types import get_realtime_circuit_breaker
        from .akshare_fetcher import _is_us_code
        from .us_index_mapping import is_us_index_code
        from agent_stock.config import get_config

        config = get_config()

        if fixed_source:
            return self._get_realtime_quote_fixed_source(stock_code, fixed_source=fixed_source)

        # 如果实时行情功能被禁用，直接返回 None
        if not config.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None

        # 美股指数由 YfinanceFetcher 处理（在美股股票检查之前）
        if is_us_index_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    if hasattr(fetcher, 'get_realtime_quote'):
                        try:
                            quote = fetcher.get_realtime_quote(stock_code)
                            if quote is not None:
                                logger.info(f"[实时行情] 美股指数 {stock_code} 成功获取 (来源: yfinance)")
                                return quote
                        except Exception as e:
                            logger.warning(f"[实时行情] 美股指数 {stock_code} 获取失败: {e}")
                    break
            logger.warning(f"[实时行情] 美股指数 {stock_code} 无可用数据源")
            return None

        # 美股单独处理，使用 YfinanceFetcher
        if _is_us_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    if hasattr(fetcher, 'get_realtime_quote'):
                        try:
                            quote = fetcher.get_realtime_quote(stock_code)
                            if quote is not None:
                                logger.info(f"[实时行情] 美股 {stock_code} 成功获取 (来源: yfinance)")
                                return quote
                        except Exception as e:
                            logger.warning(f"[实时行情] 美股 {stock_code} 获取失败: {e}")
                    break
            logger.warning(f"[实时行情] 美股 {stock_code} 无可用数据源")
            return None
        
        # 获取配置的数据源优先级
        source_priority = config.realtime_source_priority.split(',')
        
        errors = []
        # `primary_quote` 保存首个成功结果；后续可以补齐
        # 后续数据源返回的缺失字段（如 `volume_ratio`、`turnover_rate` 等）。
        primary_quote = None
        
        for source in source_priority:
            source = source.strip().lower()
            
            try:
                quote = None
                
                if source == "efinance":
                    # 尝试 EfinanceFetcher
                    for fetcher in self._fetchers:
                        if fetcher.name == "EfinanceFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code)
                            break
                
                elif source == "akshare_em":
                    # 尝试 AkshareFetcher 东财数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="em")
                            break
                
                elif source == "akshare_sina":
                    # 尝试 AkshareFetcher 新浪数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="sina")
                            break
                
                elif source in ("tencent", "akshare_qq"):
                    # 尝试 AkshareFetcher 腾讯数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="tencent")
                            break
                
                elif source == "tushare":
                    # 尝试 TushareFetcher（需要 Tushare Pro 积分）
                    for fetcher in self._fetchers:
                        if fetcher.name == "TushareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code)
                            break
                
                if quote is not None and quote.has_basic_data():
                    if primary_quote is None:
                        # 首个成功的数据源作为主结果
                        primary_quote = quote
                        logger.info(f"[实时行情] {stock_code} 成功获取 (来源: {source})")
                        # 若关键补充字段已齐全，则提前返回
                        if not self._quote_needs_supplement(primary_quote):
                            return primary_quote
                        # 否则继续尝试后续数据源补齐缺失字段
                        logger.debug(f"[实时行情] {stock_code} 部分字段缺失，尝试从后续数据源补充")
                        supplement_attempts = 0
                    else:
                        # 从当前数据源补齐缺失字段（限制尝试次数）
                        supplement_attempts += 1
                        if supplement_attempts > 1:
                            logger.debug(f"[实时行情] {stock_code} 补充尝试已达上限，停止继续")
                            break
                        merged = self._merge_quote_fields(primary_quote, quote)
                        if merged:
                            logger.info(f"[实时行情] {stock_code} 从 {source} 补充了缺失字段: {merged}")
                        # 关键字段全部补齐后停止继续补充
                        if not self._quote_needs_supplement(primary_quote):
                            break
                    
            except Exception as e:
                error_msg = f"[{source}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # 即使仍有少数字段缺失，也返回主结果
        if primary_quote is not None:
            return primary_quote

        # 所有数据源都失败，返回 None（降级兜底）
        if errors:
            logger.warning(f"[实时行情] {stock_code} 所有数据源均失败，降级处理: {'; '.join(errors)}")
        else:
            logger.warning(f"[实时行情] {stock_code} 无可用数据源")
        
        return None

    # 当主数据源未返回时，可从次级数据源补齐的字段
    # 按重要性排序，仅补齐值为 `None` 的字段。
    _SUPPLEMENT_FIELDS = [
        'volume_ratio', 'turnover_rate',
        'pe_ratio', 'pb_ratio', 'total_mv', 'circ_mv',
        'amplitude',
    ]

    @classmethod
    def _quote_needs_supplement(cls, quote) -> bool:
        """检查关键补充字段中是否仍有 `None`。"""
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(quote, f, None) is None:
                return True
        return False

    @classmethod
    def _merge_quote_fields(cls, primary, secondary) -> list:
        """
        将 `secondary` 中非 `None` 的字段补到 `primary` 上。

        仅当 `primary` 对应字段为 `None` 时才会覆盖，返回本次成功补齐的字段名列表。
        """
        filled = []
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(primary, f, None) is None:
                val = getattr(secondary, f, None)
                if val is not None:
                    setattr(primary, f, val)
                    filled.append(f)
        return filled

    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据（带熔断和多数据源降级）

        策略：
        1. 检查配置开关
        2. 检查熔断器状态
        3. 依次尝试多个数据源：AkshareFetcher -> TushareFetcher -> EfinanceFetcher
        4. 所有数据源失败则返回 None（降级兜底）

        参数：
            stock_code: 股票代码

        返回：
            ChipDistribution 对象，失败则返回 None
        """
        # 规范化代码（去掉 SH/SZ 前缀等）
        stock_code = normalize_stock_code(stock_code)

        from .realtime_types import get_chip_circuit_breaker
        from agent_stock.config import get_config

        config = get_config()

        # 如果筹码分布功能被禁用，直接返回 None
        if not config.enable_chip_distribution:
            logger.debug(f"[筹码分布] 功能已禁用，跳过 {stock_code}")
            return None

        circuit_breaker = get_chip_circuit_breaker()

        # 定义筹码数据源优先级列表
        chip_sources = [
            ("AkshareFetcher", "akshare_chip"),
            ("TushareFetcher", "tushare_chip"),
            ("EfinanceFetcher", "efinance_chip"),
        ]

        for fetcher_name, source_key in chip_sources:
            # 检查熔断器状态
            if not circuit_breaker.is_available(source_key):
                logger.debug(f"[熔断] {fetcher_name} 筹码接口处于熔断状态，尝试下一个")
                continue

            try:
                for fetcher in self._fetchers:
                    if fetcher.name == fetcher_name:
                        if hasattr(fetcher, 'get_chip_distribution'):
                            chip = fetcher.get_chip_distribution(stock_code)
                            if chip is not None:
                                circuit_breaker.record_success(source_key)
                                logger.info(f"[筹码分布] {stock_code} 成功获取 (来源: {fetcher_name})")
                                return chip
                        break
            except Exception as e:
                logger.warning(f"[筹码分布] {fetcher_name} 获取 {stock_code} 失败: {e}")
                circuit_breaker.record_failure(source_key, str(e))
                continue

        logger.warning(f"[筹码分布] {stock_code} 所有数据源均失败")
        return None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票中文名称（自动切换数据源）
        
        尝试从多个数据源获取股票名称：
        1. 先从实时行情缓存中获取（如果有）
        2. 依次尝试各个数据源的 get_stock_name 方法
        3. 最后尝试让大模型通过搜索获取（需要外部调用）
        
        参数：
            stock_code: 股票代码
            
        返回：
            股票中文名称，所有数据源都失败则返回 None
        """
        # 规范化代码（去掉 SH/SZ 前缀等）
        stock_code = normalize_stock_code(stock_code)

        # 1. 先检查缓存
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        
        # 初始化缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        # 2. 尝试从实时行情中获取（最快）
        quote = self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, 'name') and quote.name:
            name = quote.name
            self._stock_name_cache[stock_code] = name
            logger.info(f"[股票名称] 从实时行情获取: {stock_code} -> {name}")
            return name
        
        # 3. 依次尝试各个数据源
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_name'):
                try:
                    name = fetcher.get_stock_name(stock_code)
                    if name:
                        self._stock_name_cache[stock_code] = name
                        logger.info(f"[股票名称] 从 {fetcher.name} 获取: {stock_code} -> {name}")
                        return name
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 获取失败: {e}")
                    continue
        
        # 4. 所有数据源都失败
        logger.warning(f"[股票名称] 所有数据源都无法获取 {stock_code} 的名称")
        return None

    def batch_get_stock_names(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        批量获取股票中文名称
        
        先尝试从支持批量查询的数据源获取股票列表，
        然后再逐个查询缺失的股票名称。
        
        参数：
            stock_codes: 股票代码列表
            
        返回：
            {股票代码: 股票名称} 字典
        """
        result = {}
        missing_codes = set(stock_codes)
        
        # 1. 先检查缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        for code in stock_codes:
            if code in self._stock_name_cache:
                result[code] = self._stock_name_cache[code]
                missing_codes.discard(code)
        
        if not missing_codes:
            return result
        
        # 2. 尝试批量获取股票列表
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_list') and missing_codes:
                try:
                    stock_list = fetcher.get_stock_list()
                    if stock_list is not None and not stock_list.empty:
                        for _, row in stock_list.iterrows():
                            code = row.get('code')
                            name = row.get('name')
                            if code and name:
                                self._stock_name_cache[code] = name
                                if code in missing_codes:
                                    result[code] = name
                                    missing_codes.discard(code)
                        
                        if not missing_codes:
                            break
                        
                        logger.info(f"[股票名称] 从 {fetcher.name} 批量获取完成，剩余 {len(missing_codes)} 个待查")
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 批量获取失败: {e}")
                    continue
        
        # 3. 逐个获取剩余的
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name
                missing_codes.discard(code)
        
        logger.info(f"[股票名称] 批量获取完成，成功 {len(result)}/{len(stock_codes)}")
        return result

    def get_main_indices(self, region: str = "cn") -> List[Dict[str, Any]]:
        """获取主要指数实时行情（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_main_indices(region=region)
                if data:
                    logger.info(f"[{fetcher.name}] 获取指数行情成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取指数行情失败: {e}")
                continue
        return []

    def get_market_stats(self) -> Dict[str, Any]:
        """获取市场涨跌统计（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_market_stats()
                if data:
                    logger.info(f"[{fetcher.name}] 获取市场统计成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取市场统计失败: {e}")
                continue
        return {}

    def get_sector_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """获取板块涨跌榜（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_sector_rankings(n)
                if data:
                    logger.info(f"[{fetcher.name}] 获取板块排行成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取板块排行失败: {e}")
                continue
        return [], []
