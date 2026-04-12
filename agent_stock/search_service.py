# -*- coding: utf-8 -*-
"""搜索服务模块。

本模块负责多搜索提供方的统一封装、API Key 轮换、失败降级、正文抓取和缓存。
Signal/Analyzer 并不关心底层用的是哪家搜索服务，只消费这里产出的标准化结果。
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import unescape
from itertools import cycle
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from newspaper import Article, Config
from agent_stock.config import is_valid_secret
from agent_stock.time_utils import local_now

logger = logging.getLogger(__name__)


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def fetch_url_content(url: str, timeout: int = 5) -> str:
    """
    获取 URL 网页正文内容。

    这里只做轻量正文抽取，目标是给 AI 提供补充上下文，而不是完整网页镜像。
    """
    try:
        # `newspaper3k` 对新闻正文抽取比较稳，且能跳过大部分导航和广告区域。
        config = Config()
        config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        config.request_timeout = timeout
        config.fetch_images = False  # 不下载图片
        config.memoize_articles = False  # 不缓存

        article = Article(url, config=config, language="zh")  # 默认中文，但也支持其他
        article.download()
        article.parse()

        # 只保留正文文本，避免把过多网页噪声塞给 AI。
        text = article.text.strip()

        # 简单的后处理，去掉空行并裁剪过长正文。
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        return text[:1500]  # 限制返回长度（比 bs4 稍微多一点，因为 newspaper 解析更干净）
    except Exception as e:
        logger.debug(f"Fetch content failed for {url}: {e}")

    return ""


def _clean_html_text(value: str | None) -> str:
    """把 RSS/HTML 片段清理成适合展示的纯文本。"""
    text = unescape(str(value or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.split())


@dataclass
class SearchResult:
    """单条标准化搜索结果。"""

    title: str
    snippet: str  # 摘要
    url: str
    source: str  # 来源网站
    published_date: str | None = None

    def to_text(self) -> str:
        """转换为文本格式"""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"


@dataclass
class SearchResponse:
    """一次搜索调用的标准化响应。"""

    query: str
    results: list[SearchResult]
    provider: str  # 使用的搜索引擎
    success: bool = True
    error_message: str | None = None
    search_time: float = 0.0  # 搜索耗时（秒）

    def to_context(self, max_results: int = 5) -> str:
        """将搜索结果转换为可用于 AI 分析的上下文"""
        if not self.success or not self.results:
            return f"搜索 '{self.query}' 未找到相关结果。"

        lines = [f"【{self.query} 搜索结果】（来源：{self.provider}）"]
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {result.to_text()}")

        return "\n".join(lines)


class BaseSearchProvider(ABC):
    """搜索引擎基类。"""

    def __init__(self, api_keys: list[str], name: str):
        """
        初始化搜索引擎

        参数：
            api_keys: API Key 列表（支持多个 key 负载均衡）
            name: 搜索引擎名称
        """
        self._api_keys = api_keys
        self._name = name
        self._key_cycle = cycle(api_keys) if api_keys else None
        self._key_usage: dict[str, int] = {key: 0 for key in api_keys}
        self._key_errors: dict[str, int] = {key: 0 for key in api_keys}

    @property
    def name(self) -> str:
        """返回搜索引擎名称。"""
        return self._name

    @property
    def is_available(self) -> bool:
        """检查是否有可用的 API Key"""
        return bool(self._api_keys)

    def _get_next_key(self) -> str | None:
        """
        获取下一个可用的 API Key（负载均衡）

        策略：轮询 + 跳过错误过多的 key
        """
        if not self._key_cycle:
            return None

        # 最多尝试所有 key
        for _ in range(len(self._api_keys)):
            key = next(self._key_cycle)
            # 跳过错误次数过多的 key（超过 3 次）
            if self._key_errors.get(key, 0) < 3:
                return key

        # 所有 key 都有问题时，不立刻报废整个提供方，而是重置计数后再给一次机会。
        logger.warning(f"[{self._name}] 所有 API Key 都有错误记录，重置错误计数")
        self._key_errors = {key: 0 for key in self._api_keys}
        return self._api_keys[0] if self._api_keys else None

    def _record_success(self, key: str) -> None:
        """记录成功使用"""
        self._key_usage[key] = self._key_usage.get(key, 0) + 1
        # 成功后减少错误计数
        if key in self._key_errors and self._key_errors[key] > 0:
            self._key_errors[key] -= 1

    def _record_error(self, key: str) -> None:
        """记录错误"""
        self._key_errors[key] = self._key_errors.get(key, 0) + 1
        logger.warning(f"[{self._name}] API Key {key[:8]}... 错误计数: {self._key_errors[key]}")

    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            return domain or "未知来源"
        except Exception:
            return "未知来源"

    @classmethod
    def _build_result(
        cls,
        *,
        title: str,
        snippet: str,
        url: str,
        source: str | None = None,
        published_date: str | None = None,
        snippet_limit: int | None = None,
    ) -> SearchResult:
        """构造统一的搜索结果对象。"""
        if snippet_limit is not None and snippet:
            snippet = snippet[:snippet_limit]
        return SearchResult(
            title=title,
            snippet=snippet,
            url=url,
            source=source or cls._extract_domain(url),
            published_date=published_date,
        )

    def _success_response(self, query: str, results: list[SearchResult]) -> SearchResponse:
        """构造成功响应。"""
        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            success=True,
        )

    def _failure_response(self, query: str, error_message: str) -> SearchResponse:
        """构造失败响应。"""
        return SearchResponse(
            query=query,
            results=[],
            provider=self.name,
            success=False,
            error_message=error_message,
        )

    @abstractmethod
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行搜索（子类实现）"""
        pass

    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """
        执行搜索

        参数：
            query: 搜索关键词
            max_results: 最大返回结果数
            days: 搜索最近几天的时间范围（默认7天）

        返回：
            SearchResponse 对象
        """
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} 未配置 API Key",
            )

        start_time = time.time()
        try:
            response = self._do_search(query, api_key, max_results, days=days)
            response.search_time = time.time() - start_time

            if response.success:
                self._record_success(api_key)
                logger.info(
                    f"[{self._name}] 搜索 '{query}' 成功，返回 {len(response.results)} 条结果，耗时 {response.search_time:.2f}s"
                )
            else:
                self._record_error(api_key)

            return response

        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query, results=[], provider=self._name, success=False, error_message=str(e), search_time=elapsed
            )


class PublicNewsRSSProvider(BaseSearchProvider):
    """公共 RSS 新闻提供方，无需第三方 API Key。"""

    API_ENDPOINT = "https://news.google.com/rss/search"

    def __init__(self):
        super().__init__(["public_rss"], "GoogleNewsRSS")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        del api_key

        try:
            is_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
            locale = {
                "hl": "zh-CN" if is_chinese else "en-US",
                "gl": "CN" if is_chinese else "US",
                "ceid": "CN:zh-Hans" if is_chinese else "US:en",
            }
            search_query = f"{query} when:{max(1, days)}d"
            response = requests.get(
                self.API_ENDPOINT,
                params={"q": search_query, **locale},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()

            root = ET.fromstring(response.text)
            items: list[SearchResult] = []
            for item in root.findall("./channel/item")[:max_results]:
                title = _clean_html_text(item.findtext("title"))
                link = str(item.findtext("link") or "").strip()
                description = _clean_html_text(item.findtext("description"))
                source = _clean_html_text(item.findtext("source")) or self._extract_domain(link)
                published_date = None
                raw_pub_date = str(item.findtext("pubDate") or "").strip()
                if raw_pub_date:
                    try:
                        published_date = parsedate_to_datetime(raw_pub_date).astimezone().strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        published_date = raw_pub_date

                if not link or not title:
                    continue

                items.append(
                    self._build_result(
                        title=title,
                        snippet=description,
                        url=link,
                        source=source,
                        published_date=published_date,
                        snippet_limit=500,
                    )
                )

            return self._success_response(query, items)
        except requests.exceptions.RequestException as exc:
            return self._failure_response(query, f"公共 RSS 请求失败: {exc}")
        except ET.ParseError as exc:
            return self._failure_response(query, f"公共 RSS 解析失败: {exc}")


class TavilySearchProvider(BaseSearchProvider):
    """
    Tavily 搜索引擎

    特点：
    - 专为 AI/LLM 优化的搜索 API
    - 免费版每月 1000 次请求
    - 返回结构化的搜索结果

    文档：https://docs.tavily.com/
    """

    def __init__(self, api_keys: list[str]):
        """初始化 Tavily 搜索提供方。"""
        super().__init__(api_keys, "Tavily")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 Tavily 搜索"""
        try:
            # noinspection PyPackageRequirements
            from tavily import TavilyClient
        except ImportError:
            return self._failure_response(query, "tavily-python 未安装，请运行: pip install tavily-python")

        try:
            client = TavilyClient(api_key=api_key)

            # 执行搜索（优化：使用advanced深度、限制最近几天）
            response = client.search(
                query=query,
                search_depth="advanced",  # advanced 获取更多结果
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                days=days,  # 搜索最近天数的内容
            )

            # 记录原始响应到日志
            logger.info(f"[Tavily] 搜索完成，query='{query}', 返回 {len(response.get('results', []))} 条结果")
            logger.debug(f"[Tavily] 原始响应: {response}")

            # 解析结果
            results = []
            for item in response.get("results", []):
                results.append(
                    self._build_result(
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        url=item.get("url", ""),
                        snippet_limit=500,
                        published_date=item.get("published_date"),
                    )
                )

            return self._success_response(query, results)

        except Exception as e:
            error_msg = str(e)
            # 检查是否是配额问题
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                error_msg = f"API 配额已用尽: {error_msg}"

            return self._failure_response(query, error_msg)


class SerpAPISearchProvider(BaseSearchProvider):
    """
    SerpAPI 搜索引擎

    特点：
    - 支持 Google、Bing、百度等多种搜索引擎
    - 免费版每月 100 次请求
    - 返回真实的搜索结果

    文档：https://serpapi.com/baidu-search-api?utm_source=github_daily_stock_analysis
    """

    def __init__(self, api_keys: list[str]):
        """初始化 SerpAPI 搜索提供方。"""
        super().__init__(api_keys, "SerpAPI")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 SerpAPI 搜索"""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return self._failure_response(
                query, "google-search-results 未安装，请运行: pip install google-search-results"
            )

        try:
            # 确定时间范围参数 tbs
            tbs = "qdr:w"  # 默认一周
            if days <= 1:
                tbs = "qdr:d"  # 过去24小时
            elif days <= 7:
                tbs = "qdr:w"  # 过去一周
            elif days <= 30:
                tbs = "qdr:m"  # 过去一月
            else:
                tbs = "qdr:y"  # 过去一年

            # 使用 Google 搜索（获取 Knowledge Graph、Answer Box 等）
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "google_domain": "google.com.hk",  # 使用香港谷歌，中文支持较好
                "hl": "zh-cn",  # 中文界面
                "gl": "cn",  # 中国地区偏好
                "tbs": tbs,  # 时间范围限制
                "num": max_results,  # 请求的结果数量，注意：Google API有时不严格遵守
            }

            search = GoogleSearch(params)
            response = search.get_dict()

            # 记录原始响应到日志
            logger.debug(f"[SerpAPI] 原始响应 keys: {response.keys()}")

            # 解析结果
            results = []

            # 1. 解析 Knowledge Graph （知识图谱）
            kg = response.get("knowledge_graph", {})
            if kg:
                title = kg.get("title", "知识图谱")
                desc = kg.get("description", "")

                # 提取额外属性
                details = []
                for key in ["type", "founded", "headquarters", "employees", "ceo"]:
                    val = kg.get(key)
                    if val:
                        details.append(f"{key}: {val}")

                snippet = f"{desc}\n" + " | ".join(details) if details else desc

                results.append(
                    self._build_result(
                        title=f"[知识图谱] {title}",
                        snippet=snippet,
                        url=kg.get("source", {}).get("link", ""),
                        source="Google Knowledge Graph",
                    )
                )

            # 2. 解析 Answer Box （精选回答/行情卡片）
            ab = response.get("answer_box", {})
            if ab:
                ab_title = ab.get("title", "精选回答")
                ab_snippet = ""

                # 财经类回答
                if ab.get("type") == "finance_results":
                    stock = ab.get("stock", "")
                    price = ab.get("price", "")
                    currency = ab.get("currency", "")
                    movement = ab.get("price_movement", {})
                    mv_val = movement.get("percentage", 0)
                    mv_dir = movement.get("movement", "")

                    ab_title = f"[行情卡片] {stock}"
                    ab_snippet = f"价格: {price} {currency}\n涨跌: {mv_dir} {mv_val}%"

                    # 提取表格数据
                    if "table" in ab:
                        table_data = []
                        for row in ab["table"]:
                            if "name" in row and "value" in row:
                                table_data.append(f"{row['name']}: {row['value']}")
                        if table_data:
                            ab_snippet += "\n" + "; ".join(table_data)

                # 普通文本回答
                elif "snippet" in ab:
                    ab_snippet = ab.get("snippet", "")
                    list_items = ab.get("list", [])
                    if list_items:
                        ab_snippet += "\n" + "\n".join([f"- {item}" for item in list_items])

                elif "answer" in ab:
                    ab_snippet = ab.get("answer", "")

                if ab_snippet:
                    results.append(
                        self._build_result(
                            title=f"[精选回答] {ab_title}",
                            snippet=ab_snippet,
                            url=ab.get("link", "") or ab.get("displayed_link", ""),
                            source="Google Answer Box",
                        )
                    )

            # 3. 解析 Related Questions （相关问题）
            rqs = response.get("related_questions", [])
            for rq in rqs[:3]:  # 取前3个
                question = rq.get("question", "")
                snippet = rq.get("snippet", "")
                link = rq.get("link", "")

                if question and snippet:
                    results.append(
                        self._build_result(
                            title=f"[相关问题] {question}",
                            snippet=snippet,
                            url=link,
                            source="Google Related Questions",
                        )
                    )

            # 4. 解析 Organic Results （自然搜索结果）
            organic_results = response.get("organic_results", [])

            for item in organic_results[:max_results]:
                link = item.get("link", "")
                snippet = item.get("snippet", "")

                # 增强：如果需要，解析网页正文
                # 策略：如果摘要太短，或者为了获取更多信息，可以请求网页
                # 这里我们对所有结果尝试获取正文，但为了性能，仅获取前 1000 个字符
                content = ""
                if link:
                    try:
                        fetched_content = fetch_url_content(link, timeout=5)
                        if fetched_content:
                            # 如果获取到了正文，将其拼接到 snippet 中，或者替换 snippet
                            # 这里选择拼接，保留原摘要
                            content = fetched_content
                            if len(content) > 500:
                                snippet = f"{snippet}\n\n【网页详情】\n{content[:500]}..."
                            else:
                                snippet = f"{snippet}\n\n【网页详情】\n{content}"
                    except Exception as e:
                        logger.debug(f"[SerpAPI] Fetch content failed: {e}")

                results.append(
                    self._build_result(
                        title=item.get("title", ""),
                        snippet=snippet,
                        url=link,
                        source=item.get("source"),
                        snippet_limit=1000,
                        published_date=item.get("date"),
                    )
                )

            return self._success_response(query, results)

        except Exception as e:
            error_msg = str(e)
            return self._failure_response(query, error_msg)


class BochaSearchProvider(BaseSearchProvider):
    """
    博查搜索引擎

    特点：
    - 专为AI优化的中文搜索API
    - 结果准确、摘要完整
    - 支持时间范围过滤和AI摘要
    - 兼容Bing Search API格式

    文档：https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
    """

    def __init__(self, api_keys: list[str]):
        """初始化博查搜索提供方。"""
        super().__init__(api_keys, "Bocha")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行博查搜索"""
        try:
            import requests
        except ImportError:
            return self._failure_response(query, "requests 未安装，请运行: pip install requests")

        try:
            # API 端点
            url = "https://api.bocha.cn/v1/web-search"

            # 请求头
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            # 确定时间范围
            freshness = "oneWeek"
            if days <= 1:
                freshness = "oneDay"
            elif days <= 7:
                freshness = "oneWeek"
            elif days <= 30:
                freshness = "oneMonth"
            else:
                freshness = "oneYear"

            # 请求参数（严格按照API文档）
            payload = {
                "query": query,
                "freshness": freshness,  # 动态时间范围
                "summary": True,  # 启用AI摘要
                "count": min(max_results, 50),  # 最大50条
            }

            # 执行搜索
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            # 检查HTTP状态码
            if response.status_code != 200:
                # 尝试解析错误信息
                try:
                    if response.headers.get("content-type", "").startswith("application/json"):
                        error_data = response.json()
                        error_message = error_data.get("message", response.text)
                    else:
                        error_message = response.text
                except Exception:
                    error_message = response.text

                # 根据错误码处理
                if response.status_code == 403:
                    error_msg = f"余额不足: {error_message}"
                elif response.status_code == 401:
                    error_msg = f"API KEY无效: {error_message}"
                elif response.status_code == 400:
                    error_msg = f"请求参数错误: {error_message}"
                elif response.status_code == 429:
                    error_msg = f"请求频率达到限制: {error_message}"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_message}"

                logger.warning(f"[Bocha] 搜索失败: {error_msg}")

                return self._failure_response(query, error_msg)

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Bocha] {error_msg}")
                return self._failure_response(query, error_msg)

            # 检查响应code
            if data.get("code") != 200:
                error_msg = data.get("msg") or f"API返回错误码: {data.get('code')}"
                return self._failure_response(query, error_msg)

            # 记录原始响应到日志
            logger.info(f"[Bocha] 搜索完成，query='{query}'")
            logger.debug(f"[Bocha] 原始响应: {data}")

            # 解析搜索结果
            results = []
            web_pages = data.get("data", {}).get("webPages", {})
            value_list = web_pages.get("value", [])

            for item in value_list[:max_results]:
                # 优先使用summary（AI 摘要），回退到snippet
                snippet = item.get("summary") or item.get("snippet", "")

                results.append(
                    self._build_result(
                        title=item.get("name", ""),
                        snippet=snippet,
                        url=item.get("url", ""),
                        source=item.get("siteName") or None,
                        snippet_limit=500,
                        published_date=item.get("datePublished"),  # UTC+8 时间格式，无需转换
                    )
                )

            logger.info(f"[Bocha] 成功解析 {len(results)} 条结果")

            return self._success_response(query, results)

        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Bocha] {error_msg}")
            return self._failure_response(query, error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return self._failure_response(query, error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return self._failure_response(query, error_msg)


class BraveSearchProvider(BaseSearchProvider):
    """
    Brave 搜索引擎

    特点：
    - 隐私优先的独立搜索引擎
    - 索引超过300亿页面
    - 免费层可用
    - 支持时间范围过滤

    文档：https://brave.com/search/api/
    """

    API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_keys: list[str]):
        """初始化 Brave 搜索提供方。"""
        super().__init__(api_keys, "Brave")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 Brave 搜索"""
        try:
            # 请求头
            headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

            # 确定时间范围（freshness 参数）
            if days <= 1:
                freshness = "pd"  # 过去一天（24 小时）
            elif days <= 7:
                freshness = "pw"  # 过去一周
            elif days <= 30:
                freshness = "pm"  # 过去一个月
            else:
                freshness = "py"  # 过去一年

            # 请求参数
            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave 最大支持20条
                "freshness": freshness,
                "search_lang": "en",  # 英文内容（US股票优先）
                "country": "US",  # 美国区域偏好
                "safesearch": "moderate",
            }

            # 执行搜索（GET 请求）
            response = requests.get(self.API_ENDPOINT, headers=headers, params=params, timeout=10)

            # 检查 HTTP 状态码
            if response.status_code != 200:
                error_msg = self._parse_error(response)
                logger.warning(f"[Brave] 搜索失败: {error_msg}")
                return self._failure_response(query, error_msg)

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Brave] {error_msg}")
                return self._failure_response(query, error_msg)

            logger.info(f"[Brave] 搜索完成，query='{query}'")
            logger.debug(f"[Brave] 原始响应: {data}")

            # 解析搜索结果
            results = []
            web_data = data.get("web", {})
            web_results = web_data.get("results", [])

            for item in web_results[:max_results]:
                # 解析发布日期（ISO 8601 格式）
                published_date = None
                age = item.get("age") or item.get("page_age")
                if age:
                    try:
                        # 转换 ISO 格式为简单日期字符串
                        dt = datetime.fromisoformat(age.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        published_date = age  # 解析失败时使用原始值

                results.append(
                    self._build_result(
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        url=item.get("url", ""),
                        snippet_limit=500,
                        published_date=published_date,
                    )
                )

            logger.info(f"[Brave] 成功解析 {len(results)} 条结果")

            return self._success_response(query, results)

        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Brave] {error_msg}")
            return self._failure_response(query, error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return self._failure_response(query, error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return self._failure_response(query, error_msg)

    def _parse_error(self, response) -> str:
        """解析错误响应"""
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                # Brave API 返回的错误格式
                if "message" in error_data:
                    return error_data["message"]
                if "error" in error_data:
                    return error_data["error"]
                return str(error_data)
            return response.text[:200]
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"


class SearchService:
    """
    搜索服务

    功能：
    1. 管理多个搜索引擎
    2. 自动故障转移
    3. 结果聚合和格式化
    4. 数据源失败时的增强搜索（股价、走势等）
    5. 港股/美股自动使用英文搜索关键词
    """

    # 增强搜索关键词模板（A股 中文）
    ENHANCED_SEARCH_KEYWORDS = [
        "{name} 股票 今日 股价",
        "{name} {code} 最新 行情 走势",
        "{name} 股票 分析 走势图",
        "{name} K线 技术分析",
        "{name} {code} 涨跌 成交量",
    ]

    # 增强搜索关键词模板（港股/美股 英文）
    ENHANCED_SEARCH_KEYWORDS_EN = [
        "{name} stock price today",
        "{name} {code} latest quote trend",
        "{name} stock analysis chart",
        "{name} technical analysis",
        "{name} {code} performance volume",
    ]

    def __init__(
        self,
        bocha_keys: list[str] | None = None,
        tavily_keys: list[str] | None = None,
        brave_keys: list[str] | None = None,
        serpapi_keys: list[str] | None = None,
        news_max_age_days: int = 3,
    ):
        """
        初始化搜索服务

        参数：
            bocha_keys: 博查搜索 API Key 列表
            tavily_keys: Tavily API Key 列表
            brave_keys: Brave 搜索 API Key 列表
            serpapi_keys: SerpAPI Key 列表
            news_max_age_days: 新闻最大时效（天）
        """
        self._providers: list[BaseSearchProvider] = []
        self.news_max_age_days = max(1, news_max_age_days)
        bocha_keys = self._sanitize_keys(bocha_keys)
        tavily_keys = self._sanitize_keys(tavily_keys)
        brave_keys = self._sanitize_keys(brave_keys)
        serpapi_keys = self._sanitize_keys(serpapi_keys)

        # 初始化搜索引擎（按优先级排序）
        # 1. 优先使用 Bocha（中文搜索优化，AI 摘要）
        if bocha_keys:
            self._providers.append(BochaSearchProvider(bocha_keys))
            logger.info(f"已配置 Bocha 搜索，共 {len(bocha_keys)} 个 API Key")

        # 2. Tavily（免费额度更多，每月 1000 次）
        if tavily_keys:
            self._providers.append(TavilySearchProvider(tavily_keys))
            logger.info(f"已配置 Tavily 搜索，共 {len(tavily_keys)} 个 API Key")

        # 3. Brave 搜索（隐私优先，全球覆盖）
        if brave_keys:
            self._providers.append(BraveSearchProvider(brave_keys))
            logger.info(f"已配置 Brave 搜索，共 {len(brave_keys)} 个 API Key")

        # 4. SerpAPI 作为备选（每月 100 次）
        if serpapi_keys:
            self._providers.append(SerpAPISearchProvider(serpapi_keys))
            logger.info(f"已配置 SerpAPI 搜索，共 {len(serpapi_keys)} 个 API Key")

        if not self._providers:
            self._providers.append(PublicNewsRSSProvider())
            logger.warning("未配置有效第三方搜索 API Key，已启用 Google News RSS 公共新闻兜底")

        # 内存搜索结果缓存：{cache_key: (timestamp, SearchResponse)}
        self._cache: dict[str, tuple[float, SearchResponse]] = {}
        # 默认缓存 TTL（秒，10 分钟）
        self._cache_ttl: int = 600

    @staticmethod
    def _sanitize_keys(keys: list[str] | None) -> list[str]:
        """过滤空值与占位符，避免把伪配置当成可用搜索源。"""
        return [str(key).strip() for key in keys or [] if is_valid_secret(str(key).strip())]

    @staticmethod
    def _is_foreign_stock(stock_code: str) -> bool:
        """判断是否为港股或美股"""
        import re

        code = stock_code.strip()
        # 美股：1-5个大写字母，可能包含点（如 BRK.B）
        if re.match(r"^[A-Za-z]{1,5}(\.[A-Za-z])?$", code):
            return True
        # 港股：带 hk 前缀或 5位纯数字
        lower = code.lower()
        if lower.startswith("hk"):
            return True
        if code.isdigit() and len(code) == 5:
            return True
        return False

    @property
    def is_available(self) -> bool:
        """检查是否有可用的搜索引擎"""
        return any(p.is_available for p in self._providers)

    def _cache_key(self, query: str, max_results: int, days: int) -> str:
        """根据查询参数构建缓存键。"""
        return f"{query}|{max_results}|{days}"

    def _get_cached(self, key: str) -> SearchResponse | None:
        """若缓存的 `SearchResponse` 仍有效则返回，否则返回 `None`。"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, response = entry
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        logger.debug(f"搜索缓存命中: {key[:60]}...")
        return response

    def _put_cache(self, key: str, response: SearchResponse) -> None:
        """将成功的 `SearchResponse` 写入缓存。"""
        # 硬上限：缓存超过限制时淘汰最早条目
        _MAX_CACHE_SIZE = 500
        if len(self._cache) >= _MAX_CACHE_SIZE:
            now = time.time()
            # 第一轮：删除已过期条目
            expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._cache_ttl]
            for k in expired:
                del self._cache[k]
            # 第二轮：若仍超限，则按先进先出淘汰最早条目
            if len(self._cache) >= _MAX_CACHE_SIZE:
                excess = len(self._cache) - _MAX_CACHE_SIZE + 1
                oldest = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[:excess]
                for k in oldest:
                    del self._cache[k]
        self._cache[key] = (time.time(), response)

    def search_stock_news(
        self, stock_code: str, stock_name: str, max_results: int = 5, focus_keywords: list[str] | None = None
    ) -> SearchResponse:
        """
        搜索股票相关新闻

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            max_results: 最大返回结果数
            focus_keywords: 重点关注的关键词列表

        返回：
            SearchResponse 对象
        """
        # 智能确定搜索时间范围
        # 策略：
        # 1. 周二至周五：搜索近1天（24小时）
        # 2. 周六、周日：搜索近2-3天（覆盖周末）
        # 3. 周一：搜索近3天（覆盖周末）
        # 4. 用 NEWS_MAX_AGE_DAYS 限制上限
        today_weekday = local_now().weekday()
        if today_weekday == 0:  # 周一
            weekday_days = 3
        elif today_weekday >= 5:  # 周六(5)、周日(6)
            weekday_days = 2
        else:  # 周二(1) - 周五(4)
            weekday_days = 1
        search_days = min(weekday_days, self.news_max_age_days)

        # 构建搜索查询（优化搜索效果）
        is_foreign = self._is_foreign_stock(stock_code)
        if focus_keywords:
            # 如果提供了关键词，直接使用关键词作为查询
            query = " ".join(focus_keywords)
        elif is_foreign:
            # 港股/美股使用英文搜索关键词
            query = f"{stock_name} {stock_code} stock latest news"
        else:
            # 默认主查询：股票名称 + 核心关键词
            query = f"{stock_name} {stock_code} 股票 最新消息"

        logger.info(f"搜索股票新闻: {stock_name}({stock_code}), query='{query}', 时间范围: 近{search_days}天")

        # 先检查缓存
        cache_key = self._cache_key(query, max_results, search_days)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(f"使用缓存搜索结果: {stock_name}({stock_code})")
            return cached

        # 依次尝试各个搜索引擎
        for provider in self._providers:
            if not provider.is_available:
                continue

            response = provider.search(query, max_results, days=search_days)

            if response.success and response.results:
                logger.info(f"使用 {provider.name} 搜索成功")
                self._put_cache(cache_key, response)
                return response
            else:
                logger.warning(f"{provider.name} 搜索失败: {response.error_message}，尝试下一个引擎")

        # 所有引擎都失败
        return SearchResponse(
            query=query, results=[], provider="None", success=False, error_message="所有搜索引擎都不可用或搜索失败"
        )

    def search_stock_events(
        self, stock_code: str, stock_name: str, event_types: list[str] | None = None
    ) -> SearchResponse:
        """
        搜索股票特定事件（年报预告、减持等）

        专门针对交易决策相关的重要事件进行搜索

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            event_types: 事件类型列表

        返回：
            SearchResponse 对象
        """
        if event_types is None:
            if self._is_foreign_stock(stock_code):
                event_types = ["earnings report", "insider selling", "quarterly results"]
            else:
                event_types = ["年报预告", "减持公告", "业绩快报"]

        # 构建针对性查询
        event_query = " OR ".join(event_types)
        query = f"{stock_name} ({event_query})"

        logger.info(f"搜索股票事件: {stock_name}({stock_code}) - {event_types}")

        # 依次尝试各个搜索引擎
        for provider in self._providers:
            if not provider.is_available:
                continue

            response = provider.search(query, max_results=5)

            if response.success:
                return response

        return SearchResponse(query=query, results=[], provider="None", success=False, error_message="事件搜索失败")

    def search_comprehensive_intel(
        self, stock_code: str, stock_name: str, max_searches: int = 3
    ) -> dict[str, SearchResponse]:
        """
        多维度情报搜索（同时使用多个搜索引擎，并覆盖多个分析维度）

        搜索维度：
        1. 最新消息 - 近期新闻动态
        2. 风险排查 - 减持、处罚、利空
        3. 业绩预期 - 年报预告、业绩快报

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            max_searches: 最大搜索次数

        返回：
            {维度名称: SearchResponse} 字典
        """
        results = {}
        search_count = 0

        # 根据股票类型选择搜索关键词语言
        is_foreign = self._is_foreign_stock(stock_code)

        # 定义搜索维度
        if is_foreign:
            search_dimensions = [
                {"name": "latest_news", "query": f"{stock_name} {stock_code} latest news events", "desc": "最新消息"},
                {
                    "name": "market_analysis",
                    "query": f"{stock_name} analyst rating target price report",
                    "desc": "机构分析",
                },
                {
                    "name": "risk_check",
                    "query": f"{stock_name} risk insider selling lawsuit litigation",
                    "desc": "风险排查",
                },
                {
                    "name": "earnings",
                    "query": f"{stock_name} earnings revenue profit growth forecast",
                    "desc": "业绩预期",
                },
                {
                    "name": "industry",
                    "query": f"{stock_name} industry competitors market share outlook",
                    "desc": "行业分析",
                },
            ]
        else:
            search_dimensions = [
                {"name": "latest_news", "query": f"{stock_name} {stock_code} 最新 新闻 重大 事件", "desc": "最新消息"},
                {"name": "market_analysis", "query": f"{stock_name} 研报 目标价 评级 深度分析", "desc": "机构分析"},
                {"name": "risk_check", "query": f"{stock_name} 减持 处罚 违规 诉讼 利空 风险", "desc": "风险排查"},
                {"name": "earnings", "query": f"{stock_name} 业绩预告 财报 营收 净利润 同比增长", "desc": "业绩预期"},
                {"name": "industry", "query": f"{stock_name} 所在行业 竞争对手 市场份额 行业前景", "desc": "行业分析"},
            ]

        logger.info(f"开始多维度情报搜索: {stock_name}({stock_code})")

        # 轮流使用不同的搜索引擎
        provider_index = 0

        for dim in search_dimensions:
            if search_count >= max_searches:
                break

            # 选择搜索引擎（轮流使用）
            available_providers = [p for p in self._providers if p.is_available]
            if not available_providers:
                break

            provider = available_providers[provider_index % len(available_providers)]
            provider_index += 1

            logger.info(f"[情报搜索] {dim['desc']}: 使用 {provider.name}")

            response = provider.search(dim["query"], max_results=3, days=self.news_max_age_days)
            results[dim["name"]] = response
            search_count += 1

            if response.success:
                logger.info(f"[情报搜索] {dim['desc']}: 获取 {len(response.results)} 条结果")
            else:
                logger.warning(f"[情报搜索] {dim['desc']}: 搜索失败 - {response.error_message}")

            # 短暂延迟避免请求过快
            time.sleep(0.5)

        return results

    def format_intel_report(self, intel_results: dict[str, SearchResponse], stock_name: str) -> str:
        """
        格式化情报搜索结果为报告

        参数：
            intel_results: 多维度搜索结果
            stock_name: 股票名称

        返回：
            格式化的情报报告文本
        """
        lines = [f"【{stock_name} 情报搜索结果】"]

        # 维度展示顺序
        display_order = ["latest_news", "market_analysis", "risk_check", "earnings", "industry"]

        for dim_name in display_order:
            if dim_name not in intel_results:
                continue

            resp = intel_results[dim_name]

            # 获取维度描述
            dim_desc = dim_name
            if dim_name == "latest_news":
                dim_desc = "📰 最新消息"
            elif dim_name == "market_analysis":
                dim_desc = "📈 机构分析"
            elif dim_name == "risk_check":
                dim_desc = "⚠️ 风险排查"
            elif dim_name == "earnings":
                dim_desc = "📊 业绩预期"
            elif dim_name == "industry":
                dim_desc = "🏭 行业分析"

            lines.append(f"\n{dim_desc} (来源: {resp.provider}):")
            if resp.success and resp.results:
                # 增加显示条数
                for i, r in enumerate(resp.results[:4], 1):
                    date_str = f" [{r.published_date}]" if r.published_date else ""
                    lines.append(f"  {i}. {r.title}{date_str}")
                    # 如果摘要太短，可能信息量不足
                    snippet = r.snippet[:150] if len(r.snippet) > 20 else r.snippet
                    lines.append(f"     {snippet}...")
            else:
                lines.append("  未找到相关信息")

        return "\n".join(lines)

    def batch_search(
        self, stocks: list[dict[str, str]], max_results_per_stock: int = 3, delay_between: float = 1.0
    ) -> dict[str, SearchResponse]:
        """
        为多只股票批量搜索新闻。

        参数：
            stocks: List of stocks
            max_results_per_stock: Max results per stock
            delay_between: Delay between searches (seconds)

        返回：
            Dict of results
        """
        results = {}

        for i, stock in enumerate(stocks):
            if i > 0:
                time.sleep(delay_between)

            code = stock.get("code", "")
            name = stock.get("name", "")

            response = self.search_stock_news(code, name, max_results_per_stock)
            results[code] = response

        return results

    def search_stock_price_fallback(
        self, stock_code: str, stock_name: str, max_attempts: int = 3, max_results: int = 5
    ) -> SearchResponse:
        """
        在数据源失败时增强搜索兜底。

        When all data sources (efinance, akshare, tushare, baostock, etc.) fail to get
        stock data, use search engines to find stock trends and price info as supplemental data for AI analysis.

        Strategy:
        1. Search using multiple keyword templates
        2. Try all available search engines for each keyword
        3. Aggregate and deduplicate results

        参数：
            stock_code: Stock Code
            stock_name: Stock Name
            max_attempts: Max search attempts (using different keywords)
            max_results: Max results to return

        返回：
            SearchResponse object with aggregated results
        """

        if not self.is_available:
            return SearchResponse(
                query=f"{stock_name} 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="未配置搜索引擎 API Key",
            )

        logger.info(f"[增强搜索] 数据源失败，启动增强搜索: {stock_name}({stock_code})")

        all_results = []
        seen_urls = set()
        successful_providers = []

        # 使用多个关键词模板搜索
        is_foreign = self._is_foreign_stock(stock_code)
        keywords = self.ENHANCED_SEARCH_KEYWORDS_EN if is_foreign else self.ENHANCED_SEARCH_KEYWORDS
        for i, keyword_template in enumerate(keywords[:max_attempts]):
            query = keyword_template.format(name=stock_name, code=stock_code)

            logger.info(f"[增强搜索] 第 {i+1}/{max_attempts} 次搜索: {query}")

            # 依次尝试各个搜索引擎
            for provider in self._providers:
                if not provider.is_available:
                    continue

                try:
                    response = provider.search(query, max_results=3)

                    if response.success and response.results:
                        # 去重并添加结果
                        for result in response.results:
                            if result.url not in seen_urls:
                                seen_urls.add(result.url)
                                all_results.append(result)

                        if provider.name not in successful_providers:
                            successful_providers.append(provider.name)

                        logger.info(f"[增强搜索] {provider.name} 返回 {len(response.results)} 条结果")
                        break  # 成功后跳到下一个关键词
                    else:
                        logger.debug(f"[增强搜索] {provider.name} 无结果或失败")

                except Exception as e:
                    logger.warning(f"[增强搜索] {provider.name} 搜索异常: {e}")
                    continue

            # 短暂延迟避免请求过快
            if i < max_attempts - 1:
                time.sleep(0.5)

        # 汇总结果
        if all_results:
            # 截取前 max_results 条
            final_results = all_results[:max_results]
            provider_str = ", ".join(successful_providers) if successful_providers else "None"

            logger.info(f"[增强搜索] 完成，共获取 {len(final_results)} 条结果（来源: {provider_str}）")

            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=final_results,
                provider=provider_str,
                success=True,
            )
        else:
            logger.warning("[增强搜索] 所有搜索均未返回结果")
            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="增强搜索未找到相关信息",
            )

    def search_stock_with_enhanced_fallback(
        self,
        stock_code: str,
        stock_name: str,
        include_news: bool = True,
        include_price: bool = False,
        max_results: int = 5,
    ) -> dict[str, SearchResponse]:
        """
        综合搜索接口（支持新闻和股价信息）

        当 include_price=True 时，会同时搜索新闻和股价信息。
        主要用于数据源完全失败时的兜底方案。

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            include_news: 是否搜索新闻
            include_price: 是否搜索股价/走势信息
            max_results: 每类搜索的最大结果数

        返回：
            {'news': SearchResponse, 'price': SearchResponse} 字典
        """
        results = {}

        if include_news:
            results["news"] = self.search_stock_news(stock_code, stock_name, max_results=max_results)

        if include_price:
            results["price"] = self.search_stock_price_fallback(
                stock_code, stock_name, max_attempts=3, max_results=max_results
            )

        return results

    def format_price_search_context(self, response: SearchResponse) -> str:
        """
        将股价搜索结果格式化为 AI 分析上下文

        参数：
            response: 搜索响应对象

        返回：
            格式化的文本，可直接用于 AI 分析
        """
        if not response.success or not response.results:
            return "【股价走势搜索】未找到相关信息，请以其他渠道数据为准。"

        lines = [
            f"【股价走势搜索结果】（来源: {response.provider}）",
            "⚠️ 注意：以下信息来自网络搜索，仅供参考，可能存在延迟或不准确。",
            "",
        ]

        for i, result in enumerate(response.results, 1):
            date_str = f" [{result.published_date}]" if result.published_date else ""
            lines.append(f"{i}. 【{result.source}】{result.title}{date_str}")
            lines.append(f"   {result.snippet[:200]}...")
            lines.append("")

        return "\n".join(lines)


# === 便捷函数 ===
_search_service: SearchService | None = None


def get_search_service() -> SearchService:
    """获取搜索服务单例"""
    global _search_service

    if _search_service is None:
        from agent_stock.config import get_config

        config = get_config()

        _search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            brave_keys=config.brave_api_keys,
            serpapi_keys=config.serpapi_keys,
            news_max_age_days=config.news_max_age_days,
        )

    return _search_service


def reset_search_service() -> None:
    """重置搜索服务（用于测试）"""
    global _search_service
    _search_service = None


if __name__ == "__main__":
    # 测试搜索服务
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s")

    # 手动测试（需要配置 API Key）
    service = get_search_service()

    if service.is_available:
        print("=== 测试股票新闻搜索 ===")
        response = service.search_stock_news("300389", "艾比森")
        print(f"搜索状态: {'成功' if response.success else '失败'}")
        print(f"搜索引擎: {response.provider}")
        print(f"结果数量: {len(response.results)}")
        print(f"耗时: {response.search_time:.2f}s")
        print("\n" + response.to_context())
    else:
        print("未配置搜索引擎 API Key，跳过测试")
