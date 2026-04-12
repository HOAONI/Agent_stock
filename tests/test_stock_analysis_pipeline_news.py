# -*- coding: utf-8 -*-
"""StockAnalysisPipeline 新闻条目归一化测试。"""

from __future__ import annotations

import unittest

from agent_stock.core.pipeline import StockAnalysisPipeline
from agent_stock.search_service import SearchResponse, SearchResult


class StockAnalysisPipelineNewsTestCase(unittest.TestCase):
    def test_build_news_items_normalizes_and_deduplicates_urls(self) -> None:
        intel_results = {
            "news": SearchResponse(
                query="贵州茅台 最新新闻",
                provider="mock_search",
                success=True,
                results=[
                    SearchResult(
                        title="贵州茅台发布新品",
                        snippet="公司召开新品发布会。",
                        url="https://example.com/news-1",
                        source="example.com",
                        published_date="2026-04-02T09:00:00+08:00",
                    ),
                    SearchResult(
                        title="重复新闻",
                        snippet="这条会被去重。",
                        url="https://example.com/news-1",
                        source="example.com",
                        published_date="2026-04-02T08:30:00+08:00",
                    ),
                ],
            ),
            "announcement": SearchResponse(
                query="贵州茅台 公告",
                provider="mock_search",
                success=True,
                results=[
                    SearchResult(
                        title="贵州茅台公告",
                        snippet="披露年度分红方案。",
                        url="https://example.com/news-2",
                        source="cninfo.com.cn",
                        published_date="2026-04-01",
                    ),
                    SearchResult(
                        title="无链接新闻",
                        snippet="这条因为没有 URL 不应被保留。",
                        url="",
                        source="example.com",
                        published_date="2026-04-01",
                    ),
                ],
            ),
        }

        items = StockAnalysisPipeline._build_news_items(intel_results)

        self.assertEqual([item["url"] for item in items], ["https://example.com/news-1", "https://example.com/news-2"])
        self.assertEqual(items[0]["provider"], "mock_search")
        self.assertEqual(items[0]["dimension"], "news")
        self.assertEqual(items[1]["query"], "贵州茅台 公告")


if __name__ == "__main__":
    unittest.main()
