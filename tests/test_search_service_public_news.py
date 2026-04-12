# -*- coding: utf-8 -*-
"""搜索服务占位 Key 过滤与公共新闻兜底测试。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent_stock.config import Config
from agent_stock.search_service import SearchService


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class SearchServicePublicNewsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "BOCHA_API_KEYS": os.environ.get("BOCHA_API_KEYS"),
            "TAVILY_API_KEYS": os.environ.get("TAVILY_API_KEYS"),
            "BRAVE_API_KEYS": os.environ.get("BRAVE_API_KEYS"),
            "SERPAPI_API_KEYS": os.environ.get("SERPAPI_API_KEYS"),
        }
        Config.reset_instance()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Config.reset_instance()

    def test_config_filters_placeholder_search_keys(self) -> None:
        os.environ["BOCHA_API_KEYS"] = "your_bocha_key_here,real-bocha-key-123456"
        os.environ["TAVILY_API_KEYS"] = "your_tavily_key_here"
        os.environ["BRAVE_API_KEYS"] = ""
        os.environ["SERPAPI_API_KEYS"] = "real-serpapi-key-1234567890"

        config = Config.get_instance()

        self.assertEqual(config.bocha_api_keys, ["real-bocha-key-123456"])
        self.assertEqual(config.tavily_api_keys, [])
        self.assertEqual(config.serpapi_keys, ["real-serpapi-key-1234567890"])

    @patch("agent_stock.search_service.requests.get")
    def test_public_news_fallback_returns_structured_items(self, mock_get) -> None:
        mock_get.return_value = _FakeResponse(
            """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>宁德时代签署新项目合作协议</title>
      <link>https://news.google.com/rss/articles/CBMiTmh0dHBzOi8vZXhhbXBsZS5jb20vbmV3cy0x0gEA</link>
      <description><![CDATA[<a href="https://example.com/news-1">宁德时代披露了新的合作项目，市场关注度提升。</a>]]></description>
      <pubDate>Sat, 11 Apr 2026 09:00:00 GMT</pubDate>
      <source url="https://example.com">财联社</source>
    </item>
    <item>
      <title>动力电池行业景气度回升</title>
      <link>https://news.google.com/rss/articles/CBMiTmh0dHBzOi8vZXhhbXBsZS5jb20vbmV3cy0y0gEA</link>
      <description><![CDATA[<a href="https://example.com/news-2">机构称行业需求改善，龙头公司受益。</a>]]></description>
      <pubDate>Sat, 11 Apr 2026 06:30:00 GMT</pubDate>
      <source url="https://example.com">证券时报</source>
    </item>
  </channel>
</rss>"""
        )

        service = SearchService(tavily_keys=["your_tavily_key_here"])
        response = service.search_stock_news("300750", "宁德时代", max_results=2)

        self.assertTrue(response.success)
        self.assertEqual(response.provider, "GoogleNewsRSS")
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0].source, "财联社")
        self.assertIn("合作项目", response.results[0].snippet)
        self.assertTrue(response.results[0].published_date)


if __name__ == "__main__":
    unittest.main()
