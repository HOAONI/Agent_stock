# -*- coding: utf-8 -*-
"""Unit tests for request-scoped runtime LLM overrides."""

from __future__ import annotations

import unittest

from src.config import Config, RuntimeLlmConfig


class RuntimeLlmOverrideTestCase(unittest.TestCase):
    def test_clone_for_runtime_llm_overrides_openai_compatible_provider(self):
        base = Config(
            gemini_api_key="gemini-system-key",
            gemini_model="gemini-system-model",
            anthropic_api_key="anthropic-system-key",
            anthropic_model="claude-system-model",
            openai_api_key="openai-system-key",
            openai_base_url="https://api.openai.com/v1",
            openai_model="gpt-4o-mini",
        )

        cloned = base.clone_for_runtime_llm(
            RuntimeLlmConfig(
                provider="deepseek",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_token="deepseek-personal-key",
                has_token=True,
            )
        )

        self.assertEqual(cloned.openai_api_key, "deepseek-personal-key")
        self.assertEqual(cloned.openai_base_url, "https://api.deepseek.com")
        self.assertEqual(cloned.openai_model, "deepseek-chat")
        self.assertIsNone(cloned.gemini_api_key)
        self.assertIsNone(cloned.anthropic_api_key)
        self.assertEqual(base.openai_api_key, "openai-system-key")

    def test_clone_for_runtime_llm_overrides_gemini_provider(self):
        base = Config(
            gemini_api_key="gemini-system-key",
            gemini_model="gemini-system-model",
            openai_api_key="openai-system-key",
            openai_base_url="https://api.openai.com/v1",
            openai_model="gpt-4o-mini",
        )

        cloned = base.clone_for_runtime_llm(
            RuntimeLlmConfig(
                provider="gemini",
                base_url="https://generativelanguage.googleapis.com",
                model="gemini-2.5-flash",
                api_token="gemini-personal-key",
                has_token=True,
            )
        )

        self.assertEqual(cloned.gemini_api_key, "gemini-personal-key")
        self.assertEqual(cloned.gemini_model, "gemini-2.5-flash")
        self.assertIsNone(cloned.openai_api_key)
        self.assertEqual(base.gemini_api_key, "gemini-system-key")

    def test_resolve_default_runtime_llm_prefers_openai_compatible_deepseek(self):
        base = Config(
            openai_api_key="deepseek-system-key",
            openai_base_url="https://api.deepseek.com/v1",
            openai_model="deepseek-chat",
        )

        resolved = base.resolve_default_runtime_llm()

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.provider, "deepseek")
        self.assertEqual(resolved.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(resolved.model, "deepseek-chat")
        self.assertTrue(resolved.has_token)


if __name__ == "__main__":
    unittest.main()
