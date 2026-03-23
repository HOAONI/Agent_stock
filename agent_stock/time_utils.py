# -*- coding: utf-8 -*-
"""统一的时间辅助函数。

集中约定几种时间口径：
1. `local_now` 保持与旧 `datetime.now()` 一致的“本机本地时间 + naive datetime”语义。
2. `utc_now` 用于 API/探针这类需要显式 UTC 的时间戳。
3. `shanghai_now` 用于 A 股交易时区相关逻辑。
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def local_now() -> datetime:
    """返回与旧 `datetime.now()` 兼容的本地 naive 时间。"""
    return datetime.now().astimezone().replace(tzinfo=None)


def utc_now() -> datetime:
    """返回显式带 UTC 时区的当前时间。"""
    return datetime.now(UTC)


def shanghai_now() -> datetime:
    """返回显式带 Asia/Shanghai 时区的当前时间。"""
    return datetime.now(SHANGHAI_TZ)
