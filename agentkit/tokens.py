"""Token 估算。

生产环境应使用模型厂商的计数接口（Anthropic: client.messages.count_tokens；
OpenAI: tiktoken），这里只做一个足够稳定的启发式估算，用于：
- 上下文预算裁剪（宁可保守）
- 成本核算的近似
- 测试里的可预测性（不依赖网络）

规则：一个 CJK 字符 ≈ 1 token，其余按 4 个字符 ≈ 1 token。
"""

from __future__ import annotations

import json
import re
from typing import Any

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_json_tokens(obj: Any) -> int:
    try:
        return estimate_tokens(json.dumps(obj, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return estimate_tokens(str(obj))
