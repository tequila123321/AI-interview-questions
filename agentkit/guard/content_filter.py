"""内容安全：输出审核 + 工具入参脱敏。

两条链路：
1. 出向（模型 → 用户）：最终回复先过规则（违禁词、PII），再过可插拔的 LLM 审核器（judge），
   动作分三档：pass / redact（脱敏后放行）/ block（替换为兜底话术）。
2. 入向（模型 → 工具）：模型可能把用户在对话里说过的手机号、卡号原样塞进搜索类工具的参数，
   发给第三方服务。对不需要 PII 的工具做参数脱敏。

规则层便宜、确定、可测；LLM 审核层覆盖语义违规，但有延迟和成本，通常只对最终回复做，
并且审核模型的判定也要记 trace（误杀率要能复盘）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..observability.trace import Tracer

DEFAULT_PII_PATTERNS: dict[str, str] = {
    "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
}

Judge = Callable[[str], tuple[bool, str]]   # (is_safe, reason)


@dataclass
class FilterResult:
    action: str                 # pass | redact | block
    text: str
    reasons: list[str] = field(default_factory=list)


def mask(value: str) -> str:
    if len(value) <= 6:
        return "*" * len(value)
    return value[:3] + "*" * (len(value) - 7) + value[-4:]


@dataclass
class ContentFilter:
    blocked_terms: tuple[str, ...] = ()
    pii_patterns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PII_PATTERNS))
    judge: Judge | None = None
    block_message: str = "抱歉，这条回复包含不适合展示的内容，已转人工处理。"
    pii_free_tools: tuple[str, ...] = ("search_kb",)   # 这些工具的参数里不允许出现 PII
    tracer: Tracer | None = None

    def _emit(self, direction: str, result: FilterResult, **fields: Any) -> FilterResult:
        if self.tracer and result.action != "pass":
            self.tracer.event("guard.content", direction=direction, action=result.action, reasons=result.reasons, **fields)
        return result

    def redact_pii(self, text: str) -> tuple[str, list[str]]:
        reasons: list[str] = []
        for name, pat in self.pii_patterns.items():
            def _sub(m: re.Match[str], _name: str = name) -> str:
                reasons.append(f"pii:{_name}")
                return mask(m.group(0))
            text = re.sub(pat, _sub, text)
        return text, reasons

    def check_output(self, text: str) -> FilterResult:
        for term in self.blocked_terms:
            if term and term in text:
                return self._emit("output", FilterResult("block", self.block_message, [f"blocked_term:{term}"]))
        redacted, reasons = self.redact_pii(text)
        if self.judge is not None:
            safe, why = self.judge(redacted)
            if not safe:
                return self._emit("output", FilterResult("block", self.block_message, reasons + [f"judge:{why}"]))
        if reasons:
            return self._emit("output", FilterResult("redact", redacted, reasons))
        return FilterResult("pass", text)

    def check_tool_args(self, tool_name: str, args: dict[str, Any]) -> tuple[dict[str, Any], FilterResult]:
        """返回（可能脱敏后的参数, 判定）。只对 pii_free_tools 生效。"""
        if tool_name not in self.pii_free_tools:
            return args, FilterResult("pass", "")
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
        redacted, reasons = self.redact_pii(raw)
        if not reasons:
            return args, FilterResult("pass", "")
        return json.loads(redacted), self._emit("tool_args", FilterResult("redact", redacted, reasons), tool=tool_name)
