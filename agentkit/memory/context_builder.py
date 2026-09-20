"""在 token 预算内组装一次模型请求的上下文。

优先级（从高到低，预算不够时从低的开始裁）：
  1. 系统提示词（稳定部分）
  2. 关键事实（pinned + 召回）—— 不参与裁剪
  3. 早先对话摘要
  4. 工作记忆里的对话原文 —— 从最旧的开始裁，且以"完整的一轮"为单位，
     绝不把 assistant 的 tool_calls 和对应的 tool 结果拆开（拆开会导致请求非法或模型困惑）

返回的 report 会记进 trace（context.built），是排查"模型为什么不知道 X"的第一手证据：
X 到底在不在这次请求里？在哪一段？如果不在，是被裁掉了还是根本没召回？

system_blocks 把稳定系统提示和波动的记忆块分成两块交给模型适配器：缓存断点只打在第一块上，
记忆块（召回结果会变）不会让稳定前缀的缓存失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import Message
from ..tokens import estimate_json_tokens, estimate_tokens
from .tiers import Fact


@dataclass
class ContextReport:
    budget_tokens: int
    tokens_system_base: int = 0
    tokens_memory_block: int = 0
    tokens_messages: int = 0
    included_messages: int = 0
    dropped_messages: int = 0
    included_turns: list[int] = field(default_factory=list)
    dropped_turns: list[int] = field(default_factory=list)
    pinned_keys: list[str] = field(default_factory=list)
    recalled_keys: list[str] = field(default_factory=list)
    memory_block: str = ""

    @property
    def total_tokens(self) -> int:
        return self.tokens_system_base + self.tokens_memory_block + self.tokens_messages


@dataclass
class BuiltContext:
    system: str                    # 稳定系统提示 + 记忆块拼成的完整文本（给不区分块的调用方）
    system_blocks: list[str]       # [稳定系统提示, 记忆块]：适配器只在第一块打缓存断点，记忆块变化不会让稳定前缀失效
    messages: list[Message]
    report: ContextReport


def render_memory_block(facts: list[Fact], summary: str) -> str:
    parts = []
    if facts:
        parts.append("## 已知关键信息（来自长期记忆，优先级高于对话摘要）\n" + "\n".join(f"- {f.render()}" for f in facts))
    if summary:
        parts.append("## 早先对话摘要\n" + summary)
    return "\n\n".join(parts)


def message_tokens(m: Message) -> int:
    n = estimate_tokens(m.content) + 4
    for tc in m.tool_calls:
        n += estimate_tokens(tc.name) + estimate_json_tokens(tc.args) + 8
    return n


def build_context(
    base_system: str,
    summary: str,
    facts: list[Fact],
    working: list[Message],
    budget_tokens: int,
    reserve_output_tokens: int = 0,
) -> BuiltContext:
    memory_block = render_memory_block(facts, summary)
    system_blocks = [base_system] + ([memory_block] if memory_block else [])
    system = "\n\n".join(system_blocks)
    report = ContextReport(
        budget_tokens=budget_tokens,
        tokens_system_base=estimate_tokens(base_system),
        tokens_memory_block=estimate_tokens(memory_block),
        pinned_keys=[f.key for f in facts if f.pinned],
        recalled_keys=[f.key for f in facts if not f.pinned],
        memory_block=memory_block,
    )
    remaining = budget_tokens - reserve_output_tokens - report.tokens_system_base - report.tokens_memory_block

    # 从最新往回装
    start = len(working)
    used = 0
    for i in range(len(working) - 1, -1, -1):
        t = message_tokens(working[i])
        if used + t > remaining and i != len(working) - 1:   # 最后一条永远保留
            break
        used += t
        start = i

    # 边界对齐：第一条必须是普通 user 消息（不能以 tool 结果或 assistant 开头）
    plain_user = [i for i, m in enumerate(working) if m.role == "user" and m.tool_call_id is None]
    aligned = [i for i in plain_user if i >= start]
    if aligned:
        start = aligned[0]
    elif plain_user:
        start = plain_user[-1]

    included = working[start:]
    report.tokens_messages = sum(message_tokens(m) for m in included)
    report.included_messages = len(included)
    report.dropped_messages = start
    report.included_turns = sorted({m.meta.get("turn") for m in included if m.meta.get("turn") is not None})
    report.dropped_turns = sorted({m.meta.get("turn") for m in working[:start] if m.meta.get("turn") is not None} - set(report.included_turns))
    return BuiltContext(system=system, system_blocks=system_blocks, messages=included, report=report)
