"""诊断："用户第 1 轮说过的信息，为什么第 N 轮模型不知道？"

排查路径是一条固定的流水线，逐段核对 trace：

  用户说过吗?  ──否──▶ never_stated（是用户以为自己说过）
      │是
  写入：抽取到长期记忆了吗?  ──否──▶ not_extracted（抽取规则/LLM 抽取漏了）
      │是
  读取：第 N 轮请求的上下文里有它吗?
      ├─ 在 memory_block 里 ──▶ present_in_context（上下文里有但模型没用：提示词位置/表述/模型问题）
      ├─ 不在，且该事实非 pinned 又没被召回 ──▶ not_recalled（召回排序问题，或该 pinned 未 pinned）
      ├─ 不在，原文轮次被压缩且摘要不含 ──▶ lost_in_compression
      └─ 不在，原文轮次被窗口裁掉且未压缩 ──▶ trimmed_from_window

每种结论对应一个明确的修法。这个函数就是把"排查路径"写成代码，面试时可以直接讲它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnosis:
    stage: str
    verdict: str
    fix: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"[{self.stage}] {self.verdict}", f"修复建议: {self.fix}"]
        for k, v in self.evidence.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


def diagnose_missing_fact(events: list[dict[str, Any]], fact_value: str, at_turn: int, aliases: list[str] | None = None) -> Diagnosis:
    """aliases：同一事实的其他写法（例如用户说 SO-000123，归一化后存的是 SO-123）。"""
    values = [fact_value, *(aliases or [])]

    def _has(text: Any) -> bool:
        return any(v in str(text) for v in values)

    # 1. 用户说过吗
    stated = [e for e in events if e.get("kind") == "turn.start" and _has(e.get("user_text", ""))]
    if not stated:
        return Diagnosis("never_stated", f"trace 里没有任何用户消息包含「{fact_value}」", "先和用户确认；可能是在别的会话/渠道说的（需要跨会话记忆）")
    stated_turn = stated[0]["turn"]

    # 2. 写入侧
    upserts = [e for e in events if e.get("kind") == "memory.fact_upsert" and _has(e.get("value", ""))]
    extracted = bool(upserts)

    # 3. 第 N 轮的上下文
    ctx_events = [e for e in events if e.get("kind") == "context.built" and e.get("turn") == at_turn]
    if not ctx_events:
        return Diagnosis("no_context_event", f"第 {at_turn} 轮没有 context.built 事件", "确认 tracer 已接入 MemoryManager.build_context", {"stated_turn": stated_turn})
    ctx = ctx_events[0]
    in_memory_block = _has(ctx.get("memory_block", ""))
    in_window = stated_turn in (ctx.get("included_turns") or [])

    if in_memory_block or in_window:
        return Diagnosis(
            "present_in_context",
            f"第 {at_turn} 轮请求里包含「{fact_value}」（{'关键信息块' if in_memory_block else '对话原文窗口'}）",
            "问题不在记忆层：检查系统提示词是否要求优先使用「已知关键信息」、信息是否被更近的矛盾信息覆盖、模型是否需要更强的指令；用同一上下文离线重放验证",
            {"stated_turn": stated_turn, "extracted": extracted, "in_memory_block": in_memory_block, "in_window": in_window},
        )

    if not extracted:
        return Diagnosis(
            "not_extracted",
            f"用户在第 {stated_turn} 轮说过，但没有任何 memory.fact_upsert 包含「{fact_value}」，且第 {at_turn} 轮原文已不在窗口内",
            "写入侧漏抽：补充抽取规则或启用 LLM 抽取；把该类信息标记为 pinned",
            {"stated_turn": stated_turn, "dropped_turns": ctx.get("dropped_turns")},
        )

    pinned = any(e.get("pinned") for e in upserts)
    if not pinned:
        return Diagnosis(
            "not_recalled",
            f"事实已写入长期记忆但未 pinned，第 {at_turn} 轮按相关性召回时没有召回它（recalled_keys={ctx.get('recalled_keys')}）",
            "把硬约束类事实标记为 pinned（永远带上）；或改进召回打分（embedding / 关键词权重）",
            {"stated_turn": stated_turn, "upserts": [(e.get("key"), e.get("pinned")) for e in upserts]},
        )

    compress = [e for e in events if e.get("kind") == "memory.compress" and stated_turn in (e.get("dropped_turns") or [])]
    if compress and not _has(compress[-1].get("summary", "")):
        return Diagnosis(
            "lost_in_compression",
            f"第 {stated_turn} 轮在第 {compress[-1].get('turn')} 轮被压缩进摘要，摘要里没有「{fact_value}」",
            "压缩时把 pinned 事实作为 must_keep 传给摘要器，并在压缩后用代码校验补回（ensure_facts_present）",
            {"stated_turn": stated_turn, "compressed_at": compress[-1].get("turn")},
        )

    return Diagnosis(
        "trimmed_from_window",
        f"第 {stated_turn} 轮原文被预算裁剪（dropped_turns={ctx.get('dropped_turns')}），且关键信息块里也没有",
        "pinned 事实不应参与裁剪：检查 build_context 是否把 pinned 事实放进了 memory_block",
        {"stated_turn": stated_turn, "pinned_keys": ctx.get("pinned_keys")},
    )
