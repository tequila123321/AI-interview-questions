"""MemoryManager：把三层记忆 + 抽取 + 压缩 + 上下文构建串起来，给 Agent 用。

一轮对话里的调用顺序：
    turn = mm.start_turn(user_text)      # 抽取事实 → 长期记忆；user 消息进工作记忆
    ctx = mm.build_context(...)          # 召回 + 摘要 + 工作窗口 → (system, messages)
    mm.append(assistant / tool 消息)     # 主循环里产生的消息
    mm.end_turn()                        # 工作记忆超过高水位 → 压缩旧轮次进摘要

"先走记忆检索还是直接调 API"这个问题的答案就在 build_context 里：每次调模型之前，
先用当前用户输入去长期记忆召回，再把 pinned 事实、召回事实、摘要、工作窗口装进预算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import Message
from ..observability.trace import Tracer
from ..tokens import estimate_tokens
from .compressor import ExtractiveSummarizer, Summarizer, ensure_facts_present
from .context_builder import BuiltContext, build_context, message_tokens
from .extractor import FactExtractor, RuleFactExtractor
from .tiers import Fact, LongTermMemory


@dataclass
class MemoryConfig:
    working_high_water_tokens: int = 3000   # 工作记忆超过这个 token 数就压缩
    keep_recent_turns: int = 4              # 压缩时保留最近 N 轮原文
    summary_max_tokens: int = 400
    recall_k: int = 5
    pin_facts_in_summary: bool = True       # 压缩时把 pinned 事实作为 must_keep 传给摘要器


@dataclass
class CompressReport:
    turn: int
    dropped_turns: list[int]
    summary: str
    missing_fixed: list[str]


@dataclass
class MemoryManager:
    extractor: FactExtractor = field(default_factory=RuleFactExtractor)
    summarizer: Summarizer = field(default_factory=ExtractiveSummarizer)
    config: MemoryConfig = field(default_factory=MemoryConfig)
    tracer: Tracer | None = None
    working: list[Message] = field(default_factory=list)
    summary: str = ""
    ltm: LongTermMemory = field(default_factory=LongTermMemory)
    turn: int = 0
    last_compress: CompressReport | None = None

    # ---- 写入 ----
    def start_turn(self, user_text: str) -> int:
        self.turn += 1
        for f in self.extractor.extract(user_text, self.turn):
            stored = self.ltm.upsert(f)
            if self.tracer:
                self.tracer.event("memory.fact_upsert", turn=self.turn, key=stored.key, value=stored.value, pinned=stored.pinned)
        self.working.append(Message.user(user_text, turn=self.turn))
        return self.turn

    def append(self, msg: Message) -> None:
        msg.meta.setdefault("turn", self.turn)
        self.working.append(msg)

    def working_tokens(self) -> int:
        return sum(message_tokens(m) for m in self.working)

    def end_turn(self) -> CompressReport | None:
        return self.maybe_compress()

    # ---- 压缩 ----
    def maybe_compress(self, force: bool = False) -> CompressReport | None:
        if not force and self.working_tokens() <= self.config.working_high_water_tokens:
            return None
        cutoff = self.turn - self.config.keep_recent_turns + 1
        old = [m for m in self.working if m.meta.get("turn", 0) < cutoff]
        if not old:
            return None
        must_keep = [f.value for f in self.ltm.pinned()] if self.config.pin_facts_in_summary else []
        summary = self.summarizer.summarize(self.summary, old, must_keep, self.config.summary_max_tokens)
        summary, missing = ensure_facts_present(summary, must_keep)   # 代码兜底：摘要器没保住的补回去
        dropped_turns = sorted({m.meta.get("turn") for m in old})
        self.summary = summary
        self.working = [m for m in self.working if m.meta.get("turn", 0) >= cutoff]
        report = CompressReport(self.turn, dropped_turns, summary, missing)
        self.last_compress = report
        if self.tracer:
            self.tracer.event(
                "memory.compress", turn=self.turn, dropped_turns=dropped_turns,
                summary_tokens=estimate_tokens(summary), summary=summary, missing_fixed=missing,
            )
        return report

    # ---- 读取 ----
    def select_facts(self, query: str) -> list[Fact]:
        return self.ltm.recall(query, k=self.config.recall_k, current_turn=self.turn)

    def build_context(self, base_system: str, query: str, budget_tokens: int, reserve_output_tokens: int = 0, step: int = 0) -> BuiltContext:
        facts = self.select_facts(query)
        ctx = build_context(base_system, self.summary, facts, self.working, budget_tokens, reserve_output_tokens)
        if self.tracer:
            r = ctx.report
            self.tracer.event(
                "context.built", turn=self.turn, step=step, total_tokens=r.total_tokens, budget_tokens=r.budget_tokens,
                tokens_system_base=r.tokens_system_base, tokens_memory_block=r.tokens_memory_block,
                tokens_messages=r.tokens_messages, included_messages=r.included_messages, dropped_messages=r.dropped_messages,
                included_turns=r.included_turns, dropped_turns=r.dropped_turns, pinned_keys=r.pinned_keys,
                recalled_keys=r.recalled_keys, memory_block=r.memory_block,
            )
        return ctx

    def current_user_text(self) -> str:
        for m in reversed(self.working):
            if m.role == "user" and m.tool_call_id is None:
                return m.content
        return ""
