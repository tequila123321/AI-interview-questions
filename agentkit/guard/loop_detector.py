"""死循环检测。

只设"最大步数"是不够的：max_rounds=30 意味着模型可以把同一个失败调用重复 30 次，
而且每一步的上下文都比上一步长（工具结果不断堆进去），token 消耗是二次方增长——
预算烧完了才停。所以要做三种更早的检测：

1. 同工具同参数连续调用：连续第 2 次 → 注入提醒（warn），第 3 次 → 强制收尾（stop）
2. 同工具同参数累计调用：A-B-A-B 这种交替也算（total）
3. 无进展：连续 N 次工具结果哈希完全相同（不同调用但结果一样，也是在原地打转）

判定动作：
  ok    正常执行
  warn  仍然执行，但把提醒写进工具返回信封（guard_note），让模型知道"这条路走不通"
  stop  不执行；返回 loop_detected 错误并让 Agent 进入"强制收尾"（下一次模型调用不给工具）

"相同参数"以 canonical 键（工具名 + 参数排序后哈希）判断，避免字典顺序不同被当成不同调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import ToolCall
from ..observability.trace import Tracer, stable_hash


@dataclass
class LoopPolicy:
    max_rounds: int = 10              # 一轮用户对话内最多几次模型调用
    same_call_warn_at: int = 2        # 连续第 N 次相同调用 → warn
    same_call_stop_at: int = 3        # 连续第 N 次相同调用 → stop
    same_call_total_stop_at: int = 4  # 累计第 N 次相同调用 → stop（防交替）
    no_progress_stop_at: int = 4      # 连续 N 次结果哈希相同 → stop


@dataclass
class LoopVerdict:
    action: str = "ok"                # ok | warn | stop
    reason: str = ""
    hint: str = ""                    # 注入给模型的话

    @property
    def stopped(self) -> bool:
        return self.action == "stop"


def call_key(call: ToolCall) -> str:
    return f"{call.name}:{stable_hash(call.args)}"


@dataclass
class LoopDetector:
    policy: LoopPolicy = field(default_factory=LoopPolicy)
    tracer: Tracer | None = None
    rounds: int = 0
    calls: list[str] = field(default_factory=list)          # canonical keys，按执行顺序
    result_hashes: list[str] = field(default_factory=list)

    def _emit(self, verdict: LoopVerdict, **fields) -> LoopVerdict:
        if self.tracer and verdict.action != "ok":
            self.tracer.event("guard.loop", action=verdict.action, reason=verdict.reason, **fields)
        return verdict

    def on_round(self) -> LoopVerdict:
        """每次调用模型前调用。"""
        self.rounds += 1
        if self.rounds > self.policy.max_rounds:
            return self._emit(
                LoopVerdict("stop", f"超过最大步数 {self.policy.max_rounds}", "已达到本轮最大步数，请直接给出目前能给的答案或说明需要人工处理"),
                rounds=self.rounds,
            )
        return LoopVerdict()

    def observe(self, call: ToolCall) -> LoopVerdict:
        """每次执行工具前调用。"""
        key = call_key(call)
        consecutive = 1
        for prev in reversed(self.calls):
            if prev == key:
                consecutive += 1
            else:
                break
        total = self.calls.count(key) + 1
        p = self.policy

        if consecutive >= p.same_call_stop_at:
            return self._emit(
                LoopVerdict(
                    "stop",
                    f"连续第 {consecutive} 次用相同参数调用 {call.name}",
                    f"你已经连续 {consecutive} 次用完全相同的参数调用 {call.name}，结果不会改变。停止重试，"
                    "根据已有信息回答用户；如果信息不足，说明原因并请用户补充或转人工。",
                ),
                tool=call.name, consecutive=consecutive, total=total,
            )
        if total >= p.same_call_total_stop_at:
            return self._emit(
                LoopVerdict(
                    "stop",
                    f"累计第 {total} 次用相同参数调用 {call.name}",
                    f"你已经多次用相同参数调用 {call.name}（含交替调用），说明当前路径无效。请直接回答或转人工。",
                ),
                tool=call.name, consecutive=consecutive, total=total,
            )
        if len(self.result_hashes) >= p.no_progress_stop_at and len(set(self.result_hashes[-p.no_progress_stop_at:])) == 1:
            return self._emit(
                LoopVerdict(
                    "stop",
                    f"连续 {p.no_progress_stop_at} 次工具结果完全相同，无进展",
                    "最近几次工具调用的结果完全相同，继续调用没有意义。请基于现有信息回答。",
                ),
                tool=call.name,
            )
        if consecutive >= p.same_call_warn_at:
            return self._emit(
                LoopVerdict(
                    "warn",
                    f"连续第 {consecutive} 次用相同参数调用 {call.name}",
                    f"注意：这是你第 {consecutive} 次用相同参数调用 {call.name}。如果结果仍然相同，"
                    "不要再重复，换参数、换工具或直接回答。",
                ),
                tool=call.name, consecutive=consecutive, total=total,
            )
        return LoopVerdict()

    def record(self, call: ToolCall, result_hash: str) -> None:
        self.calls.append(call_key(call))
        self.result_hashes.append(result_hash)
