"""日志重放。

两种用法：
1. 忠实重放（本文件）：模型响应和工具结果都从 trace 里取，把当时的运行原样复现一遍。
   用途：线上出了坏结果，本地不用真模型、不用真下游，就能单步调试当时的每一步，
   并且可以挂上新的断言看"它当时会不会被拦住"。
2. What-if 重放（见 examples/incident_replay.py）：工具结果从 trace 取，模型换成刻画其行为的策略，
   守护层换成新配置，验证"如果当时有这条规则，会在第几步停"。

重放严格模式下会校验每次模型请求的指纹和当时一致，不一致就抛 ReplayDivergence——
说明上下文构建逻辑变了（这本身也是有价值的信号）。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import LLMResponse, Message, SystemPrompt, ToolCall, Usage, system_text
from ..tools.executor import ToolOutcome
from ..tools.registry import ToolRegistry
from .trace import Tracer, stable_hash


class ReplayDivergence(Exception):
    pass


def request_fingerprint(system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> str:
    return stable_hash(
        {"system": system_text(system), "tools": [t["name"] for t in tools],
         "messages": [(m.role, m.content, [(tc.name, tc.args) for tc in m.tool_calls]) for m in messages]}
    )


class ReplayLLM:
    def __init__(self, events: list[dict[str, Any]], strict: bool = False):
        self.responses = [e for e in events if e.get("kind") == "llm.response"]
        self.strict = strict
        self.i = 0

    def complete(self, system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        if self.i >= len(self.responses):
            raise ReplayDivergence(f"重放时第 {self.i + 1} 次模型调用没有对应的记录（trace 里只有 {len(self.responses)} 次）")
        e = self.responses[self.i]
        self.i += 1
        if self.strict and e.get("fingerprint") != request_fingerprint(system, messages, tools):
            raise ReplayDivergence(f"第 {self.i} 次模型请求与记录不一致（上下文构建逻辑可能已变化）")
        u = e.get("usage") or {}
        return LLMResponse(
            text=e.get("text", ""),
            tool_calls=[ToolCall(tc["id"], tc["name"], dict(tc["args"])) for tc in e.get("tool_calls", [])],
            usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cache_read_tokens", 0), u.get("cache_write_tokens", 0)),
            stop_reason=e.get("stop_reason", "end_turn"),
            model="replay",
        )


class RecordedToolExecutor:
    """用 trace 里的 tool.result 事件代替真实工具。接口与 ToolExecutor 一致（registry + execute）。"""

    def __init__(self, events: list[dict[str, Any]], registry: ToolRegistry, tracer: Tracer | None = None):
        self.registry = registry
        self.tracer = tracer or Tracer()
        self.queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        calls = {e["call_id"]: e for e in events if e.get("kind") == "tool.call"}
        for e in events:
            if e.get("kind") == "tool.result" and e.get("call_id") in calls:
                key = f"{e['tool']}:{stable_hash(calls[e['call_id']]['args'])}"
                self.queues[key].append(e)

    def execute(self, call: ToolCall) -> ToolOutcome:
        key = f"{call.name}:{stable_hash(call.args)}"
        if not self.queues[key]:
            raise ReplayDivergence(f"重放时出现记录里没有的工具调用: {call.name} {call.args}")
        e = self.queues[key].popleft()
        p = e.get("payload") or {}
        self.tracer.event("tool.call", tool=call.name, call_id=call.id, args=call.args, replayed=True)
        outcome = ToolOutcome(
            call, ok=bool(p.get("ok")), data=p.get("data"), error=p.get("error"),
            degraded=bool(p.get("degraded")), handoff=bool(p.get("handoff")), cached=bool(p.get("cached")),
            attempts=e.get("attempts", 0), latency_ms=0.0, note=p.get("note", ""),
        )
        self.tracer.event("tool.result", tool=call.name, call_id=call.id, ok=outcome.ok, replayed=True,
                          result_hash=outcome.result_hash(), payload=outcome.envelope())
        return outcome


@dataclass
class ReplayReport:
    turns: int
    matched: bool
    diffs: list[str] = field(default_factory=list)
    llm_calls_recorded: int = 0
    llm_calls_replayed: int = 0


def replay(events: list[dict[str, Any]], agent_factory: Callable[[ReplayLLM, RecordedToolExecutor], Any], registry: ToolRegistry, *, strict: bool = False) -> ReplayReport:
    """agent_factory(llm, executor) 返回一个 Agent；本函数按 trace 里的用户轮次逐轮运行并比对结果。"""
    llm = ReplayLLM(events, strict=strict)
    executor = RecordedToolExecutor(events, registry)
    agent = agent_factory(llm, executor)
    recorded_turns = [e for e in events if e.get("kind") == "turn.start"]
    recorded_ends = {e["turn"]: e for e in events if e.get("kind") == "turn.end"}
    diffs: list[str] = []
    for te in recorded_turns:
        result = agent.run_turn(te["user_text"])
        rec = recorded_ends.get(te["turn"])
        if rec is None:
            diffs.append(f"turn {te['turn']}: 记录里没有 turn.end")
            continue
        if result.text != rec.get("text"):
            diffs.append(f"turn {te['turn']}: 最终回复不同\n  记录: {rec.get('text')}\n  重放: {result.text}")
        if result.rounds != rec.get("rounds"):
            diffs.append(f"turn {te['turn']}: 模型调用次数 记录={rec.get('rounds')} 重放={result.rounds}")
    return ReplayReport(len(recorded_turns), not diffs, diffs, len(llm.responses), llm.i)
