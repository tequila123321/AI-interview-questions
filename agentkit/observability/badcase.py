"""Bad Case 闭环。

线上每一个坏结果 → 抽象成一个 Bad Case 文件（badcases/*.json）→ 变成一条永远跑的回归测试。
Bad Case 描述三件事：
  1. 当时模型的行为（policy）和当时下游的故障（tool_faults）——复现条件
  2. 用户说了什么（user_turns）
  3. 修复后系统必须满足什么（expect）：最多几步、同参数最多重复几次、最多多少 token、必须/不得说什么、是否必须转人工

这样"修好了"不是一句话，而是一个断言；下次有人改守护层参数，回归会告诉他把哪个事故的坑又挖开了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent import Agent, TurnResult
from ..guard.loop_detector import call_key
from ..llm.base import ToolCall


@dataclass
class BadCase:
    id: str
    title: str
    origin: str = ""                       # 事故编号 / 链接
    profile: str = "guarded"
    policy: str = "customer_service"
    tool_faults: dict[str, list[str]] = field(default_factory=dict)
    user_turns: list[str] = field(default_factory=list)
    expect: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "BadCase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return BadCase(**data)

    @staticmethod
    def load_dir(directory: str | Path) -> list["BadCase"]:
        return [BadCase.load(p) for p in sorted(Path(directory).glob("*.json"))]


@dataclass
class BadCaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _max_same_call_repeats(calls: list[ToolCall]) -> int:
    counts: dict[str, int] = {}
    for c in calls:
        k = call_key(c)
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values(), default=0)


def evaluate(case: BadCase, results: list[TurnResult]) -> BadCaseResult:
    last = results[-1]
    all_calls = [c for r in results for c in r.tool_calls]
    metrics = {
        "rounds": sum(r.rounds for r in results),
        "max_same_call_repeats": _max_same_call_repeats(all_calls),
        "total_tokens": sum(r.usage.total for r in results),
        "handoff": last.handoff,
        "final_text": last.text,
        "stop_reason": last.stop_reason,
    }
    e = case.expect
    failures: list[str] = []
    if "max_rounds" in e and metrics["rounds"] > e["max_rounds"]:
        failures.append(f"rounds {metrics['rounds']} > {e['max_rounds']}")
    if "max_same_call_repeats" in e and metrics["max_same_call_repeats"] > e["max_same_call_repeats"]:
        failures.append(f"same-call repeats {metrics['max_same_call_repeats']} > {e['max_same_call_repeats']}")
    if "max_total_tokens" in e and metrics["total_tokens"] > e["max_total_tokens"]:
        failures.append(f"tokens {metrics['total_tokens']} > {e['max_total_tokens']}")
    if "final_must_contain_any" in e and not any(s in last.text for s in e["final_must_contain_any"]):
        failures.append(f"final text lacks any of {e['final_must_contain_any']}: {last.text!r}")
    if "final_must_not_contain" in e and any(s in last.text for s in e["final_must_not_contain"]):
        failures.append(f"final text contains forbidden text: {last.text!r}")
    if "must_handoff" in e and bool(e["must_handoff"]) != last.handoff:
        failures.append(f"handoff={last.handoff}, expected {e['must_handoff']}")
    return BadCaseResult(case.id, not failures, failures, metrics)


def run_badcase(case: BadCase, agent_factory: Callable[[BadCase], Agent]) -> BadCaseResult:
    agent = agent_factory(case)
    results = [agent.run_turn(t) for t in case.user_turns]
    return evaluate(case, results)
