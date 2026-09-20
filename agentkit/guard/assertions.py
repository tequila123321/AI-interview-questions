"""关键节点断言。

模型的不确定性没法消除，但可以在关键节点用确定性的代码卡住它：
- 调用写操作（退款）之前：订单号必须来自之前工具查到的结果，不能是模型编的；
- 给出最终回复之前：回复不能为空、不能在用了工具的情况下声称"没查到"却又给出具体数字……
- 记忆压缩之后：被标记为关键的事实必须仍然在摘要或长期记忆里。

hard 断言失败 → 中止当前路径（转人工），soft 断言失败 → 只记录，用于复盘。
每条断言结果都进 trace，这样 Bad Case 复盘时能看到"哪一道闸没拦住"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..observability.trace import Tracer

Check = Callable[[dict[str, Any]], bool | str]   # True=通过；False 或字符串=失败（字符串是原因）


@dataclass
class Assertion:
    name: str
    stage: str                  # before_tool | before_final | after_compress
    check: Check
    severity: str = "hard"      # hard | soft


class AssertionFailed(Exception):
    def __init__(self, name: str, detail: str):
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail


@dataclass
class AssertionResult:
    name: str
    stage: str
    passed: bool
    severity: str
    detail: str = ""


@dataclass
class AssertionRunner:
    assertions: list[Assertion] = field(default_factory=list)
    tracer: Tracer | None = None

    def run(self, stage: str, ctx: dict[str, Any]) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        first_hard: AssertionResult | None = None
        for a in self.assertions:
            if a.stage != stage:
                continue
            try:
                verdict = a.check(ctx)
            except Exception as exc:  # noqa: BLE001 - 断言本身出错也算失败
                verdict = f"assertion raised {type(exc).__name__}: {exc}"
            passed = verdict is True
            detail = "" if passed else (verdict if isinstance(verdict, str) else "check returned False")
            r = AssertionResult(a.name, stage, passed, a.severity, detail)
            results.append(r)
            if self.tracer:
                self.tracer.event("guard.assertion", name=a.name, stage=stage, passed=passed, severity=a.severity, detail=detail)
            if not passed and a.severity == "hard" and first_hard is None:
                first_hard = r
        if first_hard is not None:
            raise AssertionFailed(first_hard.name, first_hard.detail)
        return results


# ---------- 常用断言 ----------
def refund_target_must_come_from_lookup() -> Assertion:
    """退款订单号必须出现在本会话里 lookup_order 成功返回过的订单号集合中。"""

    def check(ctx: dict[str, Any]) -> bool | str:
        call = ctx.get("call")
        if call is None or call.name != "refund_order":
            return True
        seen = ctx.get("seen_order_ids", set())
        from ..tools.builtin import normalize_order_id
        oid = normalize_order_id(str(call.args.get("order_id", "")))
        if oid in seen:
            return True
        return f"退款订单号 {oid} 未经 lookup_order 确认，疑似模型编造"

    return Assertion("refund_target_must_come_from_lookup", "before_tool", check, "hard")


def final_answer_not_empty() -> Assertion:
    def check(ctx: dict[str, Any]) -> bool | str:
        text = (ctx.get("text") or "").strip()
        return True if text else "最终回复为空"

    return Assertion("final_answer_not_empty", "before_final", check, "hard")


def pinned_facts_survive_compression() -> Assertion:
    def check(ctx: dict[str, Any]) -> bool | str:
        summary: str = ctx.get("summary", "")
        missing = [v for v in ctx.get("pinned_values", []) if v not in summary]
        return True if not missing else f"压缩后摘要缺少关键信息: {missing}"

    return Assertion("pinned_facts_survive_compression", "after_compress", check, "soft")
