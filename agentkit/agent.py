"""Agent 主循环：把工具层、记忆层、守护层、可观测性接成一个可运行的整体。

一轮用户对话（run_turn）的骨架：

    start_turn：抽取事实 → 长期记忆；预算计数清零；新建本轮的死循环检测器
    loop:
        on_round（步数上限）
        build_context（召回 + 摘要 + 窗口，在预算内）
        call_llm（模型层重试：429/5xx 指数退避；4xx 直接失败 → 转人工）
        budget.charge（stop → 转人工；wrap_up → 下一轮进入收尾模式：不给工具）
        无工具调用 → before_final 断言 → 内容审核 → 结束
        有工具调用 → 逐个：
            loop_detector.observe（warn：注入提醒；stop：不执行，进入收尾模式）
            before_tool 断言（hard 失败 → 转人工）
            入参脱敏 → executor.execute（校验/重试/熔断/降级都在里面）
            handoff → 转人工
            结果信封写回消息
    end_turn：工作记忆超高水位 → 压缩；after_compress 断言

所有分支都有 trace 事件，所以任何一次线上的坏结果都能拉出完整时间线并离线重放。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .guard.assertions import AssertionFailed, AssertionRunner
from .guard.budget import Budget
from .guard.content_filter import ContentFilter
from .guard.loop_detector import LoopDetector, LoopPolicy
from .llm.base import LLMClient, LLMPermanentError, LLMResponse, LLMTransientError, Message, SystemPrompt, ToolCall, Usage
from .memory.manager import MemoryManager
from .observability.replay import request_fingerprint
from .observability.trace import Tracer
from .tools.executor import ToolExecutor, ToolOutcome
from .tools.retry import RetryPolicy, run_with_retry

DEFAULT_SYSTEM = """你是电商客服助手。规则：
1. 优先使用「已知关键信息」里的事实，不要向用户重复询问已经知道的信息。
2. 需要查询时调用工具；工具返回 ok=false 且 retryable=false 时不要重复调用，按 hint 处理。
3. 信息不足或工具不可用时如实告知用户，不要编造订单号、金额或状态。
4. 涉及退款等写操作，订单号必须来自查询结果。"""


@dataclass
class AgentConfig:
    system_prompt: str = DEFAULT_SYSTEM
    context_budget_tokens: int = 8000
    reserve_output_tokens: int = 1000
    llm_retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(max_attempts=3, base_delay_s=0.05, max_delay_s=0.5))
    handoff_message: str = "这个问题我暂时无法自动处理，已为您转接人工客服，请稍候。"
    wrap_up_instruction: str = (
        "【系统】本轮已达到工具调用或预算上限，不能再调用工具。"
        "请直接根据已有信息回答用户；信息不足时如实说明并给出下一步建议。"
    )
    refusal_message: str = "抱歉，这个请求我无法处理。"


@dataclass
class TurnResult:
    turn: int
    text: str
    rounds: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    handoff: bool = False
    degraded: bool = False
    stop_reason: str = "end_turn"      # end_turn | handoff | loop_stop | budget_stop | refusal | llm_error
    usage: Usage = field(default_factory=Usage)
    guard_events: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        memory: MemoryManager,
        config: AgentConfig | None = None,
        *,
        loop_policy: LoopPolicy | None = None,
        content_filter: ContentFilter | None = None,
        budget: Budget | None = None,
        assertions: AssertionRunner | None = None,
        tracer: Tracer | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.llm = llm
        self.executor = executor
        self.memory = memory
        self.config = config or AgentConfig()
        self.loop_policy = loop_policy or LoopPolicy()
        self.content_filter = content_filter or ContentFilter()
        self.budget = budget or Budget()
        self.assertions = assertions or AssertionRunner()
        self.tracer = tracer or Tracer()
        self.sleep = sleep
        self.seen_order_ids: set[str] = set()     # 工具查到过的订单号（给写操作断言用）
        # 让各组件共用同一个 tracer
        for comp in (self.memory, self.content_filter, self.budget, self.assertions):
            if getattr(comp, "tracer", None) is None:
                comp.tracer = self.tracer  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ 主循环
    def run_turn(self, user_text: str) -> TurnResult:
        cfg = self.config
        self.tracer.bind(turn=None, step=None)
        turn = self.memory.start_turn(user_text)
        self.tracer.bind(turn=turn)
        self.tracer.event("turn.start", turn=turn, user_text=user_text)
        self.budget.start_turn()
        detector = LoopDetector(self.loop_policy, self.tracer)

        result = TurnResult(turn=turn, text="")
        force_final = False

        while True:
            rv = detector.on_round()
            if rv.stopped:
                if force_final:                       # 收尾模式下模型还在调工具：不再给机会
                    return self._finish_handoff(result, "loop_stop", rv.reason)
                force_final = True
                result.stop_reason = "loop_stop"
                result.guard_events.append(rv.reason)
            step = detector.rounds
            self.tracer.bind(step=step)

            ctx = self.memory.build_context(cfg.system_prompt, user_text, cfg.context_budget_tokens, cfg.reserve_output_tokens, step)
            messages = list(ctx.messages)
            tools: list[dict[str, Any]] = [] if force_final else self.executor.registry.schemas()
            if force_final:
                messages.append(Message.user(cfg.wrap_up_instruction))

            resp = self._call_llm(ctx.system_blocks, messages, tools)
            if resp is None:
                return self._finish_handoff(result, "llm_error", "模型服务持续不可用")
            result.rounds = step
            result.usage = result.usage.add(resp.usage)

            bs = self.budget.charge(resp.usage)
            if bs.action != "ok":
                result.guard_events.append(bs.reason)
            if bs.action == "stop" and resp.tool_calls:          # 预算到顶且模型还想调工具：不再继续
                return self._finish_handoff(result, "budget_stop", bs.reason)
            if bs.action == "wrap_up" and not force_final:        # 接近上限：下一轮不给工具，要求收尾
                force_final = True
                if result.stop_reason == "end_turn":
                    result.stop_reason = "budget_stop"

            if resp.stop_reason == "refusal":
                result.stop_reason = "refusal"
                return self._finish_text(result, cfg.refusal_message)

            if not resp.tool_calls:
                return self._finish_text(result, resp.text)

            if force_final:                           # 已经要求收尾，模型仍然要调工具
                return self._finish_handoff(result, result.stop_reason if result.stop_reason != "end_turn" else "loop_stop", "收尾模式下模型仍尝试调用工具")

            # ---- 执行工具 ----
            self.memory.append(Message.assistant(resp.text, resp.tool_calls))
            for call in resp.tool_calls:
                result.tool_calls.append(call)
                verdict = detector.observe(call)
                if verdict.stopped:
                    payload = json.dumps(
                        {"ok": False, "error": {"kind": "loop_detected", "message": verdict.reason, "retryable": False, "hint": verdict.hint}},
                        ensure_ascii=False,
                    )
                    self.memory.append(Message.tool(call, payload, is_error=True))
                    result.guard_events.append(verdict.reason)
                    result.stop_reason = "loop_stop"
                    force_final = True
                    continue

                try:
                    self.assertions.run("before_tool", {"call": call, "seen_order_ids": self.seen_order_ids})
                except AssertionFailed as exc:
                    payload = json.dumps(
                        {"ok": False, "error": {"kind": "assertion_failed", "message": exc.detail, "retryable": False}, "handoff": True},
                        ensure_ascii=False,
                    )
                    self.memory.append(Message.tool(call, payload, is_error=True))
                    result.guard_events.append(f"assertion:{exc.name}")
                    return self._finish_handoff(result, "handoff", exc.detail)

                safe_args, fr = self.content_filter.check_tool_args(call.name, call.args)
                if fr.action != "pass":
                    result.guard_events.append(f"tool_args_{fr.action}:{call.name}")
                    call = ToolCall(call.id, call.name, safe_args)

                outcome = self.executor.execute(call)
                detector.record(call, outcome.result_hash())
                self._observe_outcome(call, outcome)
                result.degraded = result.degraded or outcome.degraded
                guard_note = verdict.hint if verdict.action == "warn" else None
                if guard_note:
                    result.guard_events.append(verdict.reason)
                self.memory.append(Message.tool(call, outcome.to_model_payload(guard_note), is_error=not outcome.ok))
                if outcome.handoff:
                    return self._finish_handoff(result, "handoff", (outcome.error or {}).get("message", "工具失败"))

    # ------------------------------------------------------------------ 模型调用
    def _call_llm(self, system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse | None:
        fingerprint = request_fingerprint(system, messages, tools)   # 与重放用同一个函数，避免两边算法漂移
        self.tracer.event("llm.request", fingerprint=fingerprint, n_messages=len(messages), n_tools=len(tools))

        def attempt() -> LLMResponse:
            return self.llm.complete(system, messages, tools)

        try:
            with self.tracer.span("llm.response", fingerprint=fingerprint) as extra:
                resp = run_with_retry(
                    attempt, self.config.llm_retry, sleep=self.sleep,
                    on_retry=lambda exc, n, d: self.tracer.event("llm.retry", attempt=n, delay_s=round(d, 3), error=str(exc)),
                )
                extra.update(
                    text=resp.text, stop_reason=resp.stop_reason, model=resp.model,
                    tool_calls=[{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                    usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                           "cache_read_tokens": resp.usage.cache_read_tokens, "cache_write_tokens": resp.usage.cache_write_tokens},
                )
                return resp
        except (LLMTransientError, LLMPermanentError) as exc:
            self.tracer.event("llm.failed", error=str(exc), retryable=exc.retryable)
            return None

    # ------------------------------------------------------------------ 收尾
    def _observe_outcome(self, call: ToolCall, outcome: ToolOutcome) -> None:
        if outcome.ok and isinstance(outcome.data, dict) and "order_id" in outcome.data and not outcome.degraded:
            self.seen_order_ids.add(str(outcome.data["order_id"]))

    def _finish_text(self, result: TurnResult, text: str) -> TurnResult:
        try:
            self.assertions.run("before_final", {"text": text, "tool_calls": result.tool_calls})
        except AssertionFailed as exc:
            return self._finish_handoff(result, "handoff", exc.detail)
        fr = self.content_filter.check_output(text)
        if fr.action != "pass":
            result.guard_events.append(f"output_{fr.action}")
        result.text = fr.text
        return self._end(result)

    def _finish_handoff(self, result: TurnResult, stop_reason: str, reason: str) -> TurnResult:
        result.handoff = True
        result.stop_reason = stop_reason
        result.text = self.config.handoff_message
        self.tracer.event("handoff", reason=reason, stop_reason=stop_reason)
        return self._end(result)

    def _end(self, result: TurnResult) -> TurnResult:
        self.memory.append(Message.assistant(result.text))
        report = self.memory.end_turn()
        if report is not None:
            try:
                self.assertions.run("after_compress", {"summary": report.summary, "pinned_values": [f.value for f in self.memory.ltm.pinned()]})
            except AssertionFailed as exc:   # after_compress 断言默认是 soft；hard 失败也只记录，不影响已生成的回复
                result.guard_events.append(f"assertion:{exc.name}")
        self.tracer.event(
            "turn.end", turn=result.turn, rounds=result.rounds, stop_reason=result.stop_reason, handoff=result.handoff,
            degraded=result.degraded, tokens=result.usage.total, text=result.text, guard_events=result.guard_events,
        )
        self.tracer.bind(step=None)
        return result
