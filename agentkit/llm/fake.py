"""确定性的假模型，用于：单元测试、Bad Case 回归、成本/死循环演示、日志重放。

两种用法：
1. 脚本模式：FakeLLM(script=[resp1, resp2, ...])，按顺序返回；
2. 策略模式：FakeLLM(policy=fn)，fn(system, messages, tools) -> LLMResponse，可以模拟
   "看到工具返回 retry 就一直重试" 这类真实模型的坏习惯（见 policies.py）。

usage 会按 tokens.estimate_tokens 估算，因此预算/成本相关逻辑在测试里同样可验证。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..tokens import estimate_json_tokens, estimate_tokens
from .base import LLMResponse, LLMTransientError, Message, SystemPrompt, ToolCall, Usage, system_text

Policy = Callable[[str, list[Message], list[dict[str, Any]]], LLMResponse]

_ids = itertools.count(1)


def new_call_id() -> str:
    return f"call_{next(_ids):04d}"


def reply(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end_turn")


def call(name: str, /, text: str = "", **args: Any) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[ToolCall(new_call_id(), name, args)], stop_reason="tool_use")


def calls(*pairs: tuple[str, dict[str, Any]], text: str = "") -> LLMResponse:
    """并行工具调用。"""
    return LLMResponse(
        text=text,
        tool_calls=[ToolCall(new_call_id(), n, a) for n, a in pairs],
        stop_reason="tool_use",
    )


def estimate_request_tokens(system: str, messages: list[Message], tools: list[dict[str, Any]]) -> int:
    n = estimate_tokens(system) + estimate_json_tokens(tools)
    for m in messages:
        n += estimate_tokens(m.content) + 4
        for tc in m.tool_calls:
            n += estimate_json_tokens(tc.args) + estimate_tokens(tc.name)
    return n


@dataclass
class RecordedRequest:
    system: str
    messages: list[Message]
    tools: list[dict[str, Any]]


@dataclass
class FakeLLM:
    policy: Policy | None = None
    script: list[LLMResponse] | None = None
    model: str = "fake-llm"
    fail_first_n: int = 0                       # 前 n 次调用抛 LLMTransientError，用于测模型层重试
    requests: list[RecordedRequest] = field(default_factory=list)
    calls_made: int = 0

    def complete(self, system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        system = system_text(system)          # 策略只看拼起来的文本
        self.calls_made += 1
        self.requests.append(RecordedRequest(system, list(messages), list(tools)))
        if self.calls_made <= self.fail_first_n:
            raise LLMTransientError("simulated 503 from model provider")

        if self.script is not None:
            if not self.script:
                raise RuntimeError("FakeLLM 脚本已耗尽：Agent 调用模型的次数超过了脚本预期")
            resp = self.script.pop(0)
        elif self.policy is not None:
            resp = self.policy(system, messages, tools)
        else:
            resp = reply("(fake) 没有配置策略")

        # 补齐 usage，让预算逻辑可测
        if resp.usage.input_tokens == 0 and resp.usage.output_tokens == 0:
            out = estimate_tokens(resp.text) + sum(estimate_json_tokens(tc.args) + 8 for tc in resp.tool_calls)
            resp.usage = Usage(input_tokens=estimate_request_tokens(system, messages, tools), output_tokens=out)
        resp.model = self.model
        return resp

    # ---- 测试辅助 ----
    def last_request(self) -> RecordedRequest:
        return self.requests[-1]

    def last_tool_result_payload(self) -> dict[str, Any] | None:
        """最近一次请求里最后一条 tool 消息的 JSON（Agent 喂给模型的工具返回信封）。"""
        for m in reversed(self.last_request().messages):
            if m.role == "tool":
                try:
                    return json.loads(m.content)
                except json.JSONDecodeError:
                    return {"raw": m.content}
        return None
