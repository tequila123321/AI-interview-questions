"""给 FakeLLM 用的"模型行为策略"。

真实模型是不确定的，但它的坏习惯是可以被刻画的。把这些坏习惯写成确定性策略，
守护层的每一条规则就都能有一个可重复的测试：

- customer_service_policy（还算靠谱的模型）：看 hint、看 retryable、看 guard_note，错误可重试时最多再试一次。
- stubborn_policy（事故里的模型）：只要工具返回里出现 "retry" 或 ok=false，就用同样的参数再来一次，永不放弃。
  这就是只有 max_rounds=30 时把预算烧光的那个行为。

策略从上下文里找信息的方式也刻意贴近真实模型：先看系统提示里的「已知关键信息」，再看窗口内的对话原文。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import LLMResponse, Message
from .fake import call, reply

_ORDER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,4}-?0*\d{3,})(?![A-Za-z0-9])")


def _last(messages: list[Message]) -> Message | None:
    return messages[-1] if messages else None


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.tool_call_id is None and not m.content.startswith("【系统】"):
            return m.content
    return ""


def _parse_env(m: Message) -> dict[str, Any]:
    try:
        return json.loads(m.content)
    except (json.JSONDecodeError, TypeError):
        return {"raw": m.content}


def _find_order_id(system: str, messages: list[Message]) -> str | None:
    """先看当前用户消息，再看系统提示里的关键信息块，最后看窗口内更早的用户消息。"""
    m = _ORDER_RE.search(_last_user_text(messages))
    if m:
        return m.group(1)
    km = re.search(r"订单号(?:#\d+)?: (\S+?)（", system)
    if km:
        return km.group(1)
    for msg in reversed(messages):
        if msg.role == "user" and msg.tool_call_id is None:
            m = _ORDER_RE.search(msg.content)
            if m:
                return m.group(1)
    return None


def _find_fact(system: str, messages: list[Message], key: str, text_pattern: str) -> str | None:
    km = re.search(rf"{key}: (\S+?)（", system)
    if km:
        return km.group(1)
    for msg in reversed(messages):
        if msg.role == "user" and msg.tool_call_id is None:
            m = re.search(text_pattern, msg.content)
            if m:
                return m.group(1)
    return None


def _last_assistant_call(messages: list[Message]) -> tuple[str, dict[str, Any]] | None:
    for m in reversed(messages):
        if m.role == "assistant" and m.tool_calls:
            tc = m.tool_calls[-1]
            return tc.name, tc.args
        if m.role == "user" and m.tool_call_id is None and not m.content.startswith("【系统】"):
            return None
    return None


def _count_same_call_streak(messages: list[Message], name: str, args: dict[str, Any]) -> int:
    n = 0
    for m in reversed(messages):
        if m.role == "assistant" and m.tool_calls:
            tc = m.tool_calls[-1]
            if tc.name == name and tc.args == args:
                n += 1
            else:
                break
        elif m.role == "user" and m.tool_call_id is None and not m.content.startswith("【系统】"):
            break
    return n


def _answer_from_tool(name: str, env: dict[str, Any]) -> str:
    data = env.get("data") or {}
    note = " （注意：这是降级默认值，信息可能不完整）" if env.get("degraded") else ""
    if name == "lookup_order":
        if "order_id" in data:
            return f"您的订单 {data['order_id']} 当前状态：{data.get('status')}，金额 {data.get('amount')} {data.get('currency', '')}。{note}"
        return f"订单服务返回：{json.dumps(data, ensure_ascii=False)}{note}"
    if name == "get_weather":
        return f"{data.get('city')} 天气：{data.get('condition')}，{data.get('temp_c')}°C。{note}"
    if name == "search_kb":
        hits = data.get("hits") or []
        return ("相关政策：" + "；".join(h.get("snippet", "") for h in hits)) if hits else f"知识库暂时没有找到相关内容。{note}"
    if name == "refund_order":
        return f"退款已提交：{data.get('refund_id')}，金额 {data.get('amount')}。"
    return f"工具 {name} 返回：{json.dumps(data, ensure_ascii=False)}"


_LOOKUP_INTENT = ("查", "到哪", "状态", "物流", "进度", "发货")


def make_policy(*, max_self_retries: int = 1, obey_hints: bool = True, retry_on_text_hint: bool = False, skip_lookup_before_refund: bool = False):
    """构造一个策略。

    max_self_retries          工具返回可重试错误时，模型自己最多再试几次
    obey_hints                看不看 retryable=false / guard_note / 收尾指令
    retry_on_text_hint        工具 data 里只要出现 "retry" 字样就重试（事故里 v1 工具透传上游文案时的行为）
    skip_lookup_before_refund 用户一说退款就直接调 refund_order，订单号和金额从用户话里拿（"编造"写操作参数）
    """

    def policy(system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        last = _last(messages)
        tool_names = {t["name"] for t in tools}
        user_text = _last_user_text(messages)

        # 收尾指令：没有工具可用，直接答
        if last is not None and last.role == "user" and last.content.startswith("【系统】"):
            return reply("抱歉，目前无法完成这次查询。您可以稍后再试，或者我为您转接人工客服。")

        # 上一条是工具返回
        if last is not None and last.role == "tool":
            env = _parse_env(last)
            prev = _last_assistant_call(messages)
            name, args = prev if prev else (last.tool_name or "", {})
            if obey_hints and env.get("guard_note"):
                return reply(f"抱歉，多次尝试后仍未成功：{(env.get('error') or {}).get('message', '')}。请您核对信息后再试或转人工。")
            if env.get("ok"):
                data = env.get("data") or {}
                if retry_on_text_hint and "retry" in json.dumps(data, ensure_ascii=False).lower() and name in tool_names:
                    return call(name, text="让我再试一次。", **args)
                return reply(_answer_from_tool(name, env))
            err = env.get("error") or {}
            retryable = bool(err.get("retryable")) or not obey_hints
            streak = _count_same_call_streak(messages, name, args)
            if retryable and name in tool_names and (not obey_hints or streak <= max_self_retries):
                return call(name, text="让我再试一次。", **args)
            if err.get("kind") == "invalid_args" and name == "lookup_order" and name in tool_names:
                oid = _find_order_id(system, messages) or "SO-123"
                return call(name, order_id=oid)
            hint = err.get("hint") or "请稍后再试"
            return reply(f"抱歉，{err.get('message', '操作失败')}。{hint}")

        # 普通用户消息：按意图路由（"退款政策"是知识库问题，不是退款请求）
        if ("政策" in user_text or "多久" in user_text) and "search_kb" in tool_names:
            return call("search_kb", query=user_text[:30])
        if "退款" in user_text and "refund_order" in tool_names:
            oid = _find_order_id(system, messages)
            if oid is None:
                return reply("请提供需要退款的订单号。")
            if skip_lookup_before_refund:
                am = re.search(r"(\d+(?:\.\d+)?)\s*元", user_text)
                return call("refund_order", order_id=oid, amount=float(am.group(1)) if am else 1.0, idempotency_key=f"refund-{oid}")
            # 找之前 lookup 的金额；没查过就先查
            for m in reversed(messages):
                if m.role == "tool" and m.tool_name == "lookup_order":
                    env = _parse_env(m)
                    if env.get("ok") and env.get("data", {}).get("order_id"):
                        d = env["data"]
                        return call("refund_order", order_id=d["order_id"], amount=float(d["amount"]), idempotency_key=f"refund-{d['order_id']}")
            return call("lookup_order", order_id=oid)
        if "订单号是多少" in user_text or "我的订单号是什么" in user_text:
            oid = _find_order_id(system, messages[:-1]) or _find_fact(system, messages[:-1], "订单号", r"([A-Z]{2}-?\d{3,})")
            return reply(f"您的订单号是 {oid}。" if oid else "抱歉，我不记得您的订单号了，请再告诉我一次。")
        if "过敏" in user_text and ("什么" in user_text or "哪些" in user_text):
            a = _find_fact(system, messages[:-1], "过敏", r"对(.{1,10}?)过敏")
            return reply(f"您对{a}过敏。" if a else "抱歉，我没有您过敏信息的记录。")
        if any(k in user_text for k in _LOOKUP_INTENT) and ("订单" in user_text or _ORDER_RE.search(user_text)) and "lookup_order" in tool_names:
            oid = _find_order_id(system, messages)
            if oid is None:
                return reply("请提供订单号，我帮您查询。")
            return call("lookup_order", order_id=oid)
        if "天气" in user_text and "get_weather" in tool_names:
            m = re.search(r"([一-鿿]{2,6}?)(?:的)?天气", user_text)
            return call("get_weather", city=m.group(1) if m else "上海")
        return reply("好的，我记下了。还有什么可以帮您？")

    return policy


customer_service_policy = make_policy(max_self_retries=1, obey_hints=True)
stubborn_policy = make_policy(max_self_retries=999, obey_hints=False, retry_on_text_hint=True)
reckless_policy = make_policy(max_self_retries=1, obey_hints=True, skip_lookup_before_refund=True)

POLICIES = {
    "customer_service": customer_service_policy,
    "stubborn": stubborn_policy,
    "reckless": reckless_policy,
}
