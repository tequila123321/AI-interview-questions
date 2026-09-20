"""工具层：参数校验 / 超时重试 / 永久错误 / 契约校验 / 降级 / 熔断 / 缓存 / 非幂等不重试。"""

import time

from agentkit.llm.base import ToolCall
from agentkit.observability.trace import Tracer
from agentkit.tools.builtin import FaultInjector, make_tools
from agentkit.tools.executor import ToolExecutor
from agentkit.tools.retry import CircuitBreaker


def _executor(faults=None, version=2, breaker=None):
    tracer = Tracer()
    reg = make_tools(faults, incident_version=version)
    ex = ToolExecutor(reg, tracer, sleep=lambda s: None, breaker_factory=breaker)
    return ex, tracer


def test_invalid_args_returns_field_level_errors_for_model_to_fix():
    ex, _ = _executor()
    out = ex.execute(ToolCall("c1", "lookup_order", {}))
    assert out.ok is False
    assert out.error["kind"] == "invalid_args"
    assert out.error["retryable"] is False
    assert out.error["details"][0]["field"] == "order_id"
    assert "hint" in out.error


def test_unknown_tool_lists_available_tools():
    ex, _ = _executor()
    out = ex.execute(ToolCall("c1", "nope", {}))
    assert out.error["kind"] == "unknown_tool"
    assert "lookup_order" in out.error["hint"]


def test_timeout_is_retried_with_backoff_then_succeeds():
    f = FaultInjector()
    f.plan("lookup_order", "timeout", "timeout", "ok")
    ex, tracer = _executor(f)
    out = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    assert out.ok and out.attempts == 3
    assert out.data["status"] == "shipped"
    retries = tracer.find("tool.retry")
    assert len(retries) == 2 and all(r["delay_s"] >= 0 for r in retries)


def test_retry_exhausted_marks_error_non_retryable_for_model():
    f = FaultInjector()
    f.plan("lookup_order", "timeout")          # 单一模式：一直超时
    ex, _ = _executor(f)
    out = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    assert out.ok is False and out.attempts == 3
    assert out.error["kind"] == "timeout"
    assert out.error["retryable"] is False     # 执行器已经重试过，模型不应再试
    assert "已自动重试 3 次" in out.error["message"]


def test_permanent_error_is_not_retried_and_carries_hint():
    f = FaultInjector()
    f.plan("lookup_order", "not_found")
    ex, _ = _executor(f)
    out = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-999"}))
    assert out.ok is False and out.attempts == 1
    assert out.error["kind"] == "permanent" and out.error["retryable"] is False
    assert "核对订单号" in out.error["hint"]


def test_result_contract_violation_degrades_and_alerts():
    f = FaultInjector()
    f.plan("lookup_order", "garbage")
    ex, tracer = _executor(f)
    out = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    assert out.ok is False and out.error["kind"] == "invalid_result"
    assert tracer.find("alert", reason="result_contract_broken")


def test_v1_tool_passes_garbage_through_when_no_result_model():
    f = FaultInjector()
    f.plan("lookup_order", "garbage")
    ex, _ = _executor(f, version=1)
    out = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    assert out.ok is True and "orderId" in out.data     # 事故前：垃圾直接喂给模型


def test_fallback_default_value_marks_degraded():
    f = FaultInjector()
    f.plan("get_weather", "timeout")
    ex, _ = _executor(f)
    out = ex.execute(ToolCall("c1", "get_weather", {"city": "上海"}))
    assert out.ok and out.degraded
    assert out.data["condition"] == "unknown"
    env = out.envelope()
    assert env["degraded"] is True and "降级" in env["note"]


def test_garbage_result_with_fallback_degrades_instead_of_failing():
    f = FaultInjector()
    f.plan("get_weather", "garbage")
    ex, _ = _executor(f)
    out = ex.execute(ToolCall("c1", "get_weather", {"city": "上海"}))
    assert out.ok and out.degraded and out.error["kind"] == "invalid_result"


def test_non_idempotent_write_is_never_auto_retried_and_hands_off():
    f = FaultInjector()
    f.plan("refund_order", "transient")
    ex, _ = _executor(f)
    out = ex.execute(ToolCall("c1", "refund_order", {"order_id": "SO-123", "amount": 10, "idempotency_key": "k1"}))
    assert out.ok is False and out.attempts == 1 and out.handoff is True


def test_idempotency_key_makes_repeated_refund_safe():
    ex, _ = _executor()
    a = ex.execute(ToolCall("c1", "refund_order", {"order_id": "SO-123", "amount": 10, "idempotency_key": "k1"}))
    b = ex.execute(ToolCall("c2", "refund_order", {"order_id": "SO-123", "amount": 10, "idempotency_key": "k1"}))
    assert a.data["refund_id"] == b.data["refund_id"]
    assert len(ex.registry.refund_ledger) == 1


def test_circuit_breaker_opens_after_consecutive_failures():
    f = FaultInjector()
    f.plan("search_kb", "transient")
    ex, _ = _executor(f, breaker=lambda: CircuitBreaker(failure_threshold=2, recovery_timeout_s=60))
    ex.execute(ToolCall("c1", "search_kb", {"query": "a"}))
    ex.execute(ToolCall("c2", "search_kb", {"query": "b"}))
    out = ex.execute(ToolCall("c3", "search_kb", {"query": "c"}))
    assert out.degraded and out.error["kind"] == "circuit_open" and out.attempts == 0   # 没打下游


def test_idempotent_read_is_cached_within_ttl():
    f = FaultInjector()
    ex, _ = _executor(f)
    a = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    b = ex.execute(ToolCall("c2", "lookup_order", {"order_id": "SO-123"}))
    assert a.cached is False and b.cached is True
    assert f.calls["lookup_order"] == 1


def test_real_timeout_guard_returns_control_quickly():
    f = FaultInjector()
    f.plan("get_weather", "slow", "ok")          # 第一次真的睡 0.5s（超时 0.2s），第二次成功
    ex, _ = _executor(f)
    t0 = time.perf_counter()
    out = ex.execute(ToolCall("c1", "get_weather", {"city": "上海"}))
    assert out.ok and out.attempts == 2
    assert time.perf_counter() - t0 < 0.45      # 没有等慢调用跑完


def test_cache_never_applies_to_non_idempotent_tools_even_with_ttl():
    from agentkit.tools.builtin import RefundArgs
    from agentkit.tools.registry import ToolRegistry, ToolSpec

    calls = {"n": 0}

    def handler(args):
        calls["n"] += 1
        return {"refund_id": f"RF-{calls['n']}"}

    reg = ToolRegistry([ToolSpec("refund_order", "d", RefundArgs, handler, idempotent=False, cache_ttl_s=999)])
    ex = ToolExecutor(reg, Tracer(), sleep=lambda s: None)
    a = ex.execute(ToolCall("c1", "refund_order", {"order_id": "SO-1", "amount": 1, "idempotency_key": "k"}))
    b = ex.execute(ToolCall("c2", "refund_order", {"order_id": "SO-1", "amount": 1, "idempotency_key": "k"}))
    assert not a.cached and not b.cached and calls["n"] == 2


def test_envelope_hash_ignores_latency_and_notes():
    ex, _ = _executor()
    a = ex.execute(ToolCall("c1", "lookup_order", {"order_id": "SO-123"}))
    b = ex.execute(ToolCall("c2", "lookup_order", {"order_id": "SO-123"}))
    assert a.result_hash() == b.result_hash()
