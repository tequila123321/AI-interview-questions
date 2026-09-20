"""Agent 主循环 + 守护层的集成行为。"""

from agentkit.guard.budget import BudgetPolicy
from agentkit.harness import build_agent
from agentkit.llm.fake import FakeLLM, call, reply
from agentkit.llm.policies import customer_service_policy
from agentkit.tools.builtin import FaultInjector


def test_happy_path_uses_tool_then_answers():
    a = build_agent(profile="guarded")
    r = a.run_turn("帮我查一下订单 SO-123 到哪了")
    assert r.stop_reason == "end_turn" and r.rounds == 2
    assert "shipped" in r.text and not r.handoff


def test_stubborn_model_is_stopped_by_same_call_detector():
    f = FaultInjector()
    f.plan("lookup_order", "retry_later")
    a = build_agent(profile="guarded", policy="stubborn", faults=f)
    r = a.run_turn("我的订单 SO-000123 到哪了？")
    assert r.stop_reason == "loop_stop"
    assert r.rounds <= 4
    assert any("第 3 次" in e for e in r.guard_events)
    assert a.tracer.count("guard.loop", action="stop") == 1
    # 收尾模式：最后一次模型请求没有工具，且带了收尾指令
    last = a.llm.last_request()
    assert last.tools == [] and last.messages[-1].content.startswith("【系统】")


def test_naive_profile_only_has_max_rounds_and_burns_far_more_tokens():
    f1, f2 = FaultInjector(), FaultInjector()
    f1.plan("lookup_order", "retry_later")
    f2.plan("lookup_order", "retry_later")
    naive = build_agent(profile="naive", policy="stubborn", faults=f1).run_turn("我的订单 SO-000123 到哪了？")
    guarded = build_agent(profile="guarded", policy="stubborn", faults=f2).run_turn("我的订单 SO-000123 到哪了？")
    assert naive.rounds == 31 and guarded.rounds <= 4
    assert naive.usage.total > guarded.usage.total * 8


def test_sensible_model_retries_once_on_retryable_then_answers_by_hint():
    f = FaultInjector()
    f.plan("lookup_order", "not_found")
    a = build_agent(profile="guarded", faults=f)
    r = a.run_turn("查一下订单 SO-999 的状态")
    assert r.rounds == 2 and not r.handoff
    assert "核对订单号" in r.text


def test_budget_stop_with_pending_tool_calls_hands_off():
    f = FaultInjector()
    f.plan("lookup_order", "retry_later")
    a = build_agent(profile="guarded", policy="stubborn", faults=f, budget_policy=BudgetPolicy(max_tokens_per_turn=300))
    r = a.run_turn("我的订单 SO-000123 到哪了？")
    assert r.handoff and r.stop_reason == "budget_stop"
    assert a.tracer.count("guard.budget", action="stop") >= 1


def test_llm_transient_errors_are_retried_with_backoff():
    llm = FakeLLM(policy=customer_service_policy, fail_first_n=2)
    a = build_agent(profile="guarded", llm=llm)
    r = a.run_turn("查一下订单 SO-123 的状态")
    assert "shipped" in r.text
    assert a.tracer.count("llm.retry") == 2


def test_llm_persistent_failure_hands_off_instead_of_crashing():
    llm = FakeLLM(policy=customer_service_policy, fail_first_n=99)
    a = build_agent(profile="guarded", llm=llm)
    r = a.run_turn("查一下订单 SO-123 的状态")
    assert r.handoff and r.stop_reason == "llm_error"


def test_refund_requires_prior_lookup_assertion_blocks_fabricated_order():
    llm = FakeLLM(script=[call("refund_order", order_id="SO-456", amount=1.0, idempotency_key="k")])
    a = build_agent(profile="guarded", llm=llm)
    r = a.run_turn("给 SO-456 退款")
    assert r.handoff and "assertion:refund_target_must_come_from_lookup" in r.guard_events
    assert len(a.executor.registry.refund_ledger) == 0     # 写操作没有发生


def test_refund_after_lookup_passes_assertion_and_uses_idempotency_key():
    a = build_agent(profile="guarded")
    a.run_turn("查一下订单 SO-123 的状态")
    r = a.run_turn("这个订单帮我退款")
    assert not r.handoff and "退款已提交" in r.text
    assert len(a.executor.registry.refund_ledger) == 1


def test_refund_tool_failure_hands_off_without_retry():
    f = FaultInjector()
    f.plan("refund_order", "transient")
    a = build_agent(profile="guarded", faults=f)
    a.run_turn("查一下订单 SO-123 的状态")
    r = a.run_turn("这个订单帮我退款")
    assert r.handoff and r.stop_reason == "handoff"
    assert f.calls["refund_order"] == 1


def test_output_pii_is_redacted():
    llm = FakeLLM(script=[reply("已为您登记手机号 13812345678，稍后联系。")])
    a = build_agent(profile="guarded", llm=llm)
    r = a.run_turn("我的手机号是 13812345678")
    assert "13812345678" not in r.text and "138****5678" in r.text
    assert "output_redact" in r.guard_events


def test_blocked_term_replaces_output():
    llm = FakeLLM(script=[reply("内部密钥是 abc")])
    a = build_agent(profile="guarded", llm=llm)
    r = a.run_turn("你好")
    assert "内部密钥" not in r.text and "output_block" in r.guard_events


def test_tool_args_pii_is_redacted_before_leaving_the_system():
    llm = FakeLLM(script=[call("search_kb", query="手机 13812345678 的退款政策"), reply("好的")])
    a = build_agent(profile="guarded", llm=llm)
    a.run_turn("政策")
    ev = a.tracer.find("tool.call", tool="search_kb")[0]
    assert "13812345678" not in ev["args"]["query"]


def test_degraded_tool_result_is_surfaced_to_user():
    f = FaultInjector()
    f.plan("get_weather", "timeout")
    a = build_agent(profile="guarded", faults=f)
    r = a.run_turn("上海天气怎么样")
    assert r.degraded and "降级" in r.text and not r.handoff


def test_every_turn_is_fully_traced():
    a = build_agent(profile="guarded")
    a.run_turn("查一下订单 SO-123 的状态")
    kinds = {e["kind"] for e in a.tracer.events}
    assert {"turn.start", "context.built", "llm.request", "llm.response", "tool.call", "tool.result", "turn.end"} <= kinds
