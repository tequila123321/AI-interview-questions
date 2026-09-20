import pytest

from agentkit.guard.assertions import Assertion, AssertionFailed, AssertionRunner
from agentkit.guard.budget import Budget, BudgetPolicy
from agentkit.guard.content_filter import ContentFilter
from agentkit.llm.base import Usage


def test_budget_cost_and_thresholds():
    b = Budget(BudgetPolicy(max_tokens_per_turn=1000, max_cost_usd_per_session=1.0, wrap_up_ratio=0.8))
    assert b.cost_of(Usage(1_000_000, 0)) == 5.0
    assert b.cost_of(Usage(0, 1_000_000)) == 25.0
    assert b.cost_of(Usage(0, 0, 1_000_000)) == 0.5
    b.start_turn()
    assert b.charge(Usage(500, 100)).action == "ok"
    assert b.charge(Usage(200, 50)).action == "wrap_up"      # 850 ≥ 800
    assert b.charge(Usage(200, 0)).action == "stop"           # 1050 ≥ 1000


def test_content_filter_actions():
    cf = ContentFilter(blocked_terms=("内部密钥",), judge=lambda t: (not t.startswith("辱骂"), "toxic"))
    assert cf.check_output("正常回复").action == "pass"
    r = cf.check_output("联系 13812345678 或 test@example.com")
    assert r.action == "redact" and "138****5678" in r.text and "test@example.com" not in r.text
    assert cf.check_output("内部密钥 xxx").action == "block"
    assert cf.check_output("辱骂用户").action == "block"


def test_tool_args_pii_only_for_pii_free_tools():
    cf = ContentFilter(pii_free_tools=("search_kb",))
    args, r = cf.check_tool_args("search_kb", {"query": "13812345678 退款"})
    assert r.action == "redact" and "13812345678" not in args["query"]
    args, r = cf.check_tool_args("lookup_order", {"order_id": "13812345678"})
    assert r.action == "pass" and args["order_id"] == "13812345678"


def test_assertion_runner_hard_raises_soft_records():
    runner = AssertionRunner([
        Assertion("soft_one", "before_final", lambda ctx: "软失败", "soft"),
        Assertion("hard_one", "before_final", lambda ctx: ctx.get("ok", False), "hard"),
        Assertion("other_stage", "before_tool", lambda ctx: False, "hard"),
    ])
    results = runner.run("before_final", {"ok": True})
    assert [(r.name, r.passed) for r in results] == [("soft_one", False), ("hard_one", True)]
    with pytest.raises(AssertionFailed) as ei:
        runner.run("before_final", {"ok": False})
    assert ei.value.name == "hard_one"
