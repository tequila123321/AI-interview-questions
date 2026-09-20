import pytest

from agentkit.observability.trace import Tracer
from agentkit.tools.builtin import FaultInjector
from agentkit.workflow.example_flow import build_refund_workflow
from agentkit.workflow.state_machine import END, InMemoryCheckpointStore, Node, Workflow, WorkflowRunner


def _runner(faults=None, store=None):
    wf, effects = build_refund_workflow(faults)
    return WorkflowRunner(wf, tracer=Tracer(), checkpoints=store, sleep=lambda s: None), effects


def test_auto_refund_happy_path():
    runner, fx = _runner()
    r = runner.run({"order_id": "SO-123", "amount": 50})
    assert r.status == "completed"
    assert r.path == ["validate_order", "check_eligibility", "auto_refund", "update_inventory", "notify_customer"]
    assert len(fx["ledger"]) == 1 and fx["notifications"]


def test_branch_to_manual_review_for_large_amount():
    runner, fx = _runner()
    r = runner.run({"order_id": "SO-456", "amount": 500, "reason": "质量问题"})
    assert r.status == "completed" and r.path[-1] == "manual_review"
    assert fx["tickets"] and not fx["ledger"]


def test_branch_reject_when_not_eligible():
    runner, _ = _runner()
    r = runner.run({"order_id": "SO-123", "amount": 999})
    assert r.status == "completed" and r.path[-1] == "reject" and "不符合" in r.state["message"]


def test_node_level_retry_on_transient():
    f = FaultInjector()
    f.plan("validate_order", "transient", "ok")
    runner, _ = _runner(f)
    r = runner.run({"order_id": "SO-123", "amount": 50})
    assert r.status == "completed" and r.attempts["validate_order"] == 2
    assert runner.tracer.count("workflow.retry", node="validate_order") == 1


def test_required_node_failure_compensates_completed_side_effects():
    f = FaultInjector()
    f.plan("update_inventory", "permanent")
    runner, fx = _runner(f)
    r = runner.run({"order_id": "SO-123", "amount": 50})
    assert r.status == "compensated" and r.failed_node == "update_inventory"
    assert r.compensated == ["auto_refund"]
    assert [e["type"] for e in fx["ledger"]] == ["refund", "reversal"]     # 冲正而不是删除
    assert not fx["notifications"]


def test_optional_node_failure_only_degrades():
    f = FaultInjector()
    f.plan("notify_customer", "permanent")
    runner, fx = _runner(f)
    r = runner.run({"order_id": "SO-123", "amount": 50})
    assert r.status == "completed" and r.degraded == ["notify_customer"]
    assert len(fx["ledger"]) == 1 and r.attempts["notify_customer"] == 1


def test_resume_from_checkpoint_after_crash():
    store = InMemoryCheckpointStore()
    f = FaultInjector()
    f.plan("update_inventory", "crash", "ok")
    runner, fx = _runner(f, store)
    with pytest.raises(SystemExit):
        runner.run({"order_id": "SO-123", "amount": 50}, run_id="r1")
    cp = store.load("r1")
    assert cp["next"] == "update_inventory" and cp["completed_with_effects"] == ["auto_refund"]
    r = runner.resume("r1")
    assert r.status == "completed" and r.path == ["validate_order", "check_eligibility", "auto_refund", "update_inventory", "notify_customer"]
    assert len(fx["ledger"]) == 1        # 没有重复退款


def test_compensation_failure_is_visible_on_result_and_alerts():
    def boom(state):
        raise RuntimeError("ledger service down")

    wf = Workflow("saga", start="a")
    wf.add(Node("a", lambda s: {"a": 1}, compensate=boom), "b")
    wf.add(Node("b", lambda s: {"b": 1}, compensate=lambda s: s.__setitem__("b_reversed", True)), "c")
    wf.add(Node("c", lambda s: (_ for _ in ()).throw(RuntimeError("fail"))), END)
    runner = WorkflowRunner(wf, tracer=Tracer())
    r = runner.run({})
    assert r.status == "compensation_failed"
    assert r.compensated == ["b"] and r.compensation_failed == ["a"]     # b 补偿成功，a 补偿失败，两者都看得见
    assert r.state.get("b_reversed") is True
    assert runner.tracer.find("alert", severity="critical")


def test_edge_loop_is_bounded():
    wf = Workflow("loop", start="a")
    wf.add(Node("a", lambda s: {"n": s.get("n", 0) + 1}), "b")
    wf.add(Node("b", lambda s: None), "a")
    r = WorkflowRunner(wf, max_transitions=6).run({})
    assert r.status == "failed" and "死循环" in r.error and r.state["n"] == 3
