from agentkit.guard.loop_detector import LoopDetector, LoopPolicy, call_key
from agentkit.llm.base import ToolCall


def _c(name="lookup_order", **args):
    return ToolCall("id", name, args)


def test_same_args_consecutive_warn_then_stop():
    d = LoopDetector(LoopPolicy(same_call_warn_at=2, same_call_stop_at=3))
    c = _c(order_id="SO-1")
    assert d.observe(c).action == "ok"
    d.record(c, "h1")
    v = d.observe(c)
    assert v.action == "warn" and "第 2 次" in v.reason and v.hint
    d.record(c, "h1")
    v = d.observe(c)
    assert v.action == "stop" and "第 3 次" in v.reason


def test_canonical_key_ignores_dict_order():
    assert call_key(_c(a=1, b=2)) == call_key(_c(b=2, a=1))
    assert call_key(_c(a=1)) != call_key(_c(a=2))


def test_alternating_calls_are_caught_by_total_count():
    d = LoopDetector(LoopPolicy(same_call_stop_at=99, same_call_total_stop_at=3, no_progress_stop_at=99))
    a, b = _c(order_id="A"), _c(order_id="B")
    seq = [a, b, a, b, a]
    verdicts = []
    for c in seq:
        v = d.observe(c)
        verdicts.append(v.action)
        d.record(c, call_key(c))
    assert verdicts == ["ok", "ok", "ok", "ok", "stop"]      # 第 3 次 A


def test_no_progress_when_results_identical_across_different_calls():
    d = LoopDetector(LoopPolicy(same_call_stop_at=99, same_call_total_stop_at=99, no_progress_stop_at=3))
    for i in range(3):
        c = _c("search_kb", query=f"q{i}")
        assert d.observe(c).action == "ok"
        d.record(c, "same-hash")
    assert d.observe(_c("search_kb", query="q9")).action == "stop"


def test_max_rounds():
    d = LoopDetector(LoopPolicy(max_rounds=2))
    assert d.on_round().action == "ok"
    assert d.on_round().action == "ok"
    assert d.on_round().action == "stop"
