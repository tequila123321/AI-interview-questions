from pathlib import Path

import pytest

from agentkit.agent import Agent, AgentConfig
from agentkit.harness import GUARDED_LOOP, HarnessOptions, build_agent
from agentkit.memory.manager import MemoryConfig, MemoryManager
from agentkit.observability.badcase import BadCase, run_badcase
from agentkit.observability.replay import ReplayDivergence, replay
from agentkit.observability.trace import Tracer, load_trace
from agentkit.tools.builtin import FaultInjector, make_tools

ROOT = Path(__file__).resolve().parents[1]


def _record(tmp_path):
    tracer = Tracer(path=tmp_path / "trace.jsonl")
    f = FaultInjector()
    f.plan("lookup_order", "timeout", "ok")
    a = build_agent(profile="guarded", faults=f, tracer=tracer)
    a.run_turn("查一下订单 SO-123 的状态")
    a.run_turn("这个订单帮我退款")
    tracer.close()
    return load_trace(tmp_path / "trace.jsonl")


def _factory(system_prompt=None):
    def make(llm, executor):
        mem = MemoryManager(config=MemoryConfig(working_high_water_tokens=600, keep_recent_turns=3))
        cfg = AgentConfig(system_prompt=system_prompt) if system_prompt else AgentConfig()
        return Agent(llm, executor, mem, cfg, loop_policy=GUARDED_LOOP, tracer=Tracer(), sleep=lambda s: None)
    return make


def test_faithful_replay_reproduces_run_without_model_or_downstream(tmp_path):
    events = _record(tmp_path)
    report = replay(events, _factory(), make_tools(), strict=True)
    assert report.matched, report.diffs
    assert report.turns == 2 and report.llm_calls_replayed == report.llm_calls_recorded


def test_strict_replay_detects_context_logic_change(tmp_path):
    events = _record(tmp_path)
    with pytest.raises(ReplayDivergence):
        replay(events, _factory(system_prompt="换了系统提示词"), make_tools(), strict=True)


def _badcase_factory(case: BadCase):
    f = FaultInjector()
    for tool, modes in case.tool_faults.items():
        f.plan(tool, *modes)
    return build_agent(HarnessOptions(profile=case.profile, policy=case.policy, faults=f))


@pytest.mark.parametrize("case", BadCase.load_dir(ROOT / "badcases"), ids=lambda c: c.id)
def test_badcase_regression(case: BadCase):
    result = run_badcase(case, _badcase_factory)
    assert result.passed, f"{case.title}: {result.failures} metrics={result.metrics}"


def test_badcase_would_fail_on_naive_profile():
    """证明回归用例真的有区分度：同一个用例换成事故前配置必须不通过。"""
    case = BadCase.load(ROOT / "badcases" / "BC-001_same_args_loop.json")
    case.profile = "naive"
    result = run_badcase(case, _badcase_factory)
    assert not result.passed and result.metrics["rounds"] > 20
