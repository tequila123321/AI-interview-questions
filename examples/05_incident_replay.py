"""INC-0421 事故全流程复现：发现 → 定位 → 修复 → 防复发（对应 docs/07_线上故障复盘.md）。

运行：python examples/05_incident_replay.py
产物：examples/out/inc0421_before.jsonl / inc0421_after.jsonl（可用任何工具翻）
"""

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.agent import Agent, AgentConfig
from agentkit.guard.budget import Budget
from agentkit.harness import GUARDED_LOOP, HarnessOptions, build_agent
from agentkit.llm.fake import FakeLLM
from agentkit.llm.policies import stubborn_policy
from agentkit.memory.manager import MemoryManager
from agentkit.observability.badcase import BadCase, run_badcase
from agentkit.observability.replay import RecordedToolExecutor
from agentkit.observability.trace import Tracer, load_trace
from agentkit.tools.builtin import FaultInjector, make_tools

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
QUESTION = "我的订单 SO-000123 到哪了？"


def run_profile(profile: str, path: pathlib.Path):
    if path.exists():
        path.unlink()
    tracer = Tracer(run_id=f"inc0421-{profile}", path=path)
    f = FaultInjector()
    f.plan("lookup_order", "retry_later")
    a = build_agent(profile=profile, policy="stubborn", faults=f, tracer=tracer)
    r = a.run_turn(QUESTION)
    tracer.close()
    return a, r


# ---------------------------------------------------------------- 1. 发现
print("=" * 70)
print("阶段 1｜发现：凌晨 02:13 成本告警 —— 单会话 token 超过 P99 的 10 倍")
print("=" * 70)
before_agent, before = run_profile("naive", OUT / "inc0421_before.jsonl")
cost = Budget().cost_of(before.usage)
print(f"会话 {before_agent.tracer.run_id}: 模型调用 {before.rounds} 次, 工具调用 {len(before.tool_calls)} 次, "
      f"tokens {before.usage.total}, 估算成本 ${cost:.4f}, 停止原因 {before.stop_reason}")
print("如果没有 per-session 指标：这个会话会安静地跑到 max_rounds=30，第二天看账单才知道。")

# ---------------------------------------------------------------- 2. 定位
print("\n" + "=" * 70)
print("阶段 2｜定位：拉 trace 看时间线")
print("=" * 70)
events = load_trace(OUT / "inc0421_before.jsonl")
calls = [e for e in events if e["kind"] == "tool.call"]
by_args = Counter((e["tool"], json.dumps(e["args"], ensure_ascii=False, sort_keys=True)) for e in calls)
print("工具调用按 (工具, 参数) 聚合:")
for (tool, args), n in by_args.most_common():
    print(f"  {n:>3} × {tool} {args}")
first_result = next(e for e in events if e["kind"] == "tool.result")
print("第一次工具返回（模型看到的）:", json.dumps(first_result["payload"], ensure_ascii=False))
texts = Counter(e["text"] for e in events if e["kind"] == "llm.response")
print("模型每一步说的话:", dict(texts))
print("\n结论：上游 HTTP 200 + body 里 'please retry later' 被 v1 工具原样透传，模型把它当成可重试信号；"
      "\n守护层只有 max_rounds=30，没有同参数检测 → 每步上下文变长 → 烧到上限才停。")

# ---------------------------------------------------------------- 3. 修复
print("\n" + "=" * 70)
print("阶段 3｜修复")
print("=" * 70)

# 3a. what-if 重放：工具结果用当时的记录，只加守护层，看会在第几步停
tracer = Tracer()
rec_exec = RecordedToolExecutor(events, make_tools(incident_version=1), tracer)
whatif = Agent(FakeLLM(policy=stubborn_policy), rec_exec, MemoryManager(tracer=tracer), AgentConfig(),
               loop_policy=GUARDED_LOOP, tracer=tracer, sleep=lambda s: None).run_turn(QUESTION)
print(f"3a 止血（只加同参数检测，工具不改，用当时记录的工具结果 what-if 重放）: {whatif.rounds} 次模型调用即停, tokens {whatif.usage.total}")

# 3b. 根因修复：工具 v2（错误分类 + hint + 结果校验 + 订单号归一化）+ 守护层全开
after_agent, after = run_profile("guarded", OUT / "inc0421_after.jsonl")
print(f"3b 根因修复（工具 v2 + 守护层）: {after.rounds} 次模型调用, tokens {after.usage.total}, 守护事件 {after.guard_events}")
first_after = next(e for e in load_trace(OUT / "inc0421_after.jsonl") if e["kind"] == "tool.result")
print("   修复后模型看到的第一次工具返回:", json.dumps(first_after["payload"], ensure_ascii=False))

print(f"\n{'':12}{'模型调用':>8}{'tokens':>10}{'成本$':>10}")
for name, r in (("事故前", before), ("只加守护", whatif), ("根因修复", after)):
    print(f"{name:12}{r.rounds:>8}{r.usage.total:>10}{Budget().cost_of(r.usage):>10.4f}")

# ---------------------------------------------------------------- 4. 防复发
print("\n" + "=" * 70)
print("阶段 4｜防复发：事故变成 Bad Case 回归")
print("=" * 70)
case = BadCase.load(pathlib.Path(__file__).resolve().parents[1] / "badcases" / "BC-001_same_args_loop.json")


def factory(c: BadCase):
    f = FaultInjector()
    for tool, modes in c.tool_faults.items():
        f.plan(tool, *modes)
    return build_agent(HarnessOptions(profile=c.profile, policy=c.policy, faults=f))


res = run_badcase(case, factory)
print(f"{case.id} {case.title}\n  passed={res.passed} metrics={res.metrics}")
case.profile = "naive"
res = run_badcase(case, factory)
print(f"同一用例换回事故前配置: passed={res.passed} failures={res.failures}")
