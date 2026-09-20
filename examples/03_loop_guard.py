"""死循环：只有最大步数 vs 同工具同参数检测。

运行：python examples/03_loop_guard.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.guard.budget import Budget
from agentkit.harness import build_agent
from agentkit.tools.builtin import FaultInjector

QUESTION = "我的订单 SO-000123 到哪了？"


def run(profile: str):
    f = FaultInjector()
    f.plan("lookup_order", "retry_later")            # 上游 200 + "please retry later"
    a = build_agent(profile=profile, policy="stubborn", faults=f)
    r = a.run_turn(QUESTION)
    return a, r


rows = []
for profile in ("naive", "guarded"):
    a, r = run(profile)
    per_round = [e["usage"]["input_tokens"] for e in a.tracer.find("llm.response")]
    rows.append((profile, r.rounds, len(r.tool_calls), r.usage.total, Budget().cost_of(r.usage), per_round))
    print(f"\n==== {profile} ====")
    print("停止原因:", r.stop_reason, "| 守护事件:", r.guard_events or "无")
    print("最终回复:", r.text)
    print("每一步模型输入 tokens:", per_round[:6], "..." if len(per_round) > 6 else "", per_round[-1])

print("\n对比（同一个'固执'模型、同一个上游故障）:")
print(f"{'配置':10} {'模型调用':>8} {'工具调用':>8} {'总 tokens':>10} {'估算成本(opus-5)':>18}")
for name, rounds, calls, tokens, cost, _ in rows:
    print(f"{name:10} {rounds:>8} {calls:>8} {tokens:>10} {cost:>18.4f}")
print("\n注意 naive 每一步的输入 tokens 都比上一步大（工具结果不断堆进上下文），总消耗随步数二次方增长；"
      "\n同参数检测在第 3 次相同调用就切断，比'烧到步数上限'早得多。")
