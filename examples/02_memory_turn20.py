"""第 1 轮说的订单号，第 20 轮还记得吗？两种配置对比 + 自动诊断。

运行：python examples/02_memory_turn20.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.harness import build_agent
from agentkit.memory.diagnose import diagnose_missing_fact

FILLER = ["今天上海天气怎么样", "退款政策是什么", "谢谢", "北京天气呢", "多久能到", "好的"]


def run(profile: str):
    a = build_agent(profile=profile)
    a.run_turn("先记一下：我的订单号是 SO-000123，另外我对花生过敏，不要香菜")
    for i in range(18):
        a.run_turn(FILLER[i % len(FILLER)])
    r = a.run_turn("我的订单号是多少？")
    return a, r


for profile, label in (("naive", "没有长期记忆层，压缩不保关键信息"), ("guarded", "三层记忆 + pinned 优先召回")):
    a, r = run(profile)
    ctx = a.tracer.find("context.built", turn=20)[0]
    comps = a.tracer.find("memory.compress")
    print(f"\n==== {profile}：{label} ====")
    print("第 20 轮回答:", r.text)
    print(f"第 20 轮上下文: 总 {ctx['total_tokens']} tokens | 记忆块(关键信息+摘要) {ctx['tokens_memory_block']} | 对话窗口 {ctx['tokens_messages']} "
          f"| 窗口内轮次 {ctx['included_turns']} | pinned={ctx['pinned_keys']} recalled={ctx['recalled_keys']}")
    print(f"压缩发生 {len(comps)} 次；第 1 轮在第 {next((c['turn'] for c in comps if 1 in c['dropped_turns']), '?')} 轮被压缩进摘要")
    print("关键信息块:\n  " + (ctx["memory_block"].split("## 早先对话摘要")[0].strip().replace("\n", "\n  ") or "（空）"))
    print("\n诊断「SO-000123 为什么第 20 轮不知道 / 知道」:")
    print("  " + diagnose_missing_fact(a.tracer.events, "SO-123", 20, aliases=["SO-000123"]).render().replace("\n", "\n  "))
    if profile == "guarded":
        print("\n诊断「不要香菜」（未 pinned 的软偏好）:")
        print("  " + diagnose_missing_fact(a.tracer.events, "香菜", 20).render().replace("\n", "\n  "))
