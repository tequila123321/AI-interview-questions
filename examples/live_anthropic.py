"""用真实模型跑同一套 Agent（需要 ANTHROPIC_API_KEY 或 `ant auth login`）。

运行：python examples/live_anthropic.py
会真实计费；默认模型 claude-opus-5，两轮对话。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.harness import build_agent
from agentkit.llm.anthropic_client import AnthropicLLM
from agentkit.observability.trace import Tracer

try:
    llm = AnthropicLLM(effort="medium")
except Exception as exc:  # noqa: BLE001
    print("无法初始化 Anthropic 客户端:", exc)
    sys.exit(1)

tracer = Tracer(path=pathlib.Path(__file__).resolve().parent / "out" / "live.jsonl")
agent = build_agent(profile="guarded", llm=llm, tracer=tracer)
for text in ["我的订单 SO-123 到哪了？", "我对花生过敏，另外这个订单帮我退款"]:
    r = agent.run_turn(text)
    print(f"\n用户: {text}\n助手: {r.text}\n  rounds={r.rounds} tools={[c.name for c in r.tool_calls]} tokens={r.usage.total} "
          f"cache_read={r.usage.cache_read_tokens} guard={r.guard_events}")
tracer.close()
print("\ntrace 已写到 examples/out/live.jsonl")
