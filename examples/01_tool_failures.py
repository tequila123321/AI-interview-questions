"""工具层的每一条失败路径，以及模型最终看到的信封。

运行：python examples/01_tool_failures.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.llm.base import ToolCall
from agentkit.observability.trace import Tracer
from agentkit.tools.builtin import FaultInjector, make_tools
from agentkit.tools.executor import ToolExecutor
from agentkit.tools.retry import CircuitBreaker


def show(title: str, outcome, tracer: Tracer | None = None) -> None:
    print(f"\n=== {title} ===")
    print(f"ok={outcome.ok} degraded={outcome.degraded} handoff={outcome.handoff} cached={outcome.cached} attempts={outcome.attempts}")
    print("模型看到的信封:", json.dumps(outcome.envelope(), ensure_ascii=False))
    if tracer:
        for ev in tracer.find("tool.retry"):
            print(f"  重试 #{ev['attempt']} 等待 {ev['delay_s']}s  原因: {ev['error']}")
        for ev in tracer.find("alert"):
            print(f"  ALERT[{ev['severity']}] {ev['reason']}")


def fresh(faults: FaultInjector | None = None, **kw):
    tracer = Tracer()
    return ToolExecutor(make_tools(faults), tracer, sleep=lambda s: None, **kw), tracer


# 1. 模型给错参数
ex, tr = fresh()
show("1. 参数不符合 schema（模型的错，字段级错误还给它改）", ex.execute(ToolCall("c1", "lookup_order", {"order": "SO-123"})))

# 2. 超时两次后成功
f = FaultInjector(); f.plan("lookup_order", "timeout", "timeout", "ok")
ex, tr = fresh(f)
show("2. 下游超时两次，第三次成功（指数退避 + 抖动）", ex.execute(ToolCall("c2", "lookup_order", {"order_id": "SO-123"})), tr)

# 3. 持续超时
f = FaultInjector(); f.plan("lookup_order", "timeout")
ex, tr = fresh(f)
show("3. 持续超时：重试耗尽后信封标记 retryable=false，模型不应再试", ex.execute(ToolCall("c3", "lookup_order", {"order_id": "SO-123"})), tr)

# 4. 永久错误
f = FaultInjector(); f.plan("lookup_order", "not_found")
ex, tr = fresh(f)
show("4. 订单不存在：永久错误，不重试，hint 告诉模型下一步", ex.execute(ToolCall("c4", "lookup_order", {"order_id": "SO-999"})), tr)

# 5. 契约破坏
f = FaultInjector(); f.plan("lookup_order", "garbage")
ex, tr = fresh(f)
show("5. 上游改了字段名：结果 schema 校验失败 → 不喂垃圾给模型 + 告警", ex.execute(ToolCall("c5", "lookup_order", {"order_id": "SO-123"})), tr)

# 6. 降级默认值
f = FaultInjector(); f.plan("get_weather", "timeout")
ex, tr = fresh(f)
show("6. 非关键工具持续超时：降级为默认值，信封带 degraded 与说明", ex.execute(ToolCall("c6", "get_weather", {"city": "上海"})), tr)

# 7. 非幂等写操作
f = FaultInjector(); f.plan("refund_order", "transient")
ex, tr = fresh(f)
show("7. 退款遇到 503：非幂等，绝不自动重试 → 转人工", ex.execute(ToolCall("c7", "refund_order", {"order_id": "SO-123", "amount": 10, "idempotency_key": "k1"})), tr)

# 8. 熔断
f = FaultInjector(); f.plan("search_kb", "transient")
ex, tr = fresh(f, breaker_factory=lambda: CircuitBreaker(failure_threshold=2, recovery_timeout_s=60))
ex.execute(ToolCall("c8a", "search_kb", {"query": "a"}))
ex.execute(ToolCall("c8b", "search_kb", {"query": "b"}))
show("8. 连续失败两次后熔断：第三次不打下游，直接降级（attempts=0）", ex.execute(ToolCall("c8c", "search_kb", {"query": "c"})), tr)

# 9. 缓存
f = FaultInjector()
ex, tr = fresh(f)
ex.execute(ToolCall("c9a", "lookup_order", {"order_id": "SO-123"}))
show("9. 幂等读缓存：同参数第二次不打下游", ex.execute(ToolCall("c9b", "lookup_order", {"order_id": "SO-123"})))
print("   下游实际被调用次数:", f.calls["lookup_order"])
