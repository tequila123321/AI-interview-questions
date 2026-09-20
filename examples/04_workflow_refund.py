"""退款状态机：分支、节点重试、Saga 补偿、可选节点降级、断点恢复。

运行：python examples/04_workflow_refund.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agentkit.observability.trace import Tracer
from agentkit.tools.builtin import FaultInjector
from agentkit.workflow.example_flow import build_refund_workflow
from agentkit.workflow.state_machine import InMemoryCheckpointStore, WorkflowRunner


def run(title: str, state: dict, faults: FaultInjector | None = None, store=None, run_id="r"):
    wf, fx = build_refund_workflow(faults)
    runner = WorkflowRunner(wf, tracer=Tracer(), checkpoints=store, sleep=lambda s: None)
    print(f"\n=== {title} ===")
    try:
        r = runner.run(state, run_id=run_id)
    except SystemExit as exc:
        print(f"进程崩溃: {exc}  → checkpoint: next={store.load(run_id)['next']}")
        r = runner.resume(run_id)
        print("resume 后继续执行")
    print(f"status={r.status} path={' → '.join(r.path)}")
    if r.failed_node:
        print(f"failed_node={r.failed_node} error={r.error} compensated={r.compensated}")
    if r.degraded:
        print(f"degraded={r.degraded}")
    print(f"attempts={r.attempts}")
    print(f"账本={fx['ledger']} 通知={fx['notifications']} 工单={fx['tickets']}")
    return r


run("1. 小额自动退款（快乐路径）", {"order_id": "SO-123", "amount": 50})
run("2. 大额走人工审核（分支判定）", {"order_id": "SO-456", "amount": 500, "reason": "质量问题"})
run("3. 金额超过订单金额（分支：拒绝）", {"order_id": "SO-123", "amount": 999})

f = FaultInjector(); f.plan("validate_order", "transient", "transient", "ok")
run("4. 校验订单遇到两次 503（节点级重试）", {"order_id": "SO-123", "amount": 50}, f)

f = FaultInjector(); f.plan("update_inventory", "permanent")
run("5. 退款已记账、库存更新永久失败（Saga：逆序补偿，冲正而不是删记录）", {"order_id": "SO-123", "amount": 50}, f)

f = FaultInjector(); f.plan("notify_customer", "permanent")
run("6. 通知发送失败（可选节点：降级，不拖垮退款）", {"order_id": "SO-123", "amount": 50}, f)

store = InMemoryCheckpointStore()
f = FaultInjector(); f.plan("update_inventory", "crash", "ok")
run("7. 进程在库存更新时被杀（checkpoint 恢复，不重复退款）", {"order_id": "SO-123", "amount": 50}, f, store, "r7")
