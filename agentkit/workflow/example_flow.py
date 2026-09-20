"""示例：退款流程（Saga）。

    validate_order ──▶ check_eligibility ──┬─▶ auto_refund ──▶ update_inventory ──▶ notify_customer ──▶ END
                                            ├─▶ manual_review ──▶ END
                                            └─▶ reject ──▶ END

- validate_order：下游抖动可重试（节点级重试）
- 分支：金额 ≤ 100 且符合条件走自动退款；符合条件但金额大走人工审核；不符合直接拒绝
- auto_refund：有副作用，带补偿（冲正）
- update_inventory：必需节点；它失败会触发 auto_refund 的补偿——这就是"分支执行失败怎么回滚"
- notify_customer：可选节点；失败只标记 degraded，不能让整个退款失败——"防止单点故障拖垮整条链路"
"""

from __future__ import annotations

from typing import Any

from ..tools.builtin import ORDERS, FaultInjector
from ..tools.errors import ToolPermanentError, ToolTransientError
from ..tools.retry import RetryPolicy
from .state_machine import END, Node, Workflow


def build_refund_workflow(faults: FaultInjector | None = None) -> tuple[Workflow, dict[str, Any]]:
    faults = faults or FaultInjector()
    side_effects: dict[str, Any] = {"ledger": [], "inventory": [], "notifications": [], "tickets": []}

    def _mode(name: str) -> str:
        mode = faults.next_mode(name)
        if mode == "crash":
            raise SystemExit("simulated process crash")   # 模拟进程被杀：不是 Exception，不会被重试/补偿逻辑捕获
        if mode == "transient":
            raise ToolTransientError(f"{name}: 503")
        if mode == "permanent":
            raise ToolPermanentError(f"{name}: 业务拒绝")
        return mode

    def validate_order(state: dict[str, Any]) -> dict[str, Any]:
        _mode("validate_order")
        order = ORDERS.get(state["order_id"])
        if order is None:
            raise ToolPermanentError(f"订单 {state['order_id']} 不存在")
        return {"order": dict(order)}

    def check_eligibility(state: dict[str, Any]) -> dict[str, Any]:
        _mode("check_eligibility")
        order = state["order"]
        eligible = order["status"] in ("paid", "shipped", "delivered") and state["amount"] <= order["amount"]
        return {"eligible": eligible}

    def route_after_check(state: dict[str, Any]) -> str:
        if not state["eligible"]:
            return "reject"
        return "auto_refund" if state["amount"] <= 100 else "manual_review"

    def auto_refund(state: dict[str, Any]) -> dict[str, Any]:
        _mode("auto_refund")
        entry = {"order_id": state["order_id"], "amount": state["amount"], "type": "refund"}
        side_effects["ledger"].append(entry)
        return {"refund_entry": entry}

    def reverse_refund(state: dict[str, Any]) -> None:
        # 补偿：冲正，而不是删除记录（账务不可篡改）
        side_effects["ledger"].append({"order_id": state["order_id"], "amount": -state["amount"], "type": "reversal"})

    def update_inventory(state: dict[str, Any]) -> dict[str, Any]:
        _mode("update_inventory")
        side_effects["inventory"].append({"order_id": state["order_id"], "restock": state["order"]["items"]})
        return {"inventory_updated": True}

    def notify_customer(state: dict[str, Any]) -> dict[str, Any]:
        _mode("notify_customer")
        side_effects["notifications"].append({"order_id": state["order_id"], "msg": "退款已完成"})
        return {"notified": True}

    def manual_review(state: dict[str, Any]) -> dict[str, Any]:
        _mode("manual_review")
        ticket = {"order_id": state["order_id"], "amount": state["amount"], "reason": state.get("reason", "")}
        side_effects["tickets"].append(ticket)
        return {"ticket": ticket}

    def reject(state: dict[str, Any]) -> dict[str, Any]:
        return {"message": "该订单不符合退款条件"}

    fast_retry = RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05)
    wf = Workflow(name="refund", start="validate_order")
    wf.add(Node("validate_order", validate_order, retry=fast_retry, timeout_s=1.0), "check_eligibility")
    wf.add(Node("check_eligibility", check_eligibility), route_after_check)
    wf.add(Node("auto_refund", auto_refund, compensate=reverse_refund), "update_inventory")
    wf.add(Node("update_inventory", update_inventory, retry=fast_retry), "notify_customer")
    wf.add(Node("notify_customer", notify_customer, retry=fast_retry, optional=True), END)
    wf.add(Node("manual_review", manual_review), END)
    wf.add(Node("reject", reject), END)
    return wf, side_effects
