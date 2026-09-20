"""示例工具（电商客服域）+ 故障注入器。

FaultInjector 让测试和演示可以精确控制"第几次调用发生什么"：
    faults.plan("lookup_order", "timeout", "timeout", "ok")   # 前两次超时，第三次成功
可用模式：
    ok            正常
    timeout       抛 ToolTimeout（模拟 httpx 超时）
    slow          真的 sleep，用来测 run_with_timeout
    transient     抛 ToolTransientError（503）
    not_found     业务上找不到（永久错误）
    garbage       返回不符合契约的结构（模拟上游契约变更）
    retry_later   【事故复现】上游 HTTP 200 但 body 是 {"code":"NOT_FOUND","message":"... please retry later"}

lookup_order 有两个版本：
    lookup_order_v1  事故前：把上游 body 原样当作 data 返回（错误文案会诱导模型重试）
    lookup_order_v2  事故后：归一化订单号、把业务错误分类为永久错误并给出 hint
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel, Field

from .errors import ToolPermanentError, ToolTimeout, ToolTransientError
from .registry import ToolRegistry, ToolSpec
from .retry import RetryPolicy


class FaultInjector:
    def __init__(self) -> None:
        self.plans: dict[str, list[str]] = {}
        self.calls: dict[str, int] = {}

    def plan(self, tool: str, *modes: str) -> None:
        self.plans[tool] = list(modes)

    def next_mode(self, tool: str) -> str:
        self.calls[tool] = self.calls.get(tool, 0) + 1
        plan = self.plans.get(tool)
        if not plan:
            return "ok"
        if len(plan) == 1:
            return plan[0]           # 只剩一个模式时保持该模式（例如持续 retry_later）
        return plan.pop(0)


ORDERS: dict[str, dict[str, Any]] = {
    "SO-123": {"order_id": "SO-123", "status": "shipped", "amount": 89.0, "currency": "CNY", "items": ["保温杯"]},
    "SO-456": {"order_id": "SO-456", "status": "paid", "amount": 1299.0, "currency": "CNY", "items": ["耳机"]},
}


# ---------- schema ----------
class LookupOrderArgs(BaseModel):
    order_id: str = Field(description="订单号，形如 SO-123", min_length=3)


class OrderInfo(BaseModel):
    order_id: str
    status: str
    amount: float
    currency: str
    items: list[str]


class WeatherArgs(BaseModel):
    city: str = Field(description="城市名")


class WeatherInfo(BaseModel):
    city: str
    temp_c: float
    condition: str


class SearchKBArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class RefundArgs(BaseModel):
    order_id: str
    amount: float = Field(gt=0)
    idempotency_key: str = Field(description="调用方生成的幂等键，同键重复请求不会重复退款")


def normalize_order_id(raw: str) -> str:
    """SO-000123 / so123 / SO 123 → SO-123（事故修复项之一：兼容上游改格式前后的写法）。"""
    m = re.match(r"^\s*([A-Za-z]+)[\s-]*0*(\d+)\s*$", raw)
    if not m:
        return raw.strip().upper()
    return f"{m.group(1).upper()}-{m.group(2)}"


# ---------- 工具实现 ----------
def make_tools(faults: FaultInjector | None = None, *, incident_version: int = 2) -> ToolRegistry:
    faults = faults or FaultInjector()
    refund_ledger: dict[str, dict[str, Any]] = {}

    def _inject(tool: str) -> str:
        mode = faults.next_mode(tool)
        if mode == "crash":
            raise SystemExit("simulated process crash")   # 不是 Exception 子类：模拟进程被杀，用于测 checkpoint 恢复
        if mode == "timeout":
            raise ToolTimeout("upstream read timeout", hint="下游超时")
        if mode == "slow":
            time.sleep(0.5)
        if mode == "transient":
            raise ToolTransientError("503 Service Unavailable", retry_after=0.01)
        return mode

    def _upstream_lookup(order_id: str, mode: str) -> dict[str, Any]:
        """模拟上游订单服务。"""
        if mode == "garbage":
            return {"orderId": order_id, "state": "SHIPPED"}   # 字段名变了：契约被破坏
        if mode == "retry_later":
            return {"code": "NOT_FOUND", "message": f"order {order_id} not found, please retry later"}
        if mode == "not_found" or order_id not in ORDERS:
            return {"code": "NOT_FOUND", "message": f"order {order_id} not found"}
        return dict(ORDERS[order_id])

    def lookup_order_v1(args: LookupOrderArgs) -> Any:
        mode = _inject("lookup_order")
        return _upstream_lookup(args.order_id, mode)          # 原样透传：上游怎么说模型就看到什么

    def lookup_order_v2(args: LookupOrderArgs) -> Any:
        mode = _inject("lookup_order")
        body = _upstream_lookup(normalize_order_id(args.order_id), mode)
        if body.get("code") == "NOT_FOUND":
            raise ToolPermanentError(
                f"订单 {args.order_id} 不存在",
                hint="订单号可能有误，请向用户核对订单号（格式 SO-数字），不要重复用同一个订单号查询",
            )
        return body

    def get_weather(args: WeatherArgs) -> Any:
        mode = _inject("get_weather")
        if mode == "garbage":
            return {"city": args.city, "temperature": "26"}   # 字段名/类型都变了：契约被破坏
        return {"city": args.city, "temp_c": 26.0, "condition": "sunny"}

    def search_kb(args: SearchKBArgs) -> Any:
        _inject("search_kb")
        return {"hits": [{"title": "退款政策", "snippet": "签收后 7 天内可申请退款"}][: args.top_k]}

    def refund_order(args: RefundArgs) -> Any:
        _inject("refund_order")
        if args.idempotency_key in refund_ledger:
            return refund_ledger[args.idempotency_key]        # 幂等：同键直接返回上次结果
        oid = normalize_order_id(args.order_id)
        if oid not in ORDERS:
            raise ToolPermanentError(f"订单 {oid} 不存在，无法退款")
        if args.amount > ORDERS[oid]["amount"]:
            raise ToolPermanentError("退款金额超过订单金额", hint="退款金额不能超过订单实付金额")
        rec = {"refund_id": f"RF-{len(refund_ledger) + 1}", "order_id": oid, "amount": args.amount, "status": "done"}
        refund_ledger[args.idempotency_key] = rec
        return rec

    lookup_impl = lookup_order_v1 if incident_version == 1 else lookup_order_v2
    registry = ToolRegistry(
        [
            ToolSpec(
                name="lookup_order",
                description="按订单号查询订单状态、金额和商品",
                args_model=LookupOrderArgs,
                handler=lookup_impl,
                result_model=None if incident_version == 1 else OrderInfo,   # v1 没做结果校验（事故根因之一）
                timeout_s=0.2,
                retry=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05),
                idempotent=True,
                cache_ttl_s=0.0 if incident_version == 1 else 60.0,
            ),
            ToolSpec(
                name="get_weather",
                description="查询城市天气",
                args_model=WeatherArgs,
                handler=get_weather,
                result_model=WeatherInfo,
                timeout_s=0.2,
                retry=RetryPolicy(max_attempts=2, base_delay_s=0.01),
                fallback=lambda a, e: {"city": a.city, "temp_c": None, "condition": "unknown"},   # 非关键信息：默认值降级
                cache_ttl_s=300.0,
            ),
            ToolSpec(
                name="search_kb",
                description="在客服知识库中检索政策与常见问题",
                args_model=SearchKBArgs,
                handler=search_kb,
                timeout_s=0.2,
                retry=RetryPolicy(max_attempts=2, base_delay_s=0.01),
                fallback=lambda a, e: {"hits": []},
            ),
            ToolSpec(
                name="refund_order",
                description="对订单发起退款（需要幂等键）",
                args_model=RefundArgs,
                handler=refund_order,
                timeout_s=1.0,
                idempotent=False,               # 写操作：执行器不自动重试
                handoff_on_failure=True,        # 关键操作失败：转人工，不给默认值
            ),
        ]
    )
    registry.refund_ledger = refund_ledger  # type: ignore[attr-defined]  # 测试用
    return registry
