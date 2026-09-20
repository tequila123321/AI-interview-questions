"""Token / 成本预算。

三个层级的上限，各解决不同问题：
- 单次请求输入上限：由上下文构建器在 AgentConfig.context_budget_tokens 内裁剪（见 memory/context_builder.py）；
- 单轮对话累计上限（这里）：一轮里模型来回调工具的总消耗，配合死循环检测；
- 会话成本上限（这里）：无论如何一个会话不能烧超过 X 美元（硬止损，报警）。

到 wrap_up_ratio 时不是直接掐断，而是让 Agent 进入"收尾模式"：下一次调用不再给工具，
要求模型用现有信息作答——用户得到的是一个降级但完整的回复，而不是半截话。

价格默认按 claude-opus-5（$5 / $25 每百万 token，缓存读 $0.50）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import Usage
from ..observability.trace import Tracer


@dataclass
class BudgetPolicy:
    max_tokens_per_turn: int = 50_000
    max_cost_usd_per_session: float = 0.50
    wrap_up_ratio: float = 0.8
    price_input_per_mtok: float = 5.0
    price_output_per_mtok: float = 25.0
    price_cache_read_per_mtok: float = 0.50
    price_cache_write_per_mtok: float = 6.25


@dataclass
class BudgetStatus:
    action: str = "ok"            # ok | wrap_up | stop
    reason: str = ""
    turn_tokens: int = 0
    session_cost_usd: float = 0.0


@dataclass
class Budget:
    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    tracer: Tracer | None = None
    session_usage: Usage = field(default_factory=Usage)
    turn_usage: Usage = field(default_factory=Usage)
    session_cost_usd: float = 0.0

    def cost_of(self, u: Usage) -> float:
        p = self.policy
        return (
            u.input_tokens * p.price_input_per_mtok
            + u.output_tokens * p.price_output_per_mtok
            + u.cache_read_tokens * p.price_cache_read_per_mtok
            + u.cache_write_tokens * p.price_cache_write_per_mtok
        ) / 1_000_000

    def start_turn(self) -> None:
        self.turn_usage = Usage()

    def charge(self, u: Usage) -> BudgetStatus:
        self.turn_usage = self.turn_usage.add(u)
        self.session_usage = self.session_usage.add(u)
        self.session_cost_usd += self.cost_of(u)
        p = self.policy
        status = BudgetStatus(turn_tokens=self.turn_usage.total, session_cost_usd=round(self.session_cost_usd, 6))

        if self.session_cost_usd >= p.max_cost_usd_per_session:
            status.action, status.reason = "stop", f"会话成本 ${self.session_cost_usd:.4f} 已达上限 ${p.max_cost_usd_per_session}"
        elif self.turn_usage.total >= p.max_tokens_per_turn:
            status.action, status.reason = "stop", f"本轮 token {self.turn_usage.total} 已达上限 {p.max_tokens_per_turn}"
        elif (
            self.turn_usage.total >= p.max_tokens_per_turn * p.wrap_up_ratio
            or self.session_cost_usd >= p.max_cost_usd_per_session * p.wrap_up_ratio
        ):
            status.action, status.reason = "wrap_up", "接近预算上限，进入收尾模式"

        if self.tracer and status.action != "ok":
            self.tracer.event("guard.budget", action=status.action, reason=status.reason, turn_tokens=status.turn_tokens,
                              session_tokens=self.session_usage.total, session_cost_usd=status.session_cost_usd)
        return status
