"""组装器：按"配置档"构造一个完整的 Agent。测试、示例、Bad Case 回归都从这里拿 Agent，
保证大家跑的是同一套接线。

profile:
  naive    事故前的配置：只有最大步数（30）；工具 v1（透传上游文案、无结果校验）；没有长期记忆层；
           压缩时不保关键信息；无内容审核；预算宽松。
  guarded  事故后的配置：全部守护开启；工具 v2；规则抽取 + pinned；压缩保关键信息；断言；预算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agent import Agent, AgentConfig
from .guard.assertions import AssertionRunner, final_answer_not_empty, pinned_facts_survive_compression, refund_target_must_come_from_lookup
from .guard.budget import Budget, BudgetPolicy
from .guard.content_filter import ContentFilter
from .guard.loop_detector import LoopPolicy
from .llm.base import LLMClient
from .llm.fake import FakeLLM
from .llm.policies import POLICIES
from .memory.compressor import ExtractiveSummarizer
from .memory.extractor import NullExtractor, RuleFactExtractor
from .memory.manager import MemoryConfig, MemoryManager
from .observability.trace import Tracer
from .tools.builtin import FaultInjector, make_tools
from .tools.executor import ToolExecutor

NAIVE_LOOP = LoopPolicy(max_rounds=30, same_call_warn_at=10**6, same_call_stop_at=10**6, same_call_total_stop_at=10**6, no_progress_stop_at=10**6)
GUARDED_LOOP = LoopPolicy(max_rounds=10, same_call_warn_at=2, same_call_stop_at=3, same_call_total_stop_at=4, no_progress_stop_at=4)


@dataclass
class HarnessOptions:
    profile: str = "guarded"
    policy: str = "customer_service"
    faults: FaultInjector | None = None
    tracer: Tracer | None = None
    llm: LLMClient | None = None
    memory_config: MemoryConfig | None = None
    budget_policy: BudgetPolicy | None = None
    context_budget_tokens: int | None = None


def build_agent(opts: HarnessOptions | None = None, **kwargs) -> Agent:
    opts = opts or HarnessOptions(**kwargs)
    tracer = opts.tracer or Tracer()
    faults = opts.faults or FaultInjector()
    no_sleep: Callable[[float], None] = lambda _s: None    # 测试/演示不真的等退避时间

    if opts.profile == "naive":
        registry = make_tools(faults, incident_version=1)
        memory = MemoryManager(
            extractor=NullExtractor(),
            summarizer=ExtractiveSummarizer(keep_facts=False),
            config=opts.memory_config or MemoryConfig(working_high_water_tokens=600, keep_recent_turns=3, pin_facts_in_summary=False),
            tracer=tracer,
        )
        loop = NAIVE_LOOP
        content_filter = ContentFilter(pii_patterns={}, pii_free_tools=(), tracer=tracer)
        budget = Budget(opts.budget_policy or BudgetPolicy(max_tokens_per_turn=10**9, max_cost_usd_per_session=10**9), tracer=tracer)
        assertions = AssertionRunner([], tracer=tracer)
        context_budget = opts.context_budget_tokens or 100_000
    else:
        registry = make_tools(faults, incident_version=2)
        memory = MemoryManager(
            extractor=RuleFactExtractor(),
            summarizer=ExtractiveSummarizer(keep_facts=True),
            config=opts.memory_config or MemoryConfig(working_high_water_tokens=600, keep_recent_turns=3, pin_facts_in_summary=True),
            tracer=tracer,
        )
        loop = GUARDED_LOOP
        content_filter = ContentFilter(blocked_terms=("内部密钥",), tracer=tracer)
        budget = Budget(opts.budget_policy or BudgetPolicy(max_tokens_per_turn=40_000, max_cost_usd_per_session=0.50), tracer=tracer)
        assertions = AssertionRunner([refund_target_must_come_from_lookup(), final_answer_not_empty(), pinned_facts_survive_compression()], tracer=tracer)
        context_budget = opts.context_budget_tokens or 8000

    executor = ToolExecutor(registry, tracer, sleep=no_sleep)
    llm = opts.llm or FakeLLM(policy=POLICIES[opts.policy])
    return Agent(
        llm, executor, memory, AgentConfig(context_budget_tokens=context_budget),
        loop_policy=loop, content_filter=content_filter, budget=budget, assertions=assertions, tracer=tracer, sleep=no_sleep,
    )
