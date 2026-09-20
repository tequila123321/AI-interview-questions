"""重试、退避、熔断。

三条生产经验：
1. 退避必须带抖动（full jitter），否则一批实例会在同一毫秒一起重试，把刚恢复的下游再打死。
2. 只重试"可重试"的错误，并且只重试幂等操作；非幂等写操作要靠幂等键，否则重试等于重复扣款。
3. 重试解决的是"抖动"，解决不了"下游挂了"。下游持续失败时靠熔断：快速失败 + 降级，
   给下游恢复时间，也让本服务不被拖垮（线程/连接池占满是常见的级联故障来源）。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3          # 总尝试次数（含第一次）
    base_delay_s: float = 0.2
    max_delay_s: float = 5.0       # 退避等待上限（不含 Retry-After）
    max_retry_after_s: float = 60.0  # 下游明确要求的等待上限：尊重它，但不能被一个荒谬的值挂死
    multiplier: float = 2.0
    jitter: str = "full"           # "full" | "none"

    def delay_for(self, attempt: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
        """attempt 从 1 开始，表示第 attempt 次失败后的等待时间。"""
        if retry_after is not None:                      # 下游明确告诉了 Retry-After：照做（单独封顶，不受退避上限影响）
            return min(retry_after, self.max_retry_after_s)
        cap = min(self.max_delay_s, self.base_delay_s * (self.multiplier ** (attempt - 1)))
        if self.jitter == "full":
            return (rng or random).uniform(0, cap)
        return cap


NO_RETRY = RetryPolicy(max_attempts=1)


def run_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Exception, int, float], None] | None = None,
    rng: random.Random | None = None,
) -> T:
    """执行 fn；异常带 retryable=True 且未达上限时按退避重试，否则原样抛出。"""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 由 retryable 属性决定去留
            retryable = bool(getattr(exc, "retryable", False))
            if not retryable or attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt, getattr(exc, "retry_after", None), rng)
            if on_retry:
                on_retry(exc, attempt, delay)
            sleep(delay)


@dataclass
class CircuitBreaker:
    """最小实现的三态熔断器：closed → open → half_open → closed/open。"""

    failure_threshold: int = 3
    recovery_timeout_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: str = "closed"
    consecutive_failures: int = 0
    opened_at: float = field(default=0.0)

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.clock() - self.opened_at >= self.recovery_timeout_s:
                self.state = "half_open"      # 放一个探测请求过去
                return True
            return False
        return True  # half_open：允许探测

    def record_success(self) -> None:
        self.state = "closed"
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state == "half_open" or self.consecutive_failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = self.clock()
