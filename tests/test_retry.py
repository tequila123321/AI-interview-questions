import random

import pytest

from agentkit.tools.errors import ToolPermanentError, ToolTransientError
from agentkit.tools.retry import CircuitBreaker, RetryPolicy, run_with_retry


def test_backoff_grows_exponentially_and_caps():
    p = RetryPolicy(max_attempts=6, base_delay_s=1, max_delay_s=5, jitter="none")
    assert [p.delay_for(i) for i in range(1, 6)] == [1, 2, 4, 5, 5]


def test_full_jitter_stays_within_cap():
    p = RetryPolicy(base_delay_s=1, max_delay_s=10, jitter="full")
    rng = random.Random(7)
    for i in range(1, 5):
        d = p.delay_for(i, rng=rng)
        assert 0 <= d <= min(10, 2 ** (i - 1))


def test_retry_after_from_downstream_is_respected_beyond_backoff_cap():
    p = RetryPolicy(base_delay_s=1, max_delay_s=5, max_retry_after_s=60)
    assert p.delay_for(1, retry_after=3.5) == 3.5
    assert p.delay_for(1, retry_after=30) == 30       # 下游要求等 30s，不能被 5s 的退避上限吃掉
    assert p.delay_for(1, retry_after=999) == 60      # 但有单独的封顶，防止被一个荒谬的值挂死


def test_only_retryable_errors_are_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ToolTransientError("503")
        return "ok"

    slept = []
    assert run_with_retry(flaky, RetryPolicy(max_attempts=3, base_delay_s=0.1, jitter="none"), sleep=slept.append) == "ok"
    assert slept == [0.1, 0.2]

    def permanent():
        raise ToolPermanentError("404")

    with pytest.raises(ToolPermanentError):
        run_with_retry(permanent, RetryPolicy(max_attempts=3), sleep=slept.append)
    assert len(slept) == 2   # 没有为永久错误睡过


def test_circuit_breaker_transitions():
    now = {"t": 0.0}
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=10, clock=lambda: now["t"])
    assert cb.allow()
    cb.record_failure()
    assert cb.state == "closed" and cb.allow()
    cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    now["t"] = 11
    assert cb.allow() and cb.state == "half_open"    # 放一个探测
    cb.record_failure()
    assert cb.state == "open"
    now["t"] = 30
    assert cb.allow()
    cb.record_success()
    assert cb.state == "closed" and cb.consecutive_failures == 0
