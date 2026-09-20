"""工具执行器：一次工具调用从模型发起到结果回到模型的完整链路。

    模型给的调用
      → 1 工具存在吗                     UnknownTool → 结构化错误（附可用工具列表）
      → 2 参数 schema 校验               ToolArgsInvalid → 字段级错误还给模型，让它改
      → 3 会话级结果缓存（幂等读）        命中直接返回，不打下游
      → 4 熔断器                          open → 直接降级
      → 5 调用（超时保护）+ 指数退避重试   只重试 retryable 且幂等
      → 6 结果 schema 校验                ToolResultInvalid → 降级 + 告警
      → 7 降级：fallback 默认值 / 转人工 / 结构化错误
      → 8 输出信封 {"ok":..., "data"|"error":{kind, message, retryable, hint}}

关键点：模型看到的永远是结构化信封，其中 retryable 明确告诉模型"再试一次有没有意义"，
hint 告诉它"没意义时该做什么"。线上死循环的一大来源就是错误文案里带 "please retry"。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..llm.base import ToolCall
from ..observability.trace import Tracer, stable_hash
from .errors import (
    CircuitOpen,
    ToolArgsInvalid,
    ToolError,
    ToolPermanentError,
    ToolResultInvalid,
    ToolTimeout,
    UnknownTool,
)
from .registry import ToolRegistry, ToolSpec
from .retry import NO_RETRY, CircuitBreaker, run_with_retry


@dataclass
class ToolOutcome:
    call: ToolCall
    ok: bool
    data: Any = None
    error: dict[str, Any] | None = None
    degraded: bool = False          # 用了 fallback 默认值
    handoff: bool = False           # 需要转人工
    cached: bool = False
    attempts: int = 0
    latency_ms: float = 0.0
    note: str = ""                  # 给模型的补充说明（例如"这是降级默认值"）

    def envelope(self) -> dict[str, Any]:
        if self.ok:
            env: dict[str, Any] = {"ok": True, "data": self.data}
            if self.degraded:
                env["degraded"] = True
            if self.cached:
                env["cached"] = True
        else:
            env = {"ok": False, "error": self.error or {"kind": "unknown", "message": "", "retryable": False}}
            if self.handoff:
                env["handoff"] = True
        if self.note:
            env["note"] = self.note
        return env

    def to_model_payload(self, guard_note: str | None = None) -> str:
        env = self.envelope()
        if guard_note:
            env["guard_note"] = guard_note
        return json.dumps(env, ensure_ascii=False, sort_keys=True, default=str)

    def result_hash(self) -> str:
        """用于"结果有没有变化"的判断：只看 ok/data/error，不看耗时和守护提示。"""
        env = self.envelope()
        env.pop("note", None)
        env.pop("cached", None)
        return stable_hash(env)


def _simplify_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"field": ".".join(str(p) for p in e.get("loc", ())), "problem": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]


def run_with_timeout(fn: Callable[[], Any], timeout_s: float | None) -> Any:
    """最后一道超时防线。

    Python 无法杀死线程：超时后工作线程仍会跑完，这里只是不再等它。用 daemon 线程是为了
    让泄漏的线程不会拖住进程退出（ThreadPoolExecutor 的工作线程不是 daemon，解释器退出时会
    join 它们，一个卡 6 秒的下游就让进程多活 6 秒）。真正的超时必须配置在 I/O 客户端上
    （httpx timeout、DB statement_timeout），或者用 asyncio.wait_for；这个包装保证的只是
    "调用方一定能在 timeout_s 内拿回控制权"。每次调用起一个线程，生产上应改用异步 I/O。
    """
    if timeout_s is None:
        return fn()
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - 原样带回调用方线程重新抛出
            error.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ToolTimeout(f"tool exceeded {timeout_s}s", hint="下游超时；若多次超时请告知用户稍后再试或转人工")
    if error:
        raise error[0]
    return result[0]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        tracer: Tracer | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        breaker_factory: Callable[[], CircuitBreaker] | None = None,
    ):
        self.registry = registry
        self.tracer = tracer or Tracer()
        self.sleep = sleep
        self.clock = clock
        self._breaker_factory = breaker_factory or (lambda: CircuitBreaker(clock=clock))
        self.breakers: dict[str, CircuitBreaker] = {}
        self._cache: dict[str, tuple[float, Any]] = {}

    # ---- 主流程 ----
    def execute(self, call: ToolCall) -> ToolOutcome:
        t0 = self.clock()
        self.tracer.event("tool.call", tool=call.name, call_id=call.id, args=call.args)
        outcome = self._execute(call)
        outcome.latency_ms = round((self.clock() - t0) * 1000, 2)
        self.tracer.event(
            "tool.result",
            tool=call.name,
            call_id=call.id,
            ok=outcome.ok,
            degraded=outcome.degraded,
            handoff=outcome.handoff,
            cached=outcome.cached,
            attempts=outcome.attempts,
            latency_ms=outcome.latency_ms,
            error_kind=(outcome.error or {}).get("kind"),
            result_hash=outcome.result_hash(),
            payload=outcome.envelope(),
        )
        return outcome

    def _execute(self, call: ToolCall) -> ToolOutcome:
        spec = self.registry.get(call.name)
        if spec is None:
            err = UnknownTool(f"未知工具: {call.name}", hint=f"可用工具: {', '.join(self.registry.names())}")
            return ToolOutcome(call, ok=False, error=err.to_dict())

        # 1. 参数校验：模型的错，让模型改
        try:
            args = spec.args_model.model_validate(call.args)
        except ValidationError as exc:
            err = ToolArgsInvalid(
                "参数不符合 schema",
                hint="请按字段错误修正参数后重新调用；不要凭空猜测缺失的值，缺信息就先问用户",
                details=_simplify_validation_error(exc),
            )
            return ToolOutcome(call, ok=False, error=err.to_dict())

        # 2. 缓存：只对幂等读操作生效（写操作配了 TTL 也不缓存）
        cache_key = f"{spec.name}:{stable_hash(args.model_dump())}"
        cacheable = spec.idempotent and spec.cache_ttl_s > 0
        if cacheable:
            hit = self._cache.get(cache_key)
            if hit and self.clock() - hit[0] <= spec.cache_ttl_s:
                return ToolOutcome(call, ok=True, data=hit[1], cached=True, attempts=0)

        # 3. 熔断
        breaker = self.breakers.setdefault(spec.name, self._breaker_factory())
        if not breaker.allow():
            err = CircuitOpen(f"{spec.name} 最近连续失败，已熔断", hint="该服务暂不可用，请告知用户稍后再试或转人工")
            return self._degrade(call, spec, args, err, attempts=0)

        # 4. 调用 + 超时 + 重试
        attempts = 0

        def attempt() -> Any:
            nonlocal attempts
            attempts += 1
            with self.tracer.span("tool.attempt", tool=spec.name, call_id=call.id, attempt=attempts) as extra:
                try:
                    result = run_with_timeout(lambda: spec.handler(args), spec.timeout_s)
                    extra["ok"] = True
                    return result
                except Exception as exc:
                    extra["ok"] = False
                    extra["error"] = f"{type(exc).__name__}: {exc}"
                    raise

        policy = spec.retry if spec.idempotent else NO_RETRY   # 非幂等：绝不自动重试
        try:
            raw = run_with_retry(
                attempt,
                policy,
                sleep=self.sleep,
                on_retry=lambda exc, n, delay: self.tracer.event(
                    "tool.retry", tool=spec.name, call_id=call.id, attempt=n, delay_s=round(delay, 3), error=str(exc)
                ),
            )
        except ToolError as exc:
            breaker.record_failure()
            return self._degrade(call, spec, args, exc, attempts)
        except Exception as exc:  # noqa: BLE001 - 未分类异常一律视为永久错误，不重试
            breaker.record_failure()
            err = ToolPermanentError(f"{type(exc).__name__}: {exc}", hint="工具内部错误，重试无意义")
            return self._degrade(call, spec, args, err, attempts)

        # 5. 结果校验：下游的错，不能让模型背
        if spec.result_model is not None:
            try:
                payload = raw.model_dump() if isinstance(raw, BaseModel) else raw
                data = spec.result_model.model_validate(payload).model_dump()
            except ValidationError as exc:
                breaker.record_failure()
                err = ToolResultInvalid(
                    f"{spec.name} 返回结构不符合契约",
                    hint="下游返回异常数据，已按降级处理；请勿据此编造信息",
                    details=_simplify_validation_error(exc),
                )
                self.tracer.event("alert", severity="high", tool=spec.name, reason="result_contract_broken")
                return self._degrade(call, spec, args, err, attempts)
        else:
            data = raw.model_dump() if isinstance(raw, BaseModel) else raw

        breaker.record_success()
        if cacheable:
            self._cache[cache_key] = (self.clock(), data)
        return ToolOutcome(call, ok=True, data=data, attempts=attempts)

    # ---- 降级 ----
    def _degrade(self, call: ToolCall, spec: ToolSpec, args: BaseModel, err: ToolError, attempts: int) -> ToolOutcome:
        error = err.to_dict()
        if err.retryable and attempts > 1:
            # 执行器已经按策略重试过了，不能再让模型觉得"再试一次也许就好"
            error["retryable"] = False
            error["message"] = f"{error['message']}（已自动重试 {attempts} 次）"
            error["hint"] = error.get("hint") or "下游持续不可用，请告知用户稍后再试或转人工，不要再重复调用"
        if spec.fallback is not None:
            data = spec.fallback(args, err)
            return ToolOutcome(
                call, ok=True, data=data, degraded=True, attempts=attempts,
                error=error, note=f"这是降级默认值（原因: {err.kind}），请如实告知用户信息可能不完整",
            )
        if spec.handoff_on_failure:
            return ToolOutcome(call, ok=False, error=error, handoff=True, attempts=attempts)
        return ToolOutcome(call, ok=False, error=error, attempts=attempts)
