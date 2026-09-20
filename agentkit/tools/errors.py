"""工具层错误分类。

面试里"try-except 捕获一下"之所以不及格，是因为它没有回答三个问题：
1. 这个错误重试有没有意义？（retryable）
2. 重试没意义时该怎么办？（降级：默认值 / 人工兜底 / 把结构化错误还给模型让它换路）
3. 模型看到错误之后会怎么做？（错误文案会直接影响模型下一步——"please retry" 会诱导死循环）

所以每个错误都带 kind / retryable / hint，最终以结构化信封喂给模型（见 executor.ToolOutcome）。
"""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    kind = "tool_error"
    retryable = False

    def __init__(self, message: str, *, hint: str = "", retry_after: float | None = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.retry_after = retry_after
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.hint:
            d["hint"] = self.hint
        if self.details is not None:
            d["details"] = self.details
        return d


class ToolArgsInvalid(ToolError):
    """模型给的参数不符合 schema。不由执行器重试——把字段级错误还给模型，让它自己修正。"""

    kind = "invalid_args"
    retryable = False


class ToolTimeout(ToolError):
    """下游超时。幂等工具可重试（指数退避）。"""

    kind = "timeout"
    retryable = True


class ToolTransientError(ToolError):
    """5xx / 连接重置 / 限流：短暂故障，可重试。"""

    kind = "transient"
    retryable = True


class ToolPermanentError(ToolError):
    """4xx / 业务规则拒绝 / 资源不存在：重试无意义。"""

    kind = "permanent"
    retryable = False


class ToolResultInvalid(ToolError):
    """下游返回了不符合约定的结构（契约变更、上游 bug）。不重试，走降级，并且必须告警。"""

    kind = "invalid_result"
    retryable = False


class CircuitOpen(ToolError):
    """熔断器打开：最近连续失败太多，直接降级，不再打下游。"""

    kind = "circuit_open"
    retryable = False


class UnknownTool(ToolError):
    kind = "unknown_tool"
    retryable = False
