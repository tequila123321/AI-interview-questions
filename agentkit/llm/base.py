"""与厂商无关的模型接口。

设计目的：
1. Agent 主循环、记忆、守护逻辑只依赖这一层，可以用 FakeLLM 做确定性测试与日志重放；
2. 真实厂商（Anthropic）只是一个适配器（见 anthropic_client.py），换模型不动业务代码。

消息模型刻意做得很薄：role 只有 user / assistant / tool 三种，system 单独传。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

Role = Literal["user", "assistant", "tool"]

# 系统提示可以是一段文本，也可以是若干块：第一块是稳定前缀（打缓存断点），后面是会变的内容（记忆块）
SystemPrompt = str | Sequence[str]


def system_text(system: SystemPrompt) -> str:
    return system if isinstance(system, str) else "\n\n".join(system)


def system_blocks(system: SystemPrompt) -> list[str]:
    return [system] if isinstance(system, str) else [b for b in system if b]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant 发起的调用
    tool_call_id: str | None = None                            # role == tool 时对应的调用 id
    tool_name: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)         # 例如 {"turn": 3}

    @staticmethod
    def user(text: str, **meta: Any) -> "Message":
        return Message(role="user", content=text, meta=meta)

    @staticmethod
    def assistant(text: str = "", tool_calls: list[ToolCall] | None = None, **meta: Any) -> "Message":
        return Message(role="assistant", content=text, tool_calls=list(tool_calls or []), meta=meta)

    @staticmethod
    def tool(call: ToolCall, content: str, is_error: bool = False, **meta: Any) -> "Message":
        return Message(
            role="tool",
            content=content,
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=is_error,
            meta=meta,
        )


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "end_turn"
    model: str = ""
    raw: Any = None


class LLMClient(Protocol):
    """一次对话补全。tools 为 JSON schema 列表（见 tools/registry.py 的 ToolSpec.schema()）。"""

    def complete(self, system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMError(Exception):
    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTransientError(LLMError):
    """429 / 5xx / 网络抖动 / 超时：可重试。"""

    retryable = True


class LLMPermanentError(LLMError):
    """400 / 401 / 404 / 请求体非法：重试无意义。"""

    retryable = False
