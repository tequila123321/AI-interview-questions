"""Anthropic 适配器：把 agentkit 的消息模型翻译成 Messages API，把响应翻译回来。

只有这一个文件知道厂商 SDK 的形状。要点：
- 工具调用：assistant 消息里的 tool_calls → tool_use 内容块；tool 消息 → user 消息里的 tool_result 块，
  同一批并行调用的结果必须放在同一条 user 消息里。
- 提示词缓存：系统提示分块，只在第一块（稳定部分）打缓存断点，记忆块在第二块不打；最后一条消息再打一个
  "移动断点"，让同一轮里多次工具往返命中前缀缓存。缓存是前缀匹配：断点之前的任何字节变化都会让缓存失效。
- 错误分类：429 / 5xx / 连接与超时 → LLMTransientError（Agent 层指数退避）；4xx → LLMPermanentError。
- 拒答：stop_reason == "refusal" 原样映射，Agent 用兜底话术回复。
- 默认开启服务端 fallbacks（模型因安全策略拒答时自动换模型续答）；不需要可关。

用法：
    llm = AnthropicLLM()                      # 需要 ANTHROPIC_API_KEY 或 `ant auth login`
    agent = Agent(llm, executor, memory, ...)
"""

from __future__ import annotations

from typing import Any

from .base import LLMPermanentError, LLMResponse, LLMTransientError, Message, SystemPrompt, ToolCall, Usage, system_blocks

DEFAULT_MODEL = "claude-opus-5"


def to_anthropic_system(system: SystemPrompt) -> list[dict[str, Any]]:
    """第一块（稳定系统提示）打缓存断点；后面的块（记忆块等会变的内容）不打，变了也不影响前缀缓存。"""
    blocks = system_blocks(system)
    out: list[dict[str, Any]] = [{"type": "text", "text": b} for b in blocks]
    if out:
        out[0]["cache_control"] = {"type": "ephemeral"}
    return out


def to_anthropic_messages(messages: list[Message], *, moving_cache_breakpoint: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _append_user_block(block: dict[str, Any]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for m in messages:
        if m.role == "user":
            _append_user_block({"type": "text", "text": m.content})
        elif m.role == "tool":
            _append_user_block({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content, "is_error": m.is_error})
        elif m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
            if content:
                out.append({"role": "assistant", "content": content})

    if moving_cache_breakpoint and out:
        out[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return out


def from_anthropic_response(resp: Any) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
    u = resp.usage
    usage = Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
    stop = resp.stop_reason if resp.stop_reason in ("end_turn", "tool_use", "max_tokens", "refusal") else "other"
    return LLMResponse(text="".join(text_parts), tool_calls=tool_calls, usage=usage, stop_reason=stop, model=getattr(resp, "model", ""), raw=resp)


class AnthropicLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        use_fallbacks: bool = True,
        client: Any | None = None,
        timeout_s: float = 120.0,
    ):
        if client is None:
            import anthropic  # 延迟导入：没装 SDK 也能用 FakeLLM 跑全部测试

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)  # 重试交给 Agent 层统一做，便于 trace
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    def complete(self, system: SystemPrompt, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        import anthropic

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": to_anthropic_system(system),
            "messages": to_anthropic_messages(messages),
        }
        if tools:
            params["tools"] = tools
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        try:
            if self.use_fallbacks:
                resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
            else:
                resp = self.client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if getattr(exc, "response", None) is not None else None
            raise LLMTransientError(f"rate limited: {exc}", retry_after=float(retry_after) if retry_after else None) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMTransientError(f"server error {exc.status_code}: {exc}") from exc
            raise LLMPermanentError(f"api error {exc.status_code}: {exc}") from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise LLMTransientError(f"connection error: {exc}") from exc
        return from_anthropic_response(resp)
