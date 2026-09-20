"""适配器的消息翻译（不联网）。"""

from types import SimpleNamespace

from agentkit.llm.anthropic_client import AnthropicLLM, from_anthropic_response, to_anthropic_messages
from agentkit.llm.base import Message, ToolCall


def test_tool_results_are_grouped_into_one_user_message_with_moving_cache_breakpoint():
    tc1, tc2 = ToolCall("t1", "a", {"x": 1}), ToolCall("t2", "b", {})
    msgs = [
        Message.user("hi"),
        Message.assistant("让我查一下", [tc1, tc2]),
        Message.tool(tc1, '{"ok": true}'),
        Message.tool(tc2, '{"ok": false}', is_error=True),
        Message.user("【系统】收尾"),
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert out[2]["content"][1]["is_error"] is True and out[2]["content"][1]["tool_use_id"] == "t2"
    assert out[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[0]["content"][0]


def test_response_translation():
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="好的"), SimpleNamespace(type="tool_use", id="t1", name="lookup_order", input={"order_id": "SO-1"})],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=7, cache_creation_input_tokens=0),
        stop_reason="tool_use", model="claude-opus-5",
    )
    r = from_anthropic_response(resp)
    assert r.text == "好的" and r.tool_calls[0].name == "lookup_order" and r.stop_reason == "tool_use"
    assert r.usage.cache_read_tokens == 7


def test_client_request_shape_with_stub():
    captured = {}

    class _Msgs:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn", model="stub")

    stub = SimpleNamespace(beta=SimpleNamespace(messages=_Msgs()), messages=_Msgs())
    llm = AnthropicLLM(client=stub, effort="high")
    r = llm.complete(["STABLE SYS", "## 已知关键信息\n- 订单号: SO-1"], [Message.user("hi")],
                     [{"name": "t", "description": "d", "input_schema": {"type": "object"}}])
    assert r.text == "ok"
    assert captured["model"] == "claude-opus-5" and captured["fallbacks"] == "default"
    # 稳定块打缓存断点，记忆块不打：记忆块变化不会让稳定前缀失效
    assert [b["text"] for b in captured["system"]] == ["STABLE SYS", "## 已知关键信息\n- 订单号: SO-1"]
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in captured["system"][1]
    assert captured["tools"][0]["name"] == "t" and captured["output_config"] == {"effort": "high"}
