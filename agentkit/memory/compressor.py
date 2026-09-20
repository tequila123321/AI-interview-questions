"""上下文压缩（把旧的对话原文压成摘要）。

压缩一定有损，所以两条纪律：
1. 压缩输入里明确带上"必须保留"的事实（pinned），压缩输出后用代码校验它们还在，不在就补回去；
2. 压缩事件记 trace（哪些轮被压掉、摘要长度、补回了哪些事实），这样"信息是在压缩时丢的"能被证明而不是猜。

ExtractiveSummarizer 是确定性的教学实现（也是没有 LLM 时的兜底）；LLMSummarizer 用模型做摘要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from ..llm.base import LLMClient, Message
from ..tokens import estimate_tokens


class Summarizer(Protocol):
    def summarize(self, previous_summary: str, messages: list[Message], must_keep: list[str], max_tokens: int) -> str: ...


FACT_MARK = "【关键信息】"


def ensure_facts_present(summary: str, must_keep: list[str]) -> tuple[str, list[str]]:
    """把摘要里缺失的关键事实补回去。返回（新摘要, 原本缺失的事实）。"""
    missing = [v for v in must_keep if v and v not in summary]
    if missing:
        summary = (summary + "\n" if summary else "") + FACT_MARK + "、".join(missing)
    return summary, missing


def split_summary(summary: str) -> list[str]:
    """把已有摘要拆回片段，去掉旧的关键信息行（会由 ensure_facts_present 重新补上）。"""
    parts = []
    for p in summary.split("；"):
        p = p.strip()
        if not p:
            continue
        if FACT_MARK in p:
            p = p.split(FACT_MARK)[0].strip()
            if not p:
                continue
        parts.append(p)
    return parts


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


@dataclass
class ExtractiveSummarizer:
    clip_chars: int = 60
    keep_facts: bool = True     # False 用于演示"压缩不带关键信息"时怎么丢东西

    def summarize(self, previous_summary: str, messages: list[Message], must_keep: list[str], max_tokens: int) -> str:
        parts: list[str] = split_summary(previous_summary)
        for m in messages:
            turn = m.meta.get("turn", "?")
            if m.role == "user":
                parts.append(f"用户(第{turn}轮): {_clip(m.content, self.clip_chars)}")
            elif m.role == "assistant":
                if m.tool_calls:
                    parts.append("助手调用 " + ", ".join(tc.name for tc in m.tool_calls))
                if m.content:
                    parts.append(f"助手: {_clip(m.content, self.clip_chars)}")
            elif m.role == "tool":
                try:
                    env = json.loads(m.content)
                    status = "ok" if env.get("ok") else (env.get("error") or {}).get("kind", "error")
                except json.JSONDecodeError:
                    status = "?"
                parts.append(f"{m.tool_name}→{status}")
        # 超长时从最旧的片段开始丢——这就是"压缩有损"的具体形式；关键事实靠下面的 ensure_facts_present 兜底
        while len(parts) > 1 and estimate_tokens("；".join(parts)) > max_tokens:
            parts.pop(0)
        summary = "；".join(parts)
        if self.keep_facts:
            summary, _ = ensure_facts_present(summary, must_keep)
        return summary


_PROMPT = """把下面的对话压缩成不超过 {max_chars} 字的摘要，供后续对话参考。
必须逐字保留这些关键信息：{must_keep}
只输出摘要正文。

已有摘要：
{previous}

需要压缩的对话：
{dialog}"""


@dataclass
class LLMSummarizer:
    llm: LLMClient

    def summarize(self, previous_summary: str, messages: list[Message], must_keep: list[str], max_tokens: int) -> str:
        lines = []
        for m in messages:
            if m.role == "tool":
                lines.append(f"[工具 {m.tool_name} 返回] {_clip(m.content, 200)}")
            else:
                lines.append(f"[{m.role}] {m.content}" + (f" (调用 {', '.join(tc.name for tc in m.tool_calls)})" if m.tool_calls else ""))
        prompt = _PROMPT.format(
            max_chars=max_tokens, must_keep="；".join(must_keep) or "（无）", previous=previous_summary or "（无）", dialog="\n".join(lines)
        )
        try:
            resp = self.llm.complete("你是对话摘要器。", [Message.user(prompt)], [])
            summary = resp.text.strip()
        except Exception:  # noqa: BLE001 - 摘要失败退回抽取式，不能让主流程挂
            summary = ExtractiveSummarizer().summarize(previous_summary, messages, must_keep, max_tokens)
        summary, _ = ensure_facts_present(summary, must_keep)
        return summary
