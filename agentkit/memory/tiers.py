"""记忆分层。

三层，各自回答一个问题：
1. 工作记忆（working）：最近几轮对话原文。回答"刚才说到哪了"。有 token 上限，超了就压缩。
2. 情景摘要（episodic summary）：更早对话的压缩摘要。回答"之前大概聊了什么"。有损。
3. 长期事实（long-term facts）：从对话里抽出来的结构化事实（订单号、过敏、预算……）。
   回答"用户的硬约束是什么"。无损、可召回、可跨会话持久化。

"第 20 轮丢了第 1 轮的关键信息"通常就是：第 1 轮原文被压缩进摘要时丢了，而这个信息又没有
落到第 3 层。所以第 3 层的 pinned 事实在构建上下文时永远优先带上，不参与预算竞争。

召回打分刻意用了最朴素的字符 2-gram 重叠（不依赖向量库），生产上可以换成 embedding，
但"pinned 永远带上 + 其余按相关性召回"的结构不变。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]*")
_CJK = re.compile(r"[一-鿿]")


def bigrams(text: str) -> set[str]:
    text = text.lower()
    grams: set[str] = set(_WORD.findall(text))
    cjk = "".join(_CJK.findall(text))
    grams.update(cjk[i : i + 2] for i in range(len(cjk) - 1))
    if len(cjk) == 1:
        grams.add(cjk)
    return grams


@dataclass
class Fact:
    key: str
    value: str
    source_turn: int
    pinned: bool = False           # 硬约束：构建上下文时永远带上
    multi: bool = False            # 同 key 可并存多个值（例如多个订单号）
    tags: tuple[str, ...] = ()
    hits: int = 0
    last_used_turn: int = 0

    def render(self) -> str:
        return f"{self.key}: {self.value}（第 {self.source_turn} 轮）"


class LongTermMemory:
    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}

    def upsert(self, fact: Fact) -> Fact:
        key = fact.key
        existing = self.facts.get(key)
        if existing and existing.value != fact.value and fact.multi:
            n = 2
            while f"{fact.key}#{n}" in self.facts and self.facts[f"{fact.key}#{n}"].value != fact.value:
                n += 1
            key = f"{fact.key}#{n}"
            existing = self.facts.get(key)
        if existing:
            if existing.value != fact.value:          # 值变了（例如用户改了预算）：覆盖并更新来源轮次
                existing.value = fact.value
                existing.source_turn = fact.source_turn
            existing.pinned = existing.pinned or fact.pinned
            return existing
        fact.key = key
        self.facts[key] = fact
        return fact

    def all(self) -> list[Fact]:
        return sorted(self.facts.values(), key=lambda f: (f.source_turn, f.key))

    def pinned(self) -> list[Fact]:
        return [f for f in self.all() if f.pinned]

    def recall(self, query: str, k: int = 5, current_turn: int = 0) -> list[Fact]:
        """pinned 全部返回；其余按与 query 的重叠度取 top-k（重叠为 0 的不返回）。"""
        q = bigrams(query)
        chosen = list(self.pinned())
        scored: list[tuple[float, Fact]] = []
        for f in self.all():
            if f.pinned:
                continue
            overlap = len(q & bigrams(f.key + " " + f.value))
            if overlap == 0:
                continue
            recency = 1.0 / (1 + max(0, current_turn - f.source_turn))
            scored.append((overlap + 0.5 * recency + 0.1 * f.hits, f))
        scored.sort(key=lambda t: -t[0])
        chosen.extend(f for _, f in scored[:k])
        for f in chosen:
            f.hits += 1
            f.last_used_turn = current_turn
        return chosen
