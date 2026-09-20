"""关键事实抽取（记忆的写入侧）。

写入侧没抽到，读侧再怎么召回也是空的——排查"丢信息"时第一步就看 memory.fact_upsert 事件。

- RuleFactExtractor：正则规则，零成本、确定、可测。覆盖订单号、手机号、姓名、过敏、预算、地址等。
- LLMFactExtractor：让模型输出 JSON 事实列表，覆盖规则写不完的长尾；解析失败返回空（不能让抽取把主流程搞挂）。
- CompositeExtractor：规则优先，LLM 补充，按 key 去重。

生产上常见做法：规则跑在每一轮（便宜），LLM 抽取异步跑或者只在"用户消息较长"时跑。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..llm.base import LLMClient, Message
from .tiers import Fact


class FactExtractor(Protocol):
    def extract(self, text: str, turn: int) -> list[Fact]: ...


class NullExtractor:
    """不抽取——用于演示"没有长期记忆层"时第 20 轮为什么丢信息。"""

    def extract(self, text: str, turn: int) -> list[Fact]:
        return []


@dataclass
class Rule:
    key: str
    pattern: str
    pinned: bool = True
    multi: bool = False
    flags: int = 0

    def __post_init__(self) -> None:
        self.regex = re.compile(self.pattern, self.flags)


DEFAULT_RULES: list[Rule] = [
    Rule("订单号", r"(?:订单号?|单号|order(?:\s*(?:id|no\.?|number))?)\s*[:：#是为]?\s*([A-Za-z]{1,4}[-\s]?0*\d{3,})", multi=True, flags=re.I),
    Rule("订单号", r"(?<![A-Za-z0-9])([A-Z]{2,4}-\d{3,})(?![A-Za-z0-9])", multi=True),
    Rule("手机号", r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    Rule("姓名", r"(?:我叫|我的名字是|本人)\s*([一-鿿]{2,4})(?=[，,。！!\s]|$)"),
    Rule("过敏", r"对\s*([一-鿿A-Za-z、和]{1,12}?)\s*过敏"),
    Rule("预算", r"预算\s*(?:是|为|大概|在|约|大约)?\s*([0-9０-９.]+\s*(?:万|千|k|K|元|块|美元|刀)?)"),
    Rule("地址", r"(?:收货地址|地址|寄到|送到)\s*[:：是]?\s*([^\s，。,!！?？]{4,40})"),
    Rule("时限", r"(?:截止|最晚|需要在|必须在)\s*([^\s，。,]{2,15}?(?:前|之前|号|日|点))", pinned=False),
    Rule("忌口", r"(?:不要|别放|不吃|不能吃|讨厌)\s*([一-鿿]{1,8})(?=[，,。！!\s]|$)", pinned=False),
]


@dataclass
class RuleFactExtractor:
    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))

    def extract(self, text: str, turn: int) -> list[Fact]:
        out: list[Fact] = []
        seen: set[tuple[str, str]] = set()
        for rule in self.rules:
            for m in rule.regex.finditer(text):
                value = m.group(1).strip()
                if rule.key == "订单号":
                    from ..tools.builtin import normalize_order_id
                    value = normalize_order_id(value)
                if (rule.key, value) in seen:
                    continue
                seen.add((rule.key, value))
                out.append(Fact(rule.key, value, turn, pinned=rule.pinned, multi=rule.multi, tags=("rule",)))
        return out


_LLM_PROMPT = """从下面这条用户消息里抽取"后续对话可能需要引用"的事实。
只输出 JSON 数组，每项形如 {{"key": "订单号", "value": "SO-123", "pinned": true}}。
pinned=true 表示硬约束（订单号、姓名、联系方式、过敏/禁忌、预算、地址、时限）。
没有可抽取的事实就输出 []。不要输出任何解释。

用户消息：
{text}"""


@dataclass
class LLMFactExtractor:
    llm: LLMClient

    def extract(self, text: str, turn: int) -> list[Fact]:
        if len(text.strip()) < 4:
            return []
        try:
            resp = self.llm.complete("你是信息抽取器，只输出 JSON。", [Message.user(_LLM_PROMPT.format(text=text))], [])
            raw = resp.text.strip()
            start, end = raw.find("["), raw.rfind("]")
            items = json.loads(raw[start : end + 1]) if start >= 0 and end > start else []
        except Exception:  # noqa: BLE001 - 抽取失败不能影响主流程
            return []
        out = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("key") and it.get("value"):
                out.append(Fact(str(it["key"]), str(it["value"]), turn, pinned=bool(it.get("pinned", False)), tags=("llm",)))
        return out


@dataclass
class CompositeExtractor:
    extractors: list[FactExtractor]

    def extract(self, text: str, turn: int) -> list[Fact]:
        out: list[Fact] = []
        seen: set[tuple[str, str]] = set()
        for ex in self.extractors:
            for f in ex.extract(text, turn):
                if (f.key, f.value) not in seen:
                    seen.add((f.key, f.value))
                    out.append(f)
        return out
