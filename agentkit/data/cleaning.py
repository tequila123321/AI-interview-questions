"""知识库清洗。

RAG 效果差，一半以上的锅在数据：重复文档把 top-k 占满、页眉页脚噪声进了向量、
全角半角混排导致检索不命中、过期版本和新版本并存互相矛盾。清洗顺序：

  1. 归一化：Unicode NFKC（全角→半角）、统一换行、去零宽字符、压缩空白
  2. 去模板噪声：在多数文档里都出现的行（导航栏、版权、"点击查看更多"）整行删除
  3. 丢弃空文档 / 过短文档
  4. 精确去重（哈希）
  5. 近似去重（k-shingle Jaccard）：同一篇文章的两个爬取版本
  6. 可选 PII 脱敏（知识库里不该有用户手机号）
  7. 保留元数据：来源、更新时间、版本——检索时按时间过滤、给模型引用

每一步都计数进 CleanReport：清洗不是黑盒，"删了多少、为什么删"必须能回答。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..guard.content_filter import ContentFilter


@dataclass
class Doc:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanReport:
    input_docs: int = 0
    output_docs: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_exact_dup: int = 0
    dropped_near_dup: int = 0
    boilerplate_lines: list[str] = field(default_factory=list)
    boilerplate_removed: int = 0
    pii_redactions: int = 0


_ZERO_WIDTH = re.compile(r"[​-‏  ﻿]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:                              # 连续空行压成一个
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def find_boilerplate_lines(docs: list[Doc], *, min_fraction: float = 0.5, min_docs: int = 3, min_len: int = 4) -> set[str]:
    if len(docs) < min_docs:
        return set()
    counter: Counter[str] = Counter()
    for d in docs:
        for ln in set(d.text.split("\n")):
            if len(ln) >= min_len:
                counter[ln] += 1
    return {ln for ln, c in counter.items() if c / len(docs) >= min_fraction and c >= min_docs}


def shingles(text: str, k: int = 5) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= k:
        return {compact}
    return {compact[i : i + k] for i in range(len(compact) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def clean_documents(
    docs: list[Doc],
    *,
    min_chars: int = 20,
    near_dup_threshold: float = 0.85,
    redact_pii: bool = True,
    boilerplate_min_fraction: float = 0.5,
) -> tuple[list[Doc], CleanReport]:
    report = CleanReport(input_docs=len(docs))
    normalized = [Doc(d.id, normalize_text(d.text), dict(d.meta)) for d in docs]

    boiler = find_boilerplate_lines(normalized, min_fraction=boilerplate_min_fraction)
    report.boilerplate_lines = sorted(boiler)
    pii = ContentFilter() if redact_pii else None

    seen_hashes: set[str] = set()
    kept: list[Doc] = []
    kept_shingles: list[set[str]] = []
    for d in normalized:
        if boiler:
            lines = d.text.split("\n")
            new_lines = [ln for ln in lines if ln not in boiler]
            report.boilerplate_removed += len(lines) - len(new_lines)
            d.text = "\n".join(new_lines).strip()
        if not d.text:
            report.dropped_empty += 1
            continue
        if len(d.text) < min_chars:
            report.dropped_short += 1
            continue
        if pii is not None:
            d.text, reasons = pii.redact_pii(d.text)
            report.pii_redactions += len(reasons)
        h = hashlib.sha256(d.text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            report.dropped_exact_dup += 1
            continue
        sh = shingles(d.text)
        if any(jaccard(sh, s) >= near_dup_threshold for s in kept_shingles):
            report.dropped_near_dup += 1
            continue
        seen_hashes.add(h)
        kept_shingles.append(sh)
        d.meta.setdefault("content_hash", h[:16])
        kept.append(d)
    report.output_docs = len(kept)
    return kept, report
