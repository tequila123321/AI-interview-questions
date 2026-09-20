"""切分策略。

没有万能的 chunk size；有的是"按什么策略切、为什么"：

- chunk_fixed      固定长度 + 重叠。最简单、最稳，但会把一句话切成两半；重叠是为了补这一刀。
- chunk_recursive  按分隔符递归（段落 → 句子 → 词），尽量在语义边界切；超长段落再退到固定切。
- chunk_markdown   结构感知：按标题切、把标题路径写进元数据（检索命中"退款政策 > 特殊情况"比命中一段裸文本有用得多），
                   长节内部再递归切；同时产出父子关系：小块用来检索命中，父块（整节）用来喂模型。

评估怎么做：拿一组真实问题 + 标注的命中段落，对比不同策略的 recall@k；chunk 越小召回越准但上下文越碎，
一般 200–500 token 起步，按评估调。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..tokens import estimate_tokens
from .cleaning import Doc


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def _pack(pieces: list[str], size_tokens: int, joiner: str) -> list[str]:
    """把小片段按顺序装进不超过 size_tokens 的块。"""
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in pieces:
        t = estimate_tokens(p)
        if cur and cur_tokens + t > size_tokens:
            out.append(joiner.join(cur))
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += t
    if cur:
        out.append(joiner.join(cur))
    return out


def chunk_fixed(doc: Doc, size_tokens: int = 300, overlap_tokens: int = 50) -> list[Chunk]:
    text = doc.text
    # 用字符近似：CJK 1 字≈1 token，这里按 estimate 的比例反推字符步长
    ratio = max(1.0, len(text) / max(1, estimate_tokens(text)))
    size = max(1, int(size_tokens * ratio))
    overlap = int(overlap_tokens * ratio)
    chunks: list[Chunk] = []
    start = 0
    i = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(Chunk(f"{doc.id}#{i}", doc.id, text[start:end], i, {"strategy": "fixed", "start": start, "end": end, **doc.meta}))
        i += 1
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", " ")


def _split_recursive(text: str, size_tokens: int, separators: tuple[str, ...]) -> list[str]:
    if estimate_tokens(text) <= size_tokens or not separators:
        if estimate_tokens(text) <= size_tokens:
            return [text]
        return [c.text for c in chunk_fixed(Doc("_", text), size_tokens, 0)]   # 没有分隔符了：退到固定切
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, size_tokens, rest)
    parts = [p + (sep if sep in ("。", "！", "？", "；") else "") for p in text.split(sep) if p.strip()]
    pieces: list[str] = []
    for p in parts:
        pieces.extend(_split_recursive(p, size_tokens, rest) if estimate_tokens(p) > size_tokens else [p])
    return _pack(pieces, size_tokens, "" if sep in ("。", "！", "？", "；") else sep)


def chunk_recursive(doc: Doc, size_tokens: int = 300, separators: tuple[str, ...] = DEFAULT_SEPARATORS) -> list[Chunk]:
    pieces = _split_recursive(doc.text, size_tokens, separators)
    return [Chunk(f"{doc.id}#{i}", doc.id, p.strip(), i, {"strategy": "recursive", **doc.meta}) for i, p in enumerate(pieces) if p.strip()]


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(doc: Doc, size_tokens: int = 300) -> list[Chunk]:
    """按标题层级切；每节是一个父块，节内超长再递归切成子块（parent_id 指向父块）。"""
    sections: list[tuple[list[str], list[str]]] = []   # (heading_path, lines)
    path: list[str] = []
    cur_lines: list[str] = []
    for line in doc.text.split("\n"):
        m = _HEADING.match(line)
        if m:
            if cur_lines and "".join(cur_lines).strip():
                sections.append((list(path), cur_lines))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines and "".join(cur_lines).strip():
        sections.append((list(path), cur_lines))

    chunks: list[Chunk] = []
    idx = 0
    for heading_path, lines in sections:
        body = "\n".join(lines).strip()
        title = " > ".join(heading_path)
        parent_id = f"{doc.id}#{idx}"
        parent_text = (title + "\n" if title else "") + body
        chunks.append(Chunk(parent_id, doc.id, parent_text, idx, {"strategy": "markdown", "heading_path": heading_path, "role": "parent", **doc.meta}))
        idx += 1
        if estimate_tokens(body) > size_tokens:
            for piece in _split_recursive(body, size_tokens, DEFAULT_SEPARATORS):
                chunks.append(Chunk(f"{doc.id}#{idx}", doc.id, (title + "\n" if title else "") + piece.strip(), idx,
                                    {"strategy": "markdown", "heading_path": heading_path, "role": "child", "parent_id": parent_id, **doc.meta}))
                idx += 1
    return chunks
