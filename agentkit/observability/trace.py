"""结构化 trace：Agent 里每一个值得回放的动作都是一条事件。

为什么不用普通日志：
- 事件带 run_id / turn / step / kind，可以按会话拉出完整时间线（定位死循环、成本异常时第一步就是这个）；
- 记录了模型请求指纹、模型响应、工具入参/出参（哈希 + 摘要）、上下文构建报告、守护判定，
  所以可以离线重放（observability/replay.py）和做"第 20 轮为什么丢信息"的诊断（memory/diagnose.py）；
- JSONL 追加写，一行一事件，任何工具都能处理。

事件种类（kind）约定：
  turn.start / turn.end            一轮用户对话
  context.built                    本次模型请求的上下文构成（各段 token、召回的事实、被裁掉的消息数）
  llm.request / llm.response       模型调用
  tool.call / tool.attempt / tool.result
  guard.loop / guard.budget / guard.content / guard.assertion
  memory.fact_upsert / memory.compress
  workflow.node                    工作流节点执行
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def stable_hash(obj: Any) -> str:
    """对 JSON 可序列化对象求稳定哈希（键排序），用于比较"结果是否变化"。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        s = repr(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class Tracer:
    def __init__(self, run_id: str | None = None, path: str | Path | None = None, clock=time.time):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self.ctx: dict[str, Any] = {}
        self._seq = 0
        self._clock = clock
        self._fh = open(path, "a", encoding="utf-8") if path else None

    # ---- 写 ----
    def bind(self, **ctx: Any) -> None:
        """设置后续事件默认携带的上下文字段（turn / step）。传 None 表示清除。"""
        for k, v in ctx.items():
            if v is None:
                self.ctx.pop(k, None)
            else:
                self.ctx[k] = v

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        ev: dict[str, Any] = {"seq": self._seq, "ts": self._clock(), "run_id": self.run_id, "kind": kind}
        ev.update(self.ctx)
        ev.update(fields)
        self.events.append(ev)
        if self._fh:
            self._fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        return ev

    @contextmanager
    def span(self, kind: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """带耗时的事件。with 块里可以往 yield 出来的 dict 里塞额外字段。"""
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield extra
        finally:
            fields["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            fields.update(extra)
            self.event(kind, **fields)

    # ---- 读 ----
    def find(self, kind: str | None = None, **match: Any) -> list[dict[str, Any]]:
        out = []
        for ev in self.events:
            if kind is not None and ev.get("kind") != kind:
                continue
            if all(ev.get(k) == v for k, v in match.items()):
                out.append(ev)
        return out

    def count(self, kind: str, **match: Any) -> int:
        return len(self.find(kind, **match))

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for ev in self.events:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
