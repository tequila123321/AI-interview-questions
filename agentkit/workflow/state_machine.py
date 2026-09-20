"""工作流状态机（对比线性 Chain）。

为什么不是线性 Chain：
- Chain 只有"下一步"，没有"根据状态决定下一步"（分支）、"回到某一步"（循环）、"从某一步继续"（恢复）；
- Chain 一个环节抛异常整条链就断了，没有"这一步失败了该怎么办"的语义。

状态机把这些都变成节点上的显式属性：
  retry        节点级重试策略（只重试 retryable 的异常）
  compensate   回滚动作：后续必需节点失败时，已完成的有副作用节点按逆序执行补偿（Saga 模式）；
               补偿本身失败 → 结果状态 compensation_failed + critical 告警（人工介入）
  optional     可选节点失败不终止流程，只标记 degraded（通知发送失败不该让退款失败）
  timeout_s    节点超时
边（edges）可以是固定的下一个节点名，也可以是根据 state 判定的函数——这就是"分支判定逻辑"。

每个节点执行后写 checkpoint（state + 下一个节点），进程崩了可以 resume 而不是从头再来。
max_transitions 防止边判定写出死循环。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..observability.trace import Tracer
from ..tools.executor import run_with_timeout
from ..tools.retry import NO_RETRY, RetryPolicy, run_with_retry

END = "END"
NodeFn = Callable[[dict[str, Any]], dict[str, Any] | None]
Edge = str | Callable[[dict[str, Any]], str]


@dataclass
class Node:
    name: str
    run: NodeFn
    retry: RetryPolicy = NO_RETRY            # RetryPolicy 是 frozen 的，共享同一个实例是安全的
    compensate: Callable[[dict[str, Any]], None] | None = None
    optional: bool = False
    timeout_s: float | None = None


@dataclass
class Workflow:
    name: str
    start: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def add(self, node: Node, next_edge: Edge) -> "Workflow":
        self.nodes[node.name] = node
        self.edges[node.name] = next_edge
        return self

    def next_of(self, name: str, state: dict[str, Any]) -> str:
        edge = self.edges[name]
        return edge(state) if callable(edge) else edge


class CheckpointStore(Protocol):
    def save(self, run_id: str, checkpoint: dict[str, Any]) -> None: ...
    def load(self, run_id: str) -> dict[str, Any] | None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def save(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        self.data[run_id] = copy.deepcopy(checkpoint)

    def load(self, run_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.data.get(run_id))


@dataclass
class WorkflowResult:
    status: str                      # completed | failed | compensated | compensation_failed
    state: dict[str, Any]
    path: list[str] = field(default_factory=list)
    failed_node: str | None = None
    error: str = ""
    compensated: list[str] = field(default_factory=list)
    compensation_failed: list[str] = field(default_factory=list)   # 补偿本身失败的节点：必须人工介入
    degraded: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)


class WorkflowRunner:
    def __init__(
        self,
        workflow: Workflow,
        *,
        tracer: Tracer | None = None,
        checkpoints: CheckpointStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_transitions: int = 50,
    ):
        self.wf = workflow
        self.tracer = tracer or Tracer()
        self.checkpoints = checkpoints or InMemoryCheckpointStore()
        self.sleep = sleep
        self.max_transitions = max_transitions

    # ---- 入口 ----
    def run(self, state: dict[str, Any], run_id: str = "run") -> WorkflowResult:
        return self._loop(dict(state), self.wf.start, run_id, path=[], completed_with_effects=[], degraded=[], attempts={})

    def resume(self, run_id: str) -> WorkflowResult:
        cp = self.checkpoints.load(run_id)
        if cp is None:
            raise KeyError(f"no checkpoint for {run_id}")
        return self._loop(cp["state"], cp["next"], run_id, cp["path"], cp["completed_with_effects"], cp["degraded"], cp["attempts"])

    # ---- 核心 ----
    def _loop(self, state, current, run_id, path, completed_with_effects, degraded, attempts) -> WorkflowResult:
        transitions = 0
        while current != END:
            transitions += 1
            if transitions > self.max_transitions:
                return self._fail(state, path, current, f"超过最大流转次数 {self.max_transitions}（边判定可能形成死循环）", completed_with_effects, degraded, attempts)
            node = self.wf.nodes[current]
            ok, error, n = self._run_node(node, state)
            attempts[node.name] = n
            path.append(node.name)
            if ok:
                if node.compensate is not None:
                    completed_with_effects.append(node.name)
            elif node.optional:
                degraded.append(node.name)
                self.tracer.event("workflow.node", node=node.name, status="degraded", error=error)
            else:
                return self._fail(state, path, node.name, error, completed_with_effects, degraded, attempts)
            current = self.wf.next_of(node.name, state)
            self.checkpoints.save(run_id, {"state": state, "next": current, "path": path,
                                           "completed_with_effects": completed_with_effects, "degraded": degraded, "attempts": attempts})
        return WorkflowResult("completed", state, path, degraded=degraded, attempts=attempts)

    def _run_node(self, node: Node, state: dict[str, Any]) -> tuple[bool, str, int]:
        n = 0

        def attempt() -> None:
            nonlocal n
            n += 1
            with self.tracer.span("workflow.node", node=node.name, attempt=n) as extra:
                try:
                    updates = run_with_timeout(lambda: node.run(state), node.timeout_s)
                    if updates:
                        state.update(updates)
                    extra["status"] = "ok"
                except Exception as exc:
                    extra["status"] = "error"
                    extra["error"] = f"{type(exc).__name__}: {exc}"
                    raise

        try:
            run_with_retry(attempt, node.retry, sleep=self.sleep,
                           on_retry=lambda exc, k, d: self.tracer.event("workflow.retry", node=node.name, attempt=k, delay_s=round(d, 3), error=str(exc)))
            return True, "", n
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}", n

    def _fail(self, state, path, failed_node, error, completed_with_effects, degraded, attempts) -> WorkflowResult:
        compensated: list[str] = []
        compensation_failed: list[str] = []
        for name in reversed(completed_with_effects):        # Saga：逆序补偿
            node = self.wf.nodes[name]
            try:
                node.compensate(state)  # type: ignore[misc]
                compensated.append(name)
                self.tracer.event("workflow.compensate", node=name, status="ok")
            except Exception as exc:  # noqa: BLE001 - 补偿失败必须告警，人工介入
                compensation_failed.append(name)
                self.tracer.event("workflow.compensate", node=name, status="failed", error=str(exc))
                self.tracer.event("alert", severity="critical", reason=f"compensation failed for {name}: {exc}")
        if compensation_failed:
            status = "compensation_failed"        # 部分或全部补偿没做成：系统处于不一致状态，结果上必须看得出来
        elif compensated:
            status = "compensated"
        else:
            status = "failed"
        self.tracer.event("workflow.failed", failed_node=failed_node, error=error, compensated=compensated, compensation_failed=compensation_failed)
        return WorkflowResult(status, state, path, failed_node, error, compensated, compensation_failed, degraded, attempts)
