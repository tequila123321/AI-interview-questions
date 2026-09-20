"""工具定义与注册表。

一个工具不只是"一个函数"，它是一份契约：
- args_model：模型必须给出什么参数（pydantic 校验，失败把字段级错误还给模型）
- result_model：下游必须返回什么结构（校验失败 = 契约被破坏，走降级 + 告警，绝不把垃圾喂给模型）
- timeout_s / retry / idempotent：失败时的行为边界（非幂等操作不自动重试）
- fallback / handoff_on_failure：最终失败时的降级策略（默认值 or 转人工）
- cache_ttl_s：幂等读操作的会话级缓存（同参数不重复打下游，也天然抑制一部分"重复调用"）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from .errors import ToolError
from .retry import RetryPolicy

Handler = Callable[[BaseModel], Any]
Fallback = Callable[[BaseModel, ToolError], Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    result_model: type[BaseModel] | None = None
    timeout_s: float | None = 5.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = True
    fallback: Fallback | None = None
    handoff_on_failure: bool = False
    cache_ttl_s: float = 0.0

    def schema(self) -> dict[str, Any]:
        """给模型看的 JSON schema（Anthropic tools 格式；OpenAI 只需再包一层 function）。"""
        input_schema = self.args_model.model_json_schema()
        input_schema.setdefault("type", "object")
        input_schema["additionalProperties"] = False   # 配合 strict 模式，杜绝模型塞未知字段
        return {"name": self.name, "description": self.description, "input_schema": input_schema}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {}
        for s in specs or []:
            self.register(s)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        # 顺序固定：工具列表是提示词缓存前缀的一部分，顺序抖动会让缓存全失效
        return [self._specs[n].schema() for n in sorted(self._specs)]
