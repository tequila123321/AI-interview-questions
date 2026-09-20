# Agent 生产工程实践手册（面试题驱动的完整可运行项目）

起点是一篇帖子：《一上午面试了六个 Agent 开发，全是半吊子》。面试官追问的不是概念，而是落地细节：
工具报错怎么办、参数不符合 schema 怎么办、第 20 轮丢了第 1 轮的信息怎么排查、模型死循环怎么止损、
为什么用状态机不用 Chain、分支失败怎么回滚。评论区补了两条最能分辨的追问：
**讲一次真实线上故障（发现 → 定位 → 修复）**，以及**死循环除了步数上限，有没有做同工具同参数连续调用检测**。

这个项目把每一个问题都变成了**一段可运行的代码 + 一个测试 + 一篇解释**。
全部测试不需要任何 API key（用确定性的 FakeLLM 刻画模型行为），但同一套 Agent 可以一行切换到真实 Claude 模型。

题目按原帖出现顺序编号在 [题目清单.md](题目清单.md)（1–16 题，每题标注了对应的骨架、代码、测试）。

## 快速开始

```bash
cd AI面试例题
pip install -e ".[dev]"          # 依赖只有 pydantic；anthropic 可选
python -m pytest -q              # 78 个测试，约 2 秒

python examples/01_tool_failures.py     # 工具层 9 条失败路径，看模型到底收到什么
python examples/02_memory_turn20.py     # 第 20 轮为什么忘了第 1 轮 + 自动诊断
python examples/03_loop_guard.py        # 只有步数上限 vs 同参数检测：31 步 40,970 token vs 4 步 2,792 token
python examples/04_workflow_refund.py   # 状态机：分支 / 节点重试 / Saga 回滚 / 可选节点 / 断点恢复
python examples/05_incident_replay.py   # 一次完整事故：发现 → 定位 → 修复 → 变成回归用例
python examples/live_anthropic.py       # （需要 ANTHROPIC_API_KEY）用真实模型跑同一套 Agent
```

Windows 终端如果中文乱码：`set PYTHONIOENCODING=utf-8` 后再运行。

## 项目结构

```
agentkit/
  agent.py                 Agent 主循环：把下面所有组件接起来（先读这个）
  harness.py               两套配置档：naive（事故前）/ guarded（事故后），测试与示例共用
  tools/                   工具层
    errors.py              错误分类：invalid_args / timeout / transient / permanent / invalid_result / circuit_open
    retry.py               指数退避 + 全抖动、Retry-After、熔断器
    registry.py            ToolSpec：参数模型、结果模型、超时、重试、幂等、降级、缓存
    executor.py            执行链路：校验 → 缓存 → 熔断 → 超时+重试 → 结果校验 → 降级 → 结构化信封
    builtin.py             示例工具 + 故障注入器（含事故前 v1 / 事故后 v2 两个版本）
  memory/                  记忆层
    tiers.py               三层：工作记忆 / 情景摘要 / 长期事实（pinned 永远带上）
    extractor.py           事实抽取：规则 / LLM / 组合
    compressor.py          上下文压缩 + 关键事实保底
    context_builder.py     在 token 预算内组装上下文，永不拆开工具调用组，输出构成报告
    manager.py             把以上串起来
    diagnose.py            "第 N 轮为什么不知道 X"的自动排查
  guard/                   守护层
    loop_detector.py       步数上限 + 同参数连续/累计 + 无进展 三种检测
    content_filter.py      输出审核（规则 + 可插拔 LLM 审核）与工具入参脱敏
    budget.py              每轮 token / 会话成本上限，收尾模式
    assertions.py          关键节点断言（写操作参数必须来自工具结果等）
  observability/
    trace.py               结构化事件（JSONL），一切排查和重放的基础
    replay.py              忠实重放 / what-if 重放
    badcase.py             Bad Case 文件 → 回归断言
  workflow/
    state_machine.py       节点级重试、分支边、Saga 补偿、可选节点、checkpoint 恢复、流转上限
    example_flow.py        退款流程示例
  data/
    cleaning.py            知识库清洗（归一化、模板噪声、去重、近似去重、PII）
    chunking.py            固定 / 递归 / Markdown 结构感知（父子块）
  llm/
    base.py                与厂商无关的消息模型
    fake.py                确定性假模型（脚本 / 策略）
    policies.py            刻画真实模型坏习惯的策略：customer_service / stubborn / reckless
    anthropic_client.py    Anthropic 适配器（缓存断点、错误分类、拒答映射）
badcases/                  三个回归用例（每个都对应一种线上坏结果）
examples/                  五个可运行示例 + 一个真实模型示例
tests/                     78 个测试
docs/                      解释文档（见下表）
```

## 面试题 → 代码 → 测试 → 文档

| 帖子里的追问 | 代码 | 测试 | 文档 |
|---|---|---|---|
| 模型调工具报错、参数不符合 schema、下游超时，代码层面怎么做？ | `tools/executor.py` `tools/errors.py` `tools/retry.py` | `test_tools_executor.py` `test_retry.py` | [01 工具层](docs/01_工具层.md) |
| 模型直接调 API 还是先走记忆检索？工具返回做不做格式校验？ | `memory/manager.py::build_context` `tools/executor.py`（result_model） | `test_memory.py` `test_tools_executor.py::test_result_contract_violation_*` | [02](docs/02_记忆层.md) / [01](docs/01_工具层.md) |
| 超时重试指数退避、结果 schema 校验、降级到人工兜底或默认值 | `tools/retry.py` `tools/registry.py`（fallback / handoff_on_failure） | `test_tools_executor.py` | [01 工具层](docs/01_工具层.md) |
| 第 20 轮丢失最早关键信息，排查路径？记忆分层 / 压缩 / 优先召回 | `memory/*` `memory/diagnose.py` | `test_memory.py::test_turn_20_*` | [02 记忆层](docs/02_记忆层.md) |
| 死循环：放任 token 还是最大步数？（评论：同工具同参数检测） | `guard/loop_detector.py` `agent.py`（收尾模式） | `test_loop_detector.py` `test_agent_guards.py` | [03 守护层](docs/03_守护层.md) |
| 内容安全审核校验 | `guard/content_filter.py` | `test_guards_units.py` `test_agent_guards.py::test_output_pii_*` | [03 守护层](docs/03_守护层.md) |
| 成本控制：缓存和上下文裁剪 | `guard/budget.py` `memory/context_builder.py` `llm/anthropic_client.py`（缓存断点） | `test_guards_units.py` `test_memory.py::test_context_builder_*` | [03 守护层](docs/03_守护层.md) |
| 日志重放、关键节点断言、Bad Case 闭环复盘 | `observability/*` `guard/assertions.py` `badcases/` | `test_replay_badcase.py` | [04 可观测性](docs/04_可观测性.md) |
| 为什么状态机不是线性 Chain？分支失败怎么回滚？节点级重试、单点故障隔离 | `workflow/state_machine.py` | `test_workflow.py` | [05 工作流编排](docs/05_工作流编排.md) |
| 数据层：知识库怎么清洗、按什么策略切分 | `data/cleaning.py` `data/chunking.py` | `test_data.py` | [06 数据层](docs/06_数据层.md) |
| 讲一次真实线上故障：怎么发现、怎么定位、怎么修 | `examples/05_incident_replay.py` `badcases/BC-001_*.json` | `test_replay_badcase.py::test_badcase_regression[BC-001]` | [07 线上故障复盘](docs/07_线上故障复盘.md) |

每个问题的**回答骨架**（面试官在考什么、怎么答、追问预判）集中在 [docs/00_面试问题清单与回答骨架.md](docs/00_面试问题清单与回答骨架.md)。

## 建议的过一遍方式

1. 跑五个示例，先看现象。
2. 读 `docs/00`，对照每个问题的骨架，能不看文档复述再往下。
3. 按 01 → 07 的顺序，**每篇文档配合对应的代码和测试一起读**。测试名就是行为规格。
4. 做一次 [07](docs/07_线上故障复盘.md) 末尾的演练：自己把守护层删掉，看它烧钱，再把它修回来。这样"讲一次线上故障"时，至少有一次是你亲手定位和修复过的。
5. 把 `harness.py` 里 `guarded` 的每个参数改一改，跑 `pytest`，看哪个测试红了——那就是那个参数存在的理由。

## 两个诚实的说明

- `llm/policies.py` 里的"模型"是确定性策略，用来刻画真实模型的典型坏习惯（看到 retry 就重试、不查订单就退款）。守护层的价值恰恰在于它不依赖模型是否听话，所以能这样测。真实模型用 `examples/live_anthropic.py`。
- `docs/07` 的事故 INC-0421 是按真实事故模式**构造**的、在本项目里可以完整复现的案例，不是某家公司的真实事件。面试要讲你自己的；07 末尾给了把它变成自己经历的做法。
