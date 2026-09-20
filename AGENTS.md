# AGENTS.md · 给 Codex / 其他 agent 的交接说明

本文件由 Claude Code 于 2026-09-20 写下，记录这个仓库是什么、已经做了什么、怎么验证、哪些边界不要越过。

## 这个仓库是什么

来源是一篇小红书帖子《一上午面试了六个 Agent 开发，全是半吊子》（作者 月亮向北）和两条评论：
- 评论 1：把帖子里的问题全部喂给 Claude Code，让它出完整实践文档 + 构建完整项目 + 加解释。
- 评论 2（MaxwellFDE）：再加两条最能分辨的追问——讲一次真实线上故障（发现 → 定位 → 修复）；死循环除了步数上限，要做同工具同参数连续调用检测。

于是帖子里的每个追问都被做成了**可运行代码 + 测试 + 中文解释**。题目按原帖顺序编号在 `题目清单.md`（16 题）；每题的回答骨架在 `docs/00_面试问题清单与回答骨架.md`；`docs/01`–`docs/07` 逐层解释。

## 已完成的工作（全部在一次会话里完成）

| 模块 | 位置 | 做了什么 |
|---|---|---|
| 工具层 | `agentkit/tools/` | 错误分类（invalid_args / timeout / transient / permanent / invalid_result / circuit_open）、指数退避 + 全抖动、Retry-After 单独封顶、熔断器、pydantic 参数与结果校验、幂等才重试、按关键程度降级（默认值 / 转人工）、会话级结果缓存、结构化信封（含 `retryable` 和 `hint`） |
| 记忆层 | `agentkit/memory/` | 三层记忆（工作 / 摘要 / 长期事实）、规则 + LLM 事实抽取、pinned 事实永远进上下文、压缩带 must_keep 并用代码校验补回、预算内组装且不拆工具调用组、`diagnose_missing_fact` 把"第 N 轮为什么不知道 X"的排查路径写成代码 |
| 守护层 | `agentkit/guard/` | 死循环四层检测（同参数连续 2 warn / 3 stop、累计 4、结果无进展 4、max_rounds 10）+ 收尾模式、内容审核（出向 + 工具入参脱敏）、三级预算与收尾、关键节点断言 |
| 可观测性 | `agentkit/observability/` | JSONL 结构化 trace、忠实重放 / what-if 重放、Bad Case 文件 → 回归（`badcases/`） |
| 工作流 | `agentkit/workflow/` | 状态机：节点级重试、分支边、Saga 逆序补偿（冲正不删除；补偿失败有 `compensation_failed` 状态）、可选节点降级、checkpoint 恢复、流转上限 |
| 数据层 | `agentkit/data/` | 七步清洗带计数报告、固定 / 递归 / Markdown 父子块切分 |
| 主循环与组装 | `agentkit/agent.py`、`agentkit/harness.py` | 把上面接成一个 Agent；`naive`（事故前）/ `guarded`（事故后）两套配置档 |
| 模型 | `agentkit/llm/` | 厂商无关消息模型；FakeLLM 确定性策略（customer_service / stubborn / reckless）刻画模型坏习惯；Anthropic 适配器（默认 `claude-opus-5`，系统提示分两块，只在稳定块打缓存断点） |
| 示例 | `examples/01..05_*.py` | 工具失败路径、第 20 轮记忆、死循环对比、退款状态机、事故 INC-0421 全流程复现 |
| 文档 | `docs/00`–`docs/07` | 回答骨架、各层解释、构造事故复盘（含一小时演练与两分钟讲述模板） |

一次独立审稿抓出并已修复 6 处：缓存断点包住了会变的记忆块、Retry-After 被退避上限吃掉、超时线程拖住进程退出（改 daemon 线程）、补偿部分失败被吞、预算分支死代码、文档用 `max_steps` 而字段实为 `max_rounds`。

## 怎么验证

```bash
pip install -e ".[dev]"            # 依赖只有 pydantic；anthropic 可选
python -m pytest -q                # 期望 78 passed，约 2 秒，不需要任何 API key
python examples/05_incident_replay.py
```

Windows 终端先 `set PYTHONIOENCODING=utf-8`，否则中文乱码。

实测基准（改动守护层后请复核这些数字，`docs/03`、`docs/07` 引用了它们）：同一个固执模型 + 同一个上游故障，`naive` 31 次模型调用 / 40,970 token，`guarded` 4 次 / 2,792 token，what-if 重放 2,577 token。

## 约定与边界

- 测试名就是行为规格；改守护层参数后必须跑 `tests/test_replay_badcase.py`，Bad Case 回归红了说明把某个事故的坑挖开了。
- `llm/policies.py` 里的"模型"是确定性策略，不是真模型；真模型只走 `examples/live_anthropic.py`（会计费）。
- `docs/07` 的事故 INC-0421 是**构造**的可复现案例，不是真实事件。任何文档不得把它写成真实经历。
- 与 Anthropic API 相关的代码只在 `agentkit/llm/anthropic_client.py`；模型 ID 用 `claude-opus-5`，不要加日期后缀。
- 文档里的每个文件名、函数名、测试名、数字都要能在代码里找到；改代码时同步改文档。
- 生产要求但本项目**未实现**（文档已标注）：补偿幂等、checkpoint 落数据库、embedding 召回、异步 I/O 代替线程超时、长期事实跨会话持久化。

## 适合交给 Codex 的下一步

1. 独立复审：逐条核对 `docs/00`–`docs/07` 的技术说法与代码是否一致，重点是硬约束（缓存前缀语义、幂等与重试、Saga 补偿、线程超时）。
2. 扩展题库：`题目清单.md` 加新题时，同步在 `docs/00` 加骨架、在代码/测试里加证据，保持"题 → 骨架 → 代码 → 测试"四件套。
3. 用真实模型跑一遍 `examples/live_anthropic.py`，把 trace 里的 `cache_read_tokens` 作为缓存布局是否生效的证据补进 `docs/02` 第 8 节。

## Git

- 本地：`C:\Users\zhich\AI面试例题`，分支 `main`
- 远端：https://github.com/tequila123321/AI-interview-questions （公开；GitHub 不接受中文仓库名，中文名在描述和 README 标题里）
- 提交身份只在本仓库本地配置（`tequila123321` + GitHub noreply 邮箱），未改全局
- `.gitattributes` 统一 LF；`.gitignore` 排除 `examples/out/`、缓存和 egg-info
