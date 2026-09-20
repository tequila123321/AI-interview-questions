"""agentkit —— 一个把"生产级 Agent 该有的工程件"拆成独立可测模块的教学项目。

模块与面试题的对应关系（详见 docs/00_面试问题清单与回答骨架.md）：

- agentkit.tools          工具层：参数/结果 schema 校验、错误分类、指数退避、熔断、降级、结果缓存
- agentkit.memory         记忆层：三层记忆、上下文压缩、关键信息优先召回、"第 20 轮丢信息"诊断
- agentkit.guard          守护层：死循环检测（步数 + 同工具同参数）、内容审核、预算控制、关键节点断言
- agentkit.observability  可观测性：结构化 trace、日志重放、Bad Case 闭环
- agentkit.workflow       工作流：状态机、节点级重试、分支回滚（Saga 补偿）、可选节点隔离故障
- agentkit.data           数据层：知识库清洗、切分策略
- agentkit.agent          把以上组件接成一个可运行的 Agent 主循环
"""

__version__ = "0.1.0"
