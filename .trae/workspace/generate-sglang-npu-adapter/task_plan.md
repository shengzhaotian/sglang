# Task Plan: SGLang NPU模型适配技能套件开发

## Goal
设计并开发一套适用于Trae系统的技能(Skill)套件，用于在NPU设备环境中实现基于SGLang框架的新模型快速适配功能，实现不低于90%的任务完成率。

## Current Phase
Phase 6

## Phases

### Phase 1: 需求分析与架构设计
- [x] 验证Trae的Subagent调用机制
- [x] 确认上下文隔离特性
- [x] 确定正确的技能架构模式
- [x] 细化三个Subagent的职责边界
- [x] 设计数据传递协议
- [x] 定义输入输出规范
- [x] 验证Subagent工具能力（无Skill、无AskUserQuestion）
- [x] 验证并行Subagent文件系统行为
- [x] 验证Subagent实例独立性（每次调用都是全新实例）
- [x] 重新设计Subagent划分逻辑（Agent 2按需调用）
- [x] 补充用户交互设计
- [x] 补充错误处理机制
- [x] 修正工作空间位置
- **Status:** complete

### Phase 2: 知识库构建
- [x] 收集SGLang模型适配相关知识
  - [x] LLM架构基础知识
  - [x] SGLang模型模式与代码结构
  - [x] NPU特定实现（attention/rope/moe）
  - [x] 服务化运行与验证测试
- [x] 设计知识库内容详细规划
  - [x] 定义知识库优先级排序（P0/P1/P2）
  - [x] 编写Agent 1知识库详细大纲（7个文件）
  - [x] 编写Agent 2知识库详细大纲（5个文件）
  - [x] 编写Agent 3知识库详细大纲（4个文件）
  - [x] 编写Shared知识库详细大纲（3个文件）
- [x] 扩展Agent 2输入设计
  - [x] 增加current_adapter_code.md输入
  - [x] 增加original_model_code.md输入（可选）
  - [x] 增加previous_fixes.md输入（可选）
- [x] 设计结构化修复指令格式
  - [x] 定义修复类型（REPLACE_BLOCK/INSERT_BEFORE/INSERT_AFTER/DELETE_BLOCK/MODIFY_CONFIG）
  - [x] 定义主Skill执行修复的流程
- [x] 更新错误处理机制
  - [x] Debug迭代次数从3次调整为10次
  - [x] 增加渐进式迭代策略
- [x] 创建references/目录下的知识文档（共19个文件）
- **Status:** complete

### Phase 3: Subagent提示模板开发
- [x] Agent 1: 模型架构分析提示模板 (prompts/model_analyzer.md)
- [x] Agent 2: Debug工程师提示模板 (prompts/debug_engineer.md)
- [x] Agent 3: 测试验证提示模板 (prompts/test_validator.md)
- [x] 设计Agent间的反馈循环机制（通过文件系统传递，主Skill协调）
- **Status:** complete

### Phase 4: 主Skill开发
- [x] 编写SKILL.md主入口
- [x] 实现工作流编排逻辑
- [x] 实现用户交互设计
- [x] 实现错误处理和回退机制
- **Status:** complete

### Phase 5: 辅助脚本开发
- [x] 环境检查脚本 (scripts/check_environment.py)
- [x] 测试验证脚本 (scripts/run_tests.py)
- [x] 报告生成脚本 (scripts/generate_report.py)
- **Status:** complete

### Phase 6: 测试与验证
- [ ] 使用真实模型测试技能套件
- [ ] 验证各Agent协作流程
- [ ] 测量任务完成率
- [ ] 修复发现的问题
- **Status:** pending

## Key Questions
1. 如何确保Agent 2能够理解Agent 1的分析报告并正确执行适配？
2. Agent 3验证失败时，如何高效地将问题反馈给Agent 2？
3. 如何处理NPU特定的问题（如attention backend选择）？
4. 如何支持增量式适配（用户只修改部分代码）？
5. 如何处理模型权重大小和内存限制？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用内置subagent类型 | Task工具只支持 `search` 和 `general_purpose_task` 两种类型 |
| 采用提示模板模式 | 参考 `subagent-driven-development` 的官方实现方式 |
| 文件系统作为数据传递媒介 | Subagent内存上下文隔离，但共享文件系统 |
| 三阶段Agent架构 | 分析→适配→验证，符合模型适配的技术逻辑 |
| Agent 2输入扩展 | 增加current_adapter_code.md等输入，提供完整上下文 |
| 结构化修复指令 | 主Skill严格按照fix_instructions.md执行，避免歧义 |
| Debug迭代10次 | 模型适配可能遇到多种问题，需要足够的迭代空间 |
| 渐进式迭代策略 | 分阶段处理不同复杂度的错误 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Task(subagent_type="custom") | 1 | 只能使用内置类型，改用提示模板模式 |
| .trae/agents/ 不被识别 | 1 | 该目录不会被自动识别，改用prompts/目录存放模板 |

## Notes
- Subagent内存完全隔离，但共享文件系统
- 主Skill需要读取提示模板后填充参数传递给Task工具
- 同类型多个Subagent可以并行执行
- Agent 2的修复指令采用结构化格式，主Skill严格执行
- 知识库按Agent职责分离，按需加载
