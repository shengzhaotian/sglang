# NPU模型适配Sub Agent与Skill组件设计方案

## 一、概述

本方案设计并实现一套完整的sub agent及配套skill组件，用于在NPU设备环境下基于sglang框架的新模型快速适配流程。设计严格遵循上下文隔离原则与上下文窗口长度约束。

## 二、Sub Agent设计

### 2.1 模型分析代理 (model-analyzer)

**职责**：模型结构解析与兼容性评估

**触发条件**：当需要分析新模型架构、评估NPU兼容性、解析模型配置时自动调用

**输入件**：
- 模型路径/HuggingFace模型ID
- 模型类型（LLM/VLM/Embedding等）

**输出件**：
- 模型架构分析报告（JSON格式）
- 兼容性评估结果
- 所需算子清单
- 依赖项检查结果
- 交互请求（如需用户确认）

**关键流程**：
1. 解析模型config.json
2. 识别模型架构类型
3. 检查是否为已知架构变体
4. 分析attention类型（MHA/MLA/GQA等）
5. 检查MoE结构
6. 生成算子需求清单
7. 评估NPU兼容性

**交互点**：
- 模型信息确认：检测到多个可能模型类型时请求确认
- 架构识别确认：架构与已知类型相似但不完全匹配时请求确认
- 兼容性风险确认：存在不支持算子时展示风险并请求确认

**配套Skills**：
- `model-config-parser`：模型配置解析
- `architecture-detector`：架构类型检测
- `compatibility-checker`：兼容性检查

### 2.2 NPU适配代理 (npu-adapter)

**职责**：处理设备特定优化与算子映射

**触发条件**：当需要NPU算子映射、量化配置、设备特定优化时自动调用

**输入件**：
- 模型分析报告
- 目标NPU硬件类型（A2/A3）
- 部署模式（PD Mixed/PD Separation）

**输出件**：
- NPU适配配置文件
- 算子映射表
- 量化策略建议
- 环境变量配置
- 交互请求（如需用户决策）

**关键流程**：
1. 分析NPU算子支持情况
2. 生成算子映射方案
3. 确定量化策略（W8A8/W4A8/BF16等）
4. 配置attention backend
5. 设置MoE通信后端
6. 生成部署配置

**交互点**：
- 量化策略选择：多种量化方案可行时请求用户选择
- 部署模式确认：硬件支持多种部署模式时请求确认
- 环境变量确认：生成非标准配置时请求确认

**配套Skills**：
- `npu-operator-mapper`：NPU算子映射
- `quantization-strategy`：量化策略生成
- `npu-config-generator`：NPU配置生成

### 2.3 性能优化代理 (perf-optimizer)

**职责**：推理效率提升与资源占用优化

**触发条件**：当需要性能基准测试、配置优化、资源调优时自动调用

**输入件**：
- NPU适配配置
- 性能目标（延迟/吞吐量）
- 硬件资源信息

**输出件**：
- 优化配置建议
- 性能基准测试报告
- 资源使用预测
- 交互请求（如需用户确认）

**关键流程**：
1. 分析性能需求（低延迟/高吞吐）
2. 推荐部署模式
3. 优化TP/DP配置
4. 配置CUDA Graph batch sizes
5. 调整内存分配策略
6. 运行基准测试

**交互点**：
- 性能目标澄清：未明确性能优先级时请求用户选择
- 资源分配确认：资源需求接近硬件上限时请求确认
- 基准测试确认：准备执行耗时测试前请求确认

**配套Skills**：
- `perf-benchmark`：性能基准测试
- `resource-optimizer`：资源优化
- `config-tuner`：配置调优

## 三、Skill设计

### 3.1 模型分析Skills

#### model-config-parser
- 功能：解析HuggingFace模型配置
- 参数：model_path, model_type
- 返回：配置字典、架构名称、隐藏层维度等

#### architecture-detector
- 功能：检测模型架构类型
- 参数：config_dict
- 返回：架构类型、attention类型、是否MoE等

#### compatibility-checker
- 功能：检查NPU兼容性
- 参数：architecture_info, target_hardware
- 返回：兼容性状态、不支持的算子列表

### 3.2 NPU适配Skills

#### npu-operator-mapper
- 功能：生成NPU算子映射
- 参数：required_operators, hardware_type
- 返回：算子映射表、fallback方案

#### quantization-strategy
- 功能：生成量化策略
- 参数：model_info, performance_target
- 返回：量化类型、精度影响评估

#### npu-config-generator
- 功能：生成NPU部署配置
- 参数：adaptation_info, deploy_mode
- 返回：启动命令、环境变量

### 3.3 性能优化Skills

#### perf-benchmark
- 功能：运行性能基准测试
- 参数：config, dataset_spec
- 返回：TTFT、TPOT、吞吐量指标

#### resource-optimizer
- 功能：优化资源配置
- 参数：benchmark_results, hardware_info
- 返回：优化建议

#### config-tuner
- 功能：自动调优配置
- 参数：performance_target, constraints
- 返回：优化后的配置

## 四、公共知识文档

### 4.1 NPU设备技术规格
- Atlas 800I A2规格
- Atlas 800I A3规格
- 内存带宽、算力对比

### 4.2 SGLang框架API参考
- 模型注册接口
- Attention Backend接口
- 量化接口

### 4.3 模型适配最佳实践
- 模型文件结构规范
- 测试用例编写规范
- 性能基准测试流程

### 4.4 常见问题解决方案
- 算子不支持处理
- 内存溢出处理
- 精度问题排查

## 五、人机交互机制设计

### 5.1 交互架构原则
- **主Agent作为中介**：Sub Agent不直接与用户交互，所有交互通过主Agent转发
- **上下文隔离保持**：交互过程不破坏sub agent的上下文隔离
- **决策点明确**：在关键决策节点设置人工确认机制

### 5.2 交互触发场景

| 场景 | 触发条件 | 交互类型 |
|------|----------|----------|
| 模型信息缺失 | 模型路径/类型未提供 | 必填信息询问 |
| 架构不识别 | 未知模型架构 | 选择确认/自定义配置 |
| 兼容性问题 | 存在不支持算子 | 风险确认/替代方案选择 |
| 量化策略选择 | 多种可行方案 | 方案选择 |
| 部署模式选择 | 性能目标不明确 | 需求澄清 |
| 配置确认 | 关键配置生成后 | 最终确认 |

### 5.3 交互流程设计

```
[Sub Agent]
    ↓ 发现需要用户输入
    ↓ 在报告中标记: NEED_USER_INPUT
    ↓ 附带问题详情（JSON格式）
[主Agent]
    ↓ 解析问题详情
    ↓ 调用AskUserQuestion工具
    ↓ 向用户展示问题
[用户]
    ↓ 提供回答/选择/反馈
[主Agent]
    ↓ 接收用户响应
    ↓ 将响应注入sub agent上下文
[Sub Agent]
    ↓ 继续执行
```

### 5.4 问题格式规范

Sub Agent输出的交互请求格式：
```json
{
  "interaction_required": true,
  "question_type": "choice|text|confirmation",
  "header": "量化策略",
  "question": "检测到多种可行量化方案，请选择：",
  "options": [
    {"label": "W8A8 INT8", "description": "推荐，平衡精度与性能"},
    {"label": "W4A8 INT8", "description": "更高压缩，轻微精度损失"},
    {"label": "BF16", "description": "无量化，最高精度"}
  ],
  "default_option": 0,
  "context": "模型: Qwen3-32B, 硬件: A3 8卡"
}
```

### 5.5 各Sub Agent交互点设计

#### model-analyzer 交互点
1. **模型信息确认**
   - 触发：检测到多个可能的模型类型
   - 问题：请确认模型类型（LLM/VLM/Embedding）
   
2. **架构识别确认**
   - 触发：架构与已知类型相似但不完全匹配
   - 问题：检测到类似[已知架构]，是否按此架构处理？

3. **兼容性风险确认**
   - 触发：存在NPU不支持的算子
   - 问题：以下算子在NPU上不支持，是否继续？（展示fallback方案）

#### npu-adapter 交互点
1. **量化策略选择**
   - 触发：多种量化方案可行
   - 问题：请选择量化策略（展示各方案优劣）

2. **部署模式确认**
   - 触发：硬件配置支持多种部署模式
   - 问题：请选择部署模式（PD Mixed/PD Separation）

3. **环境变量确认**
   - 触发：生成非标准配置
   - 问题：检测到特殊配置需求，是否应用推荐的环境变量？

#### perf-optimizer 交互点
1. **性能目标澄清**
   - 触发：未明确性能优先级
   - 问题：请选择性能优化目标（低延迟/高吞吐/平衡）

2. **资源分配确认**
   - 触发：资源需求接近硬件上限
   - 问题：推荐配置可能占用XX%内存，是否继续？

3. **基准测试确认**
   - 触发：准备执行耗时较长的基准测试
   - 问题：基准测试预计耗时XX分钟，是否执行？

### 5.6 反馈处理机制

#### 用户反馈类型
1. **选择型反馈**：从预设选项中选择
2. **文本型反馈**：自由输入文本
3. **确认型反馈**：是/否确认
4. **修正型反馈**：对生成结果的修改建议

#### 反馈处理流程
```
用户反馈 → 主Agent解析 → 验证反馈有效性 → 注入Sub Agent上下文
                                    ↓
                            无效反馈 → 重新提问
```

### 5.7 异常处理与人工干预

#### 异常类型
| 异常类型 | 处理方式 |
|----------|----------|
| 信息不足 | 暂停执行，请求用户补充 |
| 配置冲突 | 展示冲突详情，请求用户决策 |
| 执行失败 | 回滚操作，请求用户确认重试策略 |
| 超时 | 保存当前状态，请求用户是否继续 |

#### 人工干预入口
- **暂停点**：每个sub agent执行完毕后
- **检查点**：关键配置生成后
- **回退点**：执行失败时

### 5.8 交互记录与追溯

所有交互记录保存至：
```
.trae/interactions/
├── session_YYYYMMDD_HHMMSS/
│   ├── model-analyzer_interaction.json
│   ├── npu-adapter_interaction.json
│   └── perf-optimizer_interaction.json
```

记录内容包括：
- 时间戳
- 问题内容
- 用户回答
- 决策依据
- 执行结果

## 六、上下文隔离实现

### 6.1 Sub Agent隔离策略
- 每个sub agent独立上下文窗口
- 通过标准化JSON格式传递信息
- 不共享内部状态

### 6.2 信息传递机制
- 模型分析报告 → NPU适配代理
- NPU适配配置 → 性能优化代理
- 使用文件系统作为中间存储

### 6.3 协同工作流程
```
[model-analyzer] 
    ↓ 输出：模型分析报告
    ↓ (如需交互：等待用户反馈)
[npu-adapter]
    ↓ 输出：NPU适配配置
    ↓ (如需交互：等待用户反馈)
[perf-optimizer]
    ↓ 输出：优化配置与测试报告
    ↓ (如需交互：等待用户反馈)
```

## 七、文件结构

```
.trae/
├── subagents/
│   ├── model-analyzer.md
│   ├── npu-adapter.md
│   └── perf-optimizer.md
├── skills/
│   ├── model-config-parser/
│   │   └── SKILL.md
│   ├── architecture-detector/
│   │   └── SKILL.md
│   ├── compatibility-checker/
│   │   └── SKILL.md
│   ├── env-checker/
│   │   └── SKILL.md
│   ├── weight-validator/
│   │   └── SKILL.md
│   ├── npu-operator-mapper/
│   │   └── SKILL.md
│   ├── quantization-strategy/
│   │   └── SKILL.md
│   ├── npu-config-generator/
│   │   └── SKILL.md
│   ├── perf-benchmark/
│   │   └── SKILL.md
│   ├── resource-optimizer/
│   │   └── SKILL.md
│   ├── config-tuner/
│   │   └── SKILL.md
│   └── deployment-validator/
│       └── SKILL.md
├── knowledge/
│   ├── knowledge_index.json
│   ├── npu-specs.md
│   ├── sglang-api-reference.md
│   ├── model-adaptation-best-practices.md
│   └── troubleshooting.md
├── interactions/
│   └── session_YYYYMMDD_HHMMSS/
│       ├── model-analyzer_interaction.json
│       ├── npu-adapter_interaction.json
│       └── perf-optimizer_interaction.json
├── configs/
│   └── v{timestamp}_{model_hash}/
│       ├── model_analysis.json
│       ├── npu_adaptation.json
│       ├── perf_config.json
│       └── metadata.json
└── history/
    ├── models/
    │   └── {model_hash}/
    │       ├── adaptations/
    │       │   ├── {timestamp}_success.json
    │       │   └── {timestamp}_failed.json
    │       └── summary.json
    └── patterns/
        ├── common_issues.json
        └── successful_configs.json
```

## 八、实施步骤

### 步骤1：创建Sub Agent定义文件
- 创建`.trae/subagents/`目录
- 编写`model-analyzer.md`（含交互点定义、错误处理、元数据配置）
- 编写`npu-adapter.md`（含交互点定义、错误处理、元数据配置）
- 编写`perf-optimizer.md`（含交互点定义、错误处理、元数据配置）

### 步骤2：创建Skill定义文件
- 创建各skill目录
- 编写SKILL.md文件（含参数验证规则、错误处理、依赖声明）
- 确保描述精简且触发条件明确
- 新增skills：env-checker, weight-validator, deployment-validator

### 步骤3：创建公共知识文档
- 创建`.trae/knowledge/`目录
- 编写knowledge_index.json（知识文档索引）
- 编写NPU规格文档（A2/A3对比）
- 编写最佳实践文档
- 编写故障排查指南

### 步骤4：创建支撑目录结构
- 创建`.trae/interactions/`目录
- 创建`.trae/configs/`目录
- 创建`.trae/history/`目录
- 定义交互记录格式规范
- 定义配置版本管理规范
- 定义历史记录存储规范

### 步骤5：实现错误处理机制
- 定义错误码体系
- 实现超时处理逻辑
- 实现中断/取消机制
- 实现状态恢复机制
- 实现重试机制

### 步骤6：验证与测试
- 验证sub agent描述格式
- 验证skill描述格式
- 验证知识文档索引
- 测试交互流程
- 测试错误处理
- 测试触发条件
- 测试历史记录机制
- 端到端集成测试

## 九、补充设计要点

### 9.1 Sub Agent元数据配置

#### Model配置
| Sub Agent | Model配置 | 原因 |
|-----------|-----------|------|
| model-analyzer | `inherit` | 需要深度分析能力 |
| npu-adapter | `inherit` | 需要专业知识推理 |
| perf-optimizer | `fast` | 基准测试耗时长，使用快速模型提高效率 |

#### Readonly配置
| Sub Agent | Readonly | 原因 |
|-----------|----------|------|
| model-analyzer | `false` | 需要写入分析报告 |
| npu-adapter | `false` | 需要生成配置文件 |
| perf-optimizer | `false` | 需要写入测试报告 |

### 9.2 Skill依赖关系

```
model-config-parser (无依赖)
        ↓
architecture-detector (依赖: model-config-parser)
        ↓
compatibility-checker (依赖: architecture-detector)

npu-operator-mapper (依赖: compatibility-checker)
        ↓
quantization-strategy (依赖: npu-operator-mapper)
        ↓
npu-config-generator (依赖: quantization-strategy)

perf-benchmark (依赖: npu-config-generator)
        ↓
resource-optimizer (依赖: perf-benchmark)
        ↓
config-tuner (依赖: resource-optimizer)
```

### 9.3 Skill级别错误处理

| Skill | 可能错误 | 处理方式 |
|-------|----------|----------|
| model-config-parser | 配置文件不存在/格式错误 | 返回错误，请求用户提供正确路径 |
| architecture-detector | 未知架构类型 | 标记为unknown，请求用户确认 |
| compatibility-checker | 无法确定兼容性 | 返回警告，建议手动验证 |
| npu-operator-mapper | 算子无映射方案 | 提供fallback建议，请求用户确认 |
| quantization-strategy | 量化不支持 | 返回替代方案列表 |
| npu-config-generator | 配置冲突 | 展示冲突详情，请求用户选择 |
| perf-benchmark | 测试执行失败 | 记录错误日志，提供诊断建议 |
| resource-optimizer | 资源不足警告 | 返回警告，建议降低配置 |
| config-tuner | 无法优化 | 返回当前最优配置 |

### 9.4 知识文档动态加载机制

#### 加载触发规则
```
用户请求 → 主Agent分析需求 → 确定所需知识类型
    → 按需加载对应知识文档 → 注入Sub Agent上下文
```

#### 知识文档索引
```json
{
  "knowledge_index": {
    "npu-specs": {
      "trigger_keywords": ["A2", "A3", "Atlas", "NPU", "硬件"],
      "file": "npu-specs.md",
      "size_tokens": 1500
    },
    "sglang-api": {
      "trigger_keywords": ["API", "接口", "注册", "backend"],
      "file": "sglang-api-reference.md",
      "size_tokens": 1200
    },
    "best-practices": {
      "trigger_keywords": ["最佳实践", "规范", "流程"],
      "file": "model-adaptation-best-practices.md",
      "size_tokens": 1000
    },
    "troubleshooting": {
      "trigger_keywords": ["错误", "问题", "失败", "排查"],
      "file": "troubleshooting.md",
      "size_tokens": 800
    }
  }
}
```

### 9.5 环境依赖检查

#### 必需环境检查项
| 检查项 | 检查方式 | 失败处理 |
|--------|----------|----------|
| NPU驱动版本 | `npu-smi info` | 提示升级驱动 |
| CANN版本 | 环境变量检查 | 提示安装正确版本 |
| PyTorch NPU | `import torch_npu` | 提示安装torch_npu |
| Triton Ascend | `import triton_ascend` | 提示安装triton-ascend |
| SGLang NPU Kernel | `import sgl_kernel_npu` | 提示安装sgl-kernel-npu |
| 内存/显存 | 系统检查 | 警告资源不足 |

#### 环境检查Skill
新增 `env-checker` skill：
- 功能：检查NPU环境依赖
- 参数：hardware_type
- 返回：环境状态报告、缺失依赖列表

### 9.6 模型权重处理

#### 权重下载验证流程
```
1. 检查本地是否存在模型权重
2. 如不存在，检查HuggingFace访问权限
3. 下载模型权重（支持断点续传）
4. 验证权重完整性（SHA256校验）
5. 检查权重格式兼容性
```

#### 权重验证Skill
新增 `weight-validator` skill：
- 功能：验证模型权重完整性
- 参数：model_path
- 返回：验证结果、权重信息

### 9.7 部署后验证

#### 验证测试项
| 验证项 | 测试方法 | 通过标准 |
|--------|----------|----------|
| 服务启动 | 发送健康检查请求 | 返回200状态码 |
| 基本推理 | 发送简单prompt | 返回有效响应 |
| 精度验证 | 运行标准测试集 | 精度损失<1% |
| 性能验证 | 运行基准测试 | 达到预期TPOT/吞吐量 |
| 稳定性验证 | 持续运行测试 | 无崩溃/内存泄漏 |

#### 验证Skill
新增 `deployment-validator` skill：
- 功能：验证部署结果
- 参数：server_url, test_config
- 返回：验证报告

### 9.8 输出报告格式

#### 最终输出报告结构
```json
{
  "report_version": "1.0",
  "timestamp": "ISO8601",
  "model_info": {
    "name": "Qwen3-32B",
    "architecture": "Qwen",
    "parameters": "32B"
  },
  "hardware_info": {
    "type": "Atlas 800I A3",
    "cards": 8
  },
  "adaptation_summary": {
    "compatibility": "full|partial|unsupported",
    "quantization": "W8A8 INT8",
    "deploy_mode": "PD Mixed"
  },
  "performance_metrics": {
    "ttft_ms": 150,
    "tpot_ms": 20,
    "throughput_tokens_per_sec": 5000
  },
  "deployment_config": {
    "launch_command": "python -m sglang.launch_server ...",
    "env_variables": {...}
  },
  "validation_results": {
    "accuracy_test": "passed",
    "performance_test": "passed",
    "stability_test": "passed"
  },
  "recommendations": [
    "建议启用speculative decoding以进一步降低延迟",
    "建议增加DP size以提高吞吐量"
  ]
}
```

### 9.9 历史记录利用机制

#### 历史记录存储结构
```
.trae/history/
├── models/
│   └── {model_hash}/
│       ├── adaptations/
│       │   ├── {timestamp}_success.json
│       │   └── {timestamp}_failed.json
│       └── summary.json
└── patterns/
    ├── common_issues.json
    └── successful_configs.json
```

#### 历史记录利用策略
1. **相似模型匹配**：检测到相似架构时，推荐历史成功配置
2. **问题模式识别**：识别常见失败模式，提前预警
3. **配置模板复用**：基于历史成功案例生成配置模板

### 9.10 Sub Agent重试机制

#### 重试策略
| 场景 | 重试次数 | 重试间隔 | 降级策略 |
|------|----------|----------|----------|
| 网络超时 | 3 | 指数退避 | 使用缓存数据 |
| 配置解析失败 | 2 | 无 | 请求用户手动输入 |
| 基准测试失败 | 2 | 60秒 | 跳过测试，使用默认配置 |
| 资源不足 | 1 | 无 | 降低配置要求 |

#### 重试状态保存
```json
{
  "retry_state": {
    "agent": "perf-optimizer",
    "attempt": 2,
    "max_attempts": 3,
    "last_error": "benchmark_timeout",
    "saved_state": {...}
  }
}
```

### 9.11 错误处理与恢复机制

#### Sub Agent级别错误处理
| 错误类型 | 处理策略 | 恢复机制 |
|----------|----------|----------|
| 配置解析失败 | 返回错误详情，请求用户提供正确配置 | 重新解析 |
| 算子映射失败 | 标记不支持的算子，提供fallback方案 | 用户确认后继续 |
| 基准测试失败 | 记录失败原因，提供诊断建议 | 调整配置后重试 |
| 执行超时 | 保存当前进度，询问用户是否继续 | 从断点恢复 |

#### 执行超时处理
- 模型分析：默认超时300秒
- NPU适配：默认超时600秒
- 性能优化：默认超时1800秒（基准测试可能较长）
- 超时后：保存状态 → 询问用户 → 继续或终止

### 9.2 用户中断与取消机制

#### 中断类型
1. **暂停中断**：暂停当前执行，保存状态，可恢复
2. **取消中断**：终止当前执行，清理临时文件，不可恢复
3. **跳过中断**：跳过当前步骤，进入下一阶段

#### 中断处理流程
```
用户请求中断 → 主Agent捕获请求 → 通知当前Sub Agent
    → Sub Agent保存当前状态 → 清理资源 → 返回中断确认
    → 主Agent询问后续操作（恢复/取消/跳过）
```

### 9.3 上下文窗口管理

#### 窗口大小限制
- Sub Agent单次输出：不超过4000 tokens
- 交互问题：不超过500 tokens
- 知识文档按需加载：单次不超过2000 tokens

#### 信息传递序列化规范
```json
{
  "version": "1.0",
  "timestamp": "ISO8601格式",
  "source_agent": "model-analyzer",
  "target_agent": "npu-adapter",
  "data_type": "analysis_report|config|interaction",
  "payload": { /* 实际数据 */ },
  "checksum": "数据校验和"
}
```

### 9.4 配置版本管理

#### 版本控制策略
- 每次适配生成唯一版本号：`v{timestamp}_{model_hash}`
- 配置文件存储路径：`.trae/configs/{version}/`
- 支持配置回滚：保留最近10个版本

#### 配置文件结构
```
.trae/configs/
├── v20250311_abc123/
│   ├── model_analysis.json
│   ├── npu_adaptation.json
│   ├── perf_config.json
│   └── metadata.json  # 版本信息、创建时间、用户等
```

### 9.5 多模型并行适配支持

#### 并行策略
- 同类型模型：可并行分析，串行适配
- 不同类型模型：完全并行
- 共享资源模型：串行处理（避免资源冲突）

#### 并行协调机制
```
主Agent接收多个模型 → 分类排序 → 分配并行组
    → 并行执行Sub Agent → 汇总结果 → 串行适配
```

### 9.6 安全与隐私考虑

#### 敏感信息处理
- 模型路径：支持相对路径，避免暴露绝对路径
- API密钥：不记录到日志，使用环境变量
- 用户反馈：脱敏后存储

#### 访问控制
- 配置文件：仅当前用户可读写
- 交互记录：仅当前用户可访问
- 知识文档：只读

### 9.7 测试策略

#### Sub Agent测试
- 单元测试：每个Sub Agent的核心功能
- 集成测试：Sub Agent间的信息传递
- 端到端测试：完整适配流程

#### 测试用例覆盖
| 测试类型 | 覆盖场景 |
|----------|----------|
| 正常流程 | 已知模型适配 |
| 边界条件 | 未知架构、超大模型 |
| 异常处理 | 配置错误、网络超时 |
| 交互测试 | 各种用户输入组合 |

### 9.8 端到端工作流示例

#### 示例：Qwen3-32B在A3 8卡上的适配流程
```
1. 用户输入：模型ID=Qwen/Qwen3-32B, 硬件=A3 8卡
2. [model-analyzer] 执行
   - 解析配置 → 识别为Qwen架构
   - 检测MoE结构 → 非MoE
   - 评估兼容性 → 完全兼容
   - 输出分析报告
3. [npu-adapter] 执行
   - 检测到多种量化方案 → 触发交互
   - 用户选择W8A8 INT8
   - 生成适配配置
4. [perf-optimizer] 执行
   - 用户选择高吞吐模式
   - 生成TP=16, DP=4配置
   - 运行基准测试
5. 输出最终部署配置
```

## 十、文档精简原则

1. **描述字段**：不超过200字符，明确触发条件
2. **Sub Agent内容**：聚焦核心流程，避免冗余解释
3. **Skill内容**：参数说明简洁，示例精炼
4. **知识文档**：模块化组织，按需加载
5. **交互问题**：简洁明了，选项不超过4个

## 十一、优化记录

### 11.1 上下文窗口优化

| 优化项 | 优化前 | 优化后 | 节省 |
|--------|--------|--------|------|
| Sub Agent文档 | ~110行 | ~48行 | ~56% |
| Skill文档 | ~60行 | ~24行 | ~60% |
| 知识文档 | ~70行 | ~35行 | ~50% |
| 交互示例 | 完整JSON | 简化描述 | ~70% |

### 11.2 执行效率优化

1. **快速路径**：已知架构直接匹配模板，跳过详细分析
2. **历史复用**：相似模型复用历史成功配置
3. **明确Skill调用链**：每个Sub Agent明确列出调用的Skills
4. **输出路径标准化**：明确输出至`.trae/configs/{version}/`

### 11.3 信息完备性增强

1. **错误码标准化**：定义了E001-E203错误码体系
2. **next_agent字段**：输出中明确下一个要调用的Agent
3. **依赖声明**：每个Skill明确声明依赖关系
4. **最终报告路径**：输出中包含final_report路径
