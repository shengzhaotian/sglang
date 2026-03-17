# SGLang NPU模型适配技能套件 - 详细设计文档

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           主Skill (SKILL.md)                                 │
│                                                                             │
│  职责：                                                                      │
│  - 用户交互：接收任务、确认信息、展示结果（AskUserQuestion）                   │
│  - 流程编排：决定调用哪个Agent、控制执行顺序                                  │
│  - 初始代码生成：基于分析报告生成初始适配代码                                  │
│  - 运行测试：检测报错、决定是否调用Debug工程师                                 │
│  - 应用修复：将Debug工程师的修复代码应用到项目                                │
│  - 状态管理：跟踪适配进度、控制重试次数、处理异常                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │ 必须调用            │ 按需调用            │ 必须调用
         ↓                    ↓                    ↓
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Agent 1    │      │  Agent 2    │      │  Agent 3    │
│ 架构分析师   │      │ Debug工程师 │      │ 测试验证    │
│             │      │             │      │             │
│ 触发:       │      │ 触发:       │      │ 触发:       │
│ 用户请求适配 │      │ 运行报错时   │      │ 代码完成时  │
│             │      │             │      │             │
│ 输入:       │      │ 输入:       │      │ 输入:       │
│ - 模型路径   │      │ - 报错信息   │      │ - 适配代码  │
│ - 目标设备   │      │ - 上下文     │      │ - 测试配置  │
│             │      │             │      │             │
│ 输出:       │      │ 输出:       │      │ 输出:       │
│ - 分析报告   │      │ - 修复代码   │      │ - 测试报告  │
│             │      │ - 问题分析   │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ↓
                    文件系统传递上下文
         (analysis_report.md, error_context.md, test_report.md)
```

---

## 二、设计原则

### 2.1 核心原则

| 原则 | 说明 |
|------|------|
| **独立性** | 每个Agent是完整的、独立的任务单元，能够自主完成 |
| **最小化迭代** | 避免Agent间的来回迭代，减少上下文传递 |
| **结果导向** | Agent输出最终结果，主Skill只关心结果 |
| **用户交互前置** | 所有用户决策在调用Agent前完成 |
| **按需调用** | Agent 2只在出错时调用，每次处理独立报错 |

### 2.2 输入输出格式原则

- **优先使用Markdown文档**：便于人类阅读、LLM理解、版本控制
- **自然语言描述为主**：Subagent的输出以自然语言形式返回
- **文件系统传递**：大量上下文通过文件传递，而非内存

### 2.3 Subagent实例特性

**关键发现：每次Task调用都是全新的Subagent实例**

| 特性 | 说明 |
|------|------|
| 内存隔离 | Subagent之间不共享内存 |
| 无状态 | 每次调用都是全新实例，不保留之前的状态 |
| 文件共享 | 所有Subagent共享文件系统 |
| 无用户交互 | Subagent没有AskUserQuestion工具，无法与用户交互 |

---

## 三、三阶段Agent详细设计

### Agent 1: 模型架构分析师 (Model Architecture Analyst)

#### 职责
- 分析模型架构特征
- 计算内存占用和理论运行数值
- 生成并行配置建议
- 评估NPU兼容性
- 推荐参考模型

#### 触发条件
用户请求适配模型时，首先调用（必须调用）

#### 输入
```
模型路径：/path/to/model
目标设备：NPU (Ascend 910B1)
特殊需求：（可选）
```

#### 输出
- `analysis_report.md`：完整的分析报告

#### 特点
- 独立完成，不需要与其他Agent交互
- 可使用任何可用工具（读取配置、编写脚本、加载模型等）
- 输出是结构化的Markdown文档

#### 输出示例
```markdown
# 模型架构分析报告

## 一、模型基本信息
| 属性 | 值 |
|------|-----|
| 模型名称 | Qwen2-7B |
| 架构类型 | LLM (Decoder-only) |
| 模型家族 | Qwen |

## 二、架构详细分析
- 层数: 32
- 隐藏维度: 4096
- 注意力类型: GQA
- 注意力头数: 32
- KV头数: 8

## 三、NPU兼容性评估
✅ 兼容 - 该模型可在NPU上运行

## 四、内存与计算分析
| 配置 | KV Cache | 总内存(估算) |
|------|----------|--------------|
| TP=1, BS=4 | 8GB | ~22GB |
| TP=2, BS=16 | 16GB | ~46GB/卡 |

## 五、并行配置建议
推荐配置B：TP=2, BS=16, Seq=8K

## 六、参考模型推荐
- 主要参考: python/sglang/srt/models/qwen2.py (95%相似)

## 七、适配检查清单
- [ ] 创建模型文件
- [ ] 替换Attention为RadixAttention
- [ ] 配置NPU注意力后端
- [ ] 添加EntryClass注册
```

---

### Agent 2: Debug工程师 (Debug Engineer)

#### 职责
- 分析运行报错
- 定位问题原因
- 生成修复代码（结构化格式）
- 输出问题分析报告

#### 触发条件
主Skill发现运行报错时，按需调用

#### 输入
```
必需输入：
- error_context.md：报错上下文（错误信息、堆栈、相关代码片段）
- current_adapter_code.md：当前适配代码全文
- analysis_report.md：模型架构分析报告

可选输入：
- original_model_code.md：原始模型代码（从HuggingFace或参考模型）
- previous_fixes.md：之前的修复历史（避免重复修复）
```

#### 输出
- `debug_report.md`：问题分析和修复说明
- `fix_instructions.md`：结构化修复指令（主Skill严格执行）

#### 特点
- **按需调用**：只有出错时才调用
- **独立报错处理**：每次调用处理一个独立的报错
- **无状态**：不依赖之前的调用历史
- **无迭代**：一次调用完成一个报错的分析和修复
- **结构化输出**：修复指令采用结构化格式，便于主Skill解析和执行

#### 输入示例（error_context.md）
```markdown
# 报错上下文

## 错误信息
```
RuntimeError: RoPE scaling parameter 'rope_type' not recognized
```

## 错误堆栈
```
File "qwen2_custom.py", line 156, in forward
    rotary_emb = RotaryEmbedding(...)
```

## 相关代码
```python
# qwen2_custom.py, line 150-160
rope_scaling = config.rope_scaling
self.rotary_emb = RotaryEmbedding(
    dim=self.head_dim,
    rope_scaling=rope_scaling  # 问题可能在这里
)
```

## 运行环境
- NPU: Ascend 910B1
- SGLang版本: 0.4.0
```

#### 输出示例（debug_report.md）
```markdown
# Debug分析报告

## 问题定位
**根本原因**：RoPE Scaling参数格式与SGLang期望格式不匹配

**问题位置**：qwen2_custom.py, line 155

## 问题分析
HuggingFace的rope_scaling格式：
```json
{"rope_type": "linear", "factor": 2.0}
```

SGLang期望的格式：
```json
{"type": "linear", "factor": 2.0}
```

## 修复方案
需要转换rope_scaling参数格式。

## 验证建议
修复后重新运行推理测试，确认RoPE参数正确传递。
```

#### 输出示例（fix_instructions.md）
```markdown
# 修复指令

## 修复类型
REPLACE_BLOCK

## 目标文件
python/sglang/srt/models/qwen2_custom.py

## 修复操作

### 操作1：替换代码块
**位置**：第155-160行
**原代码**：
```python
rope_scaling = config.rope_scaling
self.rotary_emb = RotaryEmbedding(
    dim=self.head_dim,
    rope_scaling=rope_scaling
)
```

**新代码**：
```python
rope_scaling = config.rope_scaling
if rope_scaling is not None and "rope_type" in rope_scaling:
    rope_scaling = {
        "type": rope_scaling["rope_type"],
        "factor": rope_scaling.get("factor", 1.0),
    }
self.rotary_emb = RotaryEmbedding(
    dim=self.head_dim,
    rope_scaling=rope_scaling
)
```

## 修复说明
- 在使用rope_scaling之前添加格式转换逻辑
- 保持向后兼容性，如果格式正确则不做修改

## 风险评估
- 风险等级：低
- 影响范围：仅影响RoPE初始化
- 回滚方案：删除添加的格式转换代码
```

---

### Agent 2 输出格式规范

#### fix_instructions.md 结构化格式

主Skill必须严格按照以下格式解析和执行修复指令：

```markdown
# 修复指令

## 修复类型
[REPLACE_BLOCK | INSERT_BEFORE | INSERT_AFTER | DELETE_BLOCK | MODIFY_CONFIG]

## 目标文件
[文件路径]

## 修复操作

### 操作N：[操作描述]
**位置**：[行号范围或代码定位标记]
**原代码**：（仅REPLACE_BLOCK和DELETE_BLOCK需要）
```python
[原始代码块]
```

**新代码**：（仅REPLACE_BLOCK、INSERT_BEFORE、INSERT_AFTER需要）
```python
[新代码块]
```

## 修复说明
[修复的详细说明]

## 风险评估
- 风险等级：[低 | 中 | 高]
- 影响范围：[描述]
- 回滚方案：[描述]
```

#### 修复类型说明

| 类型 | 说明 | 主Skill执行方式 |
|------|------|-----------------|
| REPLACE_BLOCK | 替换代码块 | 使用SearchReplace工具，old_str=原代码，new_str=新代码 |
| INSERT_BEFORE | 在指定位置前插入 | 读取目标位置代码，将新代码插入到前面 |
| INSERT_AFTER | 在指定位置后插入 | 读取目标位置代码，将新代码插入到后面 |
| DELETE_BLOCK | 删除代码块 | 使用SearchReplace工具，old_str=原代码，new_str="" |
| MODIFY_CONFIG | 修改配置文件 | 更新YAML/JSON配置文件中的指定字段 |

#### 主Skill执行修复的流程

```markdown
1. 读取 fix_instructions.md
2. 解析修复类型和目标文件
3. 对于每个修复操作：
   a. 验证目标文件存在
   b. 验证原代码存在（对于REPLACE_BLOCK和DELETE_BLOCK）
   c. 执行修复
   d. 记录修复日志
4. 更新 adapter_state.yaml 中的修复历史
5. 重新运行测试
```

---

### Agent 3: 测试验证工程师 (Test & Validation Engineer)

#### 职责
- 运行基础推理测试
- 验证输出正确性（与HF对比）
- 执行性能基准测试
- 生成测试报告

#### 触发条件
代码适配完成或修复完成后调用（必须调用）

#### 输入
```
适配代码路径：python/sglang/srt/models/qwen2_custom.py
测试配置：
  - 模型路径：/path/to/model
  - 测试模式：quick
  - 是否对比HF：是
```

#### 输出
- `test_report.md`：测试报告（通过/失败+原因）

#### 特点
- 独立完成测试验证
- 输出明确的通过/失败结果
- 不修改代码，只报告结果

#### 输出示例（成功）
```markdown
# 测试报告

## 测试结果
✅ **通过**

## 详细结果
| 测试项 | 状态 | 详情 |
|--------|------|------|
| 基础推理 | ✅ 通过 | 模型加载成功，推理正常 |
| 输出正确性 | ✅ 通过 | Logits差异 < 0.001 |
| NPU功能 | ✅ 通过 | ascend后端正常工作 |
| 性能测试 | ✅ 通过 | 吞吐量: 125.6 tok/s |

## 性能数据
- TTFT: 45.2ms
- 吞吐量: 125.6 tok/s
- 内存占用: 18.5GB/卡
```

#### 输出示例（失败）
```markdown
# 测试报告

## 测试结果
❌ **失败**

## 失败原因
输出正确性验证失败：Logits差异过大

## 详细数据
- Max Logits Diff: 2.5 (预期 < 0.01)
- Mean Logits Diff: 0.8
- 受影响层: layer_15, layer_16

## 可能原因
1. RoPE Scaling参数传递不正确
2. 位置编码计算存在精度问题

## 建议
调用Debug工程师分析此问题。
```

---

## 四、完整执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         完整执行流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 主Skill与用户交互                                            │
│     AskUserQuestion: 确认模型路径、目标设备等                    │
│                           ↓                                     │
│  2. 调用Agent 1（架构分析师）                                     │
│     Task(query="分析模型...")                                   │
│                           ↓                                     │
│  3. Agent 1输出                                                  │
│     → analysis_report.md                                        │
│                           ↓                                     │
│  4. 主Skill展示报告，与用户交互                                  │
│     AskUserQuestion: 分析结果是否满意？是否继续？                │
│                           ↓                                     │
│  5. 主Skill根据分析报告生成初始适配代码                          │
│     （主Skill直接执行，不调用Agent）                             │
│                           ↓                                     │
│  6. 主Skill尝试运行                                              │
│                           ↓                                     │
│     ┌─────────────────────────────────┐                         │
│     │ 运行成功？                       │                         │
│     └─────────────────────────────────┘                         │
│          │                    │                                 │
│         成功                 失败                                │
│          │                    │                                 │
│          │                    ↓                                 │
│          │         7. 主Skill记录报错                            │
│          │            → error_context.md                        │
│          │                    │                                 │
│          │                    ↓                                 │
│          │         8. 调用Agent 2（Debug工程师）                 │
│          │            Task(query="                               │
│          │              报错上下文：error_context.md             │
│          │              分析报告：analysis_report.md             │
│          │            ")                                         │
│          │                    │                                 │
│          │                    ↓                                 │
│          │         9. Agent 2输出                                │
│          │            → fix_code.py + debug_report.md           │
│          │                    │                                 │
│          │                    ↓                                 │
│          │         10. 主Skill应用修复，重新运行                 │
│          │                    │                                 │
│          │         ←──────────┘                                  │
│          │         （循环，最多3次）                              │
│          │                    │                                 │
│          ↓                    ↓                                 │
│  11. 调用Agent 3（测试验证工程师）                               │
│      Task(query="                                               │
│        适配代码：xxx.py                                         │
│        测试配置：xxx                                            │
│      ")                                                         │
│                           ↓                                     │
│  12. Agent 3输出                                                 │
│      → test_report.md                                           │
│                           ↓                                     │
│     ┌─────────────────────────────────┐                         │
│     │ 测试通过？                       │                         │
│     └─────────────────────────────────┘                         │
│          │                    │                                 │
│         通过                 失败                                │
│          │                    │                                 │
│          ↓                    ↓                                 │
│       完成           询问用户如何处理                            │
│                          │                                      │
│                          ├─→ 自动修复 ─→ 回到步骤7              │
│                          ├─→ 手动修改 ─→ 等待用户操作           │
│                          └─→ 放弃 ─→ 结束                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、主Skill职责详解

### 5.1 主Skill与Agent的职责划分

| 职责 | 主Skill | Agent 1 | Agent 2 | Agent 3 |
|------|---------|---------|---------|---------|
| 用户交互 | ✅ | ❌ | ❌ | ❌ |
| 流程编排 | ✅ | ❌ | ❌ | ❌ |
| 模型分析 | ❌ | ✅ | ❌ | ❌ |
| 初始代码生成 | ✅ | ❌ | ❌ | ❌ |
| 报错分析 | ❌ | ❌ | ✅ | ❌ |
| 应用修复 | ✅ | ❌ | ❌ | ❌ |
| 测试验证 | ❌ | ❌ | ❌ | ✅ |
| 状态管理 | ✅ | ❌ | ❌ | ❌ |

### 5.2 主Skill核心逻辑

```markdown
# 主Skill伪代码

## 初始化
workspace = ".trae/workspace/adapter/"
max_debug_iterations = 3

## 步骤1：用户交互
model_info = AskUserQuestion("请提供模型信息")

## 步骤2：调用Agent 1分析
analysis_result = Task(
  subagent_type="general_purpose_task",
  query="分析模型：{model_info}，输出到 {workspace}/analysis_report.md"
)

## 步骤3：展示报告，用户确认
user_choice = AskUserQuestion(
  "分析报告已生成，是否继续？",
  options=["继续", "调整配置", "取消"]
)

if user_choice == "取消":
  return

## 步骤4：生成初始适配代码
initial_code = generate_initial_code(analysis_result)

## 步骤5：运行测试
iteration = 0
while iteration < max_debug_iterations:
  run_result = run_model(initial_code)
  
  if run_result.success:
    break
  
  ## 步骤6：记录报错
  write_file(f"{workspace}/error_context.md", run_result.error)
  
  ## 步骤7：调用Agent 2
  debug_result = Task(
    subagent_type="general_purpose_task",
    query="分析报错：{workspace}/error_context.md，输出修复代码"
  )
  
  ## 步骤8：应用修复
  initial_code = apply_fix(initial_code, debug_result.fix_code)
  iteration += 1

## 步骤9：调用Agent 3验证
test_result = Task(
  subagent_type="general_purpose_task",
  query="验证适配代码：{initial_code}，输出测试报告"
)

## 步骤10：处理结果
if test_result.passed:
  show("适配成功！")
else:
  user_choice = AskUserQuestion(
    "测试失败，如何处理？",
    options=["自动修复", "手动修改", "放弃"]
  )
```

---

## 六、数据传递协议

### 6.1 文件系统布局

```
.trae/workspace/adapter/
├── analysis_report.md          # Agent 1 输出 - 分析报告
├── error_context.md            # 主Skill生成 - 报错上下文
├── current_adapter_code.md     # 主Skill生成 - 当前适配代码全文
├── original_model_code.md      # 主Skill生成 - 原始模型代码（可选）
├── debug_report.md             # Agent 2 输出 - Debug分析报告
├── fix_instructions.md         # Agent 2 输出 - 结构化修复指令
├── test_report.md              # Agent 3 输出 - 测试报告
├── adapter_state.yaml          # 主Skill维护 - 状态文件
├── fix_history.md              # 主Skill维护 - 修复历史记录
└── logs/                       # 执行日志
    └── execution.log
```

### 6.2 状态管理

```yaml
# adapter_state.yaml

current_phase: "debugging"  # analyzing, coding, debugging, testing, completed
model_info:
  path: "/path/to/model"
  name: "Qwen2-7B"
  target_device: "NPU"
  architecture: "Qwen2ForCausalLM"

iteration:
  debug_count: 1
  max_debug_iterations: 10
  consecutive_failures: 0  # 连续失败次数（相同问题）

files:
  analysis_report: "analysis_report.md"
  adapter_code: "python/sglang/srt/models/qwen2_custom.py"

fixes:
  - timestamp: "2026-03-16T12:00:00Z"
    error_type: "rope_scaling_format"
    fix_type: "REPLACE_BLOCK"
    status: "applied"
    result: "success"
  - timestamp: "2026-03-16T12:05:00Z"
    error_type: "weight_name_mismatch"
    fix_type: "REPLACE_BLOCK"
    status: "applied"
    result: "failed"

errors:
  - timestamp: "2026-03-16T12:00:00Z"
    phase: "running"
    message: "RoPE scaling parameter error"
    resolved: true
```

### 6.3 Agent间数据流

```
主Skill ──调用──► Agent 1
   │                  │
   │                  ↓
   │          analysis_report.md
   │                  │
   │◄─────返回───────┘
   │
   │（主Skill生成初始代码）
   │
   │（运行失败）
   │
   ├──生成──► error_context.md
   ├──生成──► current_adapter_code.md
   ├──生成──► original_model_code.md（可选）
   │
   ├──调用──► Agent 2
   │              │
   │              ↓
   │      fix_instructions.md + debug_report.md
   │              │
   │◄───返回──────┘
   │
   │（主Skill解析fix_instructions.md，应用修复）
   │
   │（更新fix_history.md）
   │
   │（重新运行）
   │
   │（运行成功）
   │
   ├──调用──► Agent 3
   │              │
   │              ↓
   │      test_report.md
   │              │
   │◄───返回──────┘
   │
   ↓
 完成
```

---

## 七、技能文件结构

```
.trae/skills/sglang-npu-adapter/
├── SKILL.md                          # 主Skill入口
├── prompts/
│   ├── model_analyzer.md             # Agent 1 提示模板
│   ├── debug_engineer.md             # Agent 2 提示模板
│   └── test_validator.md             # Agent 3 提示模板
│
├── references/                       # 知识库（按Agent职责分离，按需加载）
    │
    ├── agent1_analyst/               # Agent 1: 架构分析师专用
    │   ├── llm_architecture.md       # LLM架构识别知识
    │   ├── vlm_architecture.md       # VLM架构识别知识
    │   ├── moe_architecture.md       # MoE架构识别知识
    │   ├── mla_architecture.md       # MLA架构识别知识
    │   ├── memory_calculation.md     # 内存计算方法
    │   ├── npu_specifications.md     # NPU规格限制
    │   └── sglang_model_registry.md  # SGLang模型注册机制
    │
    ├── agent2_debug/                 # Agent 2: Debug工程师专用
    │   ├── common_errors.md          # 常见错误及解决方案
    │   ├── attention_debug.md        # Attention相关问题
    │   ├── rope_debug.md             # RoPE相关问题
    │   ├── moe_debug.md              # MoE相关问题
    │   └── npu_specific_issues.md    # NPU特定问题
    │
    ├── agent3_validator/             # Agent 3: 测试验证工程师专用
    │   ├── basic_inference_test.md   # 基础推理测试方法
    │   ├── correctness_validation.md # 正确性验证方法
    │   ├── performance_benchmark.md  # 性能基准测试
    │   └── npu_validation.md         # NPU特定验证
    │
    └── shared/                       # 共享知识（主Skill和所有Agent可用）
        ├── sglang_basics.md          # SGLang基础概念
        ├── npu_basics.md             # NPU基础概念
        └── code_templates_guide.md   # 代码模板使用指南
```

---

## 七-A、知识库内容详细规划

### 知识库优先级排序

| 优先级 | 文件 | 所属Agent | 说明 |
|--------|------|-----------|------|
| P0 | sglang_model_registry.md | Agent 1 | 理解模型注册机制是适配的基础 |
| P0 | llm_architecture.md | Agent 1 | LLM是最常见的模型类型 |
| P0 | common_errors.md | Agent 2 | 常见错误是Debug的核心知识 |
| P0 | sglang_basics.md | Shared | 所有Agent都需要的基础知识 |
| P1 | moe_architecture.md | Agent 1 | MoE模型适配复杂度高 |
| P1 | attention_debug.md | Agent 2 | Attention问题最常见 |
| P1 | rope_debug.md | Agent 2 | RoPE问题频发 |
| P1 | basic_inference_test.md | Agent 3 | 测试验证的基础 |
| P2 | vlm_architecture.md | Agent 1 | VLM适配相对复杂 |
| P2 | mla_architecture.md | Agent 1 | MLA是DeepSeek特有架构 |
| P2 | memory_calculation.md | Agent 1 | 内存计算用于配置建议 |
| P2 | npu_specifications.md | Agent 1 | NPU规格限制 |
| P2 | moe_debug.md | Agent 2 | MoE特定问题 |
| P2 | npu_specific_issues.md | Agent 2 | NPU特定问题 |
| P2 | correctness_validation.md | Agent 3 | 正确性验证 |
| P2 | performance_benchmark.md | Agent 3 | 性能测试 |
| P2 | npu_validation.md | Agent 3 | NPU验证 |
| P2 | npu_basics.md | Shared | NPU基础知识 |
| P2 | code_templates_guide.md | Shared | 代码模板指南 |

---

### Agent 1 知识库详细大纲

#### 1. sglang_model_registry.md (P0)
```markdown
# SGLang模型注册机制

## 1. 注册流程概述
- EntryClass的作用和定义方式
- ModelRegistry的工作原理
- 模型发现机制（import_model_classes函数）

## 2. EntryClass定义规范
- 单模型注册：EntryClass = ModelClass
- 多模型注册：EntryClass = [ModelClass1, ModelClass2]
- 命名约定：类名即为架构名

## 3. 模型文件结构
- 必要组件：Config类、Model类、EntryClass
- 可选组件：MLP、Attention、DecoderLayer
- 权重加载：load_weights方法

## 4. 注册示例
- 简单LLM：qwen2.py
- MoE模型：qwen2_moe.py
- VLM模型：qwen2_vl.py

## 5. 常见问题
- 架构名冲突处理
- 自定义模型路径配置
```

#### 2. llm_architecture.md (P0)
```markdown
# LLM架构识别知识

## 1. 核心架构组件
- Embedding层：VocabParallelEmbedding
- Decoder Layer：Attention + MLP + Norm
- LM Head：ParallelLMHead

## 2. Attention变体识别
- MHA (Multi-Head Attention)：num_heads = num_kv_heads
- MQA (Multi-Query Attention)：num_kv_heads = 1
- GQA (Grouped-Query Attention)：num_kv_heads < num_heads

## 3. MLP变体识别
- 标准MLP：gate_proj + up_proj + down_proj
- gated-MLP：使用SiluAndMul激活
- 非gated-MLP：使用其他激活函数

## 4. Norm变体识别
- RMSNorm：最常见
- LayerNorm：较少见

## 5. RoPE变体识别
- 标准RoPE：rope_scaling = None
- Linear RoPE Scaling：rope_scaling = {"type": "linear", "factor": N}
- Dynamic NTK：rope_scaling = {"type": "dynamic", ...}
- Yarn：rope_scaling = {"type": "yarn", ...}

## 6. 配置文件关键字段
- hidden_size, intermediate_size, num_hidden_layers
- num_attention_heads, num_key_value_heads
- rope_theta, rope_scaling, max_position_embeddings
- rms_norm_eps, vocab_size

## 7. HuggingFace到SGLang的映射
- config字段映射关系
- 权重名称映射关系
```

#### 3. moe_architecture.md (P1)
```markdown
# MoE架构识别知识

## 1. MoE核心概念
- 专家(Expert)：独立的MLP网络
- 路由器(Router/Gate)：决定token分配
- Top-K选择：选择激活的专家数量

## 2. MoE变体识别
- 稀疏MoE：每层只有部分专家被激活
- 共享专家：shared_expert + routed experts
- 纯MoE：所有层都是MoE
- 混合MoE：部分层是MoE，部分是Dense

## 3. SGLang MoE实现
- SparseMoeBlock类结构
- FusedMoE实现
- TopK路由实现
- DeepEP后端支持

## 4. 配置文件关键字段
- num_experts：专家总数
- num_experts_per_tok：每个token激活的专家数
- moe_intermediate_size：专家的intermediate_size
- shared_expert_intermediate_size：共享专家的intermediate_size

## 5. 权重加载特殊处理
- expert_params_mapping
- 专家权重的sharding策略
```

#### 4. vlm_architecture.md (P2)
```markdown
# VLM架构识别知识

## 1. VLM核心组件
- Vision Tower：视觉编码器
- Projector：视觉特征投影
- Language Model：语言模型

## 2. VLM架构变体
- 独立Vision Tower：如LLaVA
- 集成Vision Tower：如Qwen2-VL
- 多模态输入：图像+文本+视频

## 3. SGLang VLM实现
- vision_tower模块
- projector模块
- 多模态输入处理

## 4. 配置文件关键字段
- vision_config
- projector_hidden_act
- image_token_index
```

#### 5. mla_architecture.md (P2)
```markdown
# MLA架构识别知识（DeepSeek特有）

## 1. MLA核心概念
- Multi-Head Latent Attention
- KV压缩：将KV压缩到低维空间
- 解压缩：在attention时解压缩

## 2. MLA vs 标准Attention
- KV Cache大小对比
- 计算流程差异
- 内存带宽优势

## 3. SGLang MLA实现
- MLA Attention类
- 压缩/解压缩层
- RoPE处理

## 4. 配置文件关键字段
- kv_lora_rank：KV压缩维度
- q_lora_rank：Query压缩维度
- qk_rope_head_dim：RoPE维度
```

#### 6. memory_calculation.md (P2)
```markdown
# 内存计算方法

## 1. 模型权重内存
- 参数量计算：embedding + layers + lm_head
- 精度影响：FP16/BF16/FP32
- TP分片：每卡的参数量

## 2. KV Cache内存
- KV Cache大小公式
- 序列长度影响
- Batch Size影响

## 3. 激活内存
- 激活重计算策略
- 激活内存估算

## 4. 总内存估算公式
- 总内存 = 权重 + KV Cache + 激活 + 开销

## 5. 并行配置建议
- TP选择依据
- BS选择依据
- 序列长度限制
```

#### 7. npu_specifications.md (P2)
```markdown
# NPU规格限制

## 1. Ascend NPU规格
- 910B1规格：内存、算力
- 910B2规格：内存、算力
- 910B3规格：内存、算力

## 2. 内存限制
- HBM大小
- 可用内存估算

## 3. 算子支持限制
- 支持的Attention后端
- 支持的RoPE实现
- 支持的MoE实现

## 4. 性能特性
- 内存带宽
- 计算吞吐
- 通信带宽
```

---

### Agent 2 知识库详细大纲

#### 1. common_errors.md (P0)
```markdown
# 常见错误及解决方案

## 1. 配置相关错误
### 1.1 rope_scaling参数格式不匹配
- 错误现象：RuntimeError: RoPE scaling parameter 'rope_type' not recognized
- 原因：HF格式与SGLang格式不一致
- 解决：转换rope_scaling格式

### 1.2 配置字段缺失
- 错误现象：AttributeError: 'Config' object has no attribute 'xxx'
- 原因：新模型使用了新配置字段
- 解决：在模型文件中添加默认值

## 2. 权重加载错误
### 2.1 权重名称不匹配
- 错误现象：Parameter xxx not found in params_dict
- 原因：HF权重名称与SGLang期望不一致
- 解决：在load_weights中添加映射

### 2.2 权重形状不匹配
- 错误现象：size mismatch for xxx
- 原因：TP配置或模型配置不正确
- 解决：检查TP配置和权重sharding

## 3. 运行时错误
### 3.1 CUDA/NPU OOM
- 错误现象：OutOfMemoryError
- 原因：内存不足
- 解决：减小BS、增大TP、启用量化

### 3.2 算子不支持
- 错误现象：NotImplementedError: No operator found for xxx
- 原因：NPU不支持某些算子
- 解决：使用替代实现或CPU fallback
```

#### 2. attention_debug.md (P1)
```markdown
# Attention相关问题

## 1. Attention后端选择
- FlashAttention：GPU默认
- Triton：GPU备选
- Ascend：NPU专用
- 如何选择正确的后端

## 2. GQA相关问题
- num_heads与num_kv_heads的关系
- TP分片后的head分配
- 常见配置错误

## 3. Sliding Window Attention
- 配置方法
- 与RadixAttention的兼容性

## 4. ALiBi位置编码
- 与RoPE的区别
- 实现注意事项

## 5. 常见错误案例
- 案例1：GQA配置错误导致输出错误
- 案例2：Attention后端不兼容
- 案例3：KV Cache大小不匹配
```

#### 3. rope_debug.md (P1)
```markdown
# RoPE相关问题

## 1. RoPE Scaling类型
- linear：线性插值
- dynamic：动态NTK
- yarn：Yarn方法
- longrope：LongRoPE方法

## 2. 格式转换
- HF格式：{"rope_type": "linear", "factor": 2.0}
- SGLang格式：{"type": "linear", "factor": 2.0}
- 转换代码示例

## 3. 常见错误
- 错误1：rope_type vs type
- 错误2：factor值不正确
- 错误3：max_position_embeddings不匹配

## 4. 特殊RoPE变体
- M-RoPE（多模态）
- 3D-RoPE（视频）
```

#### 4. moe_debug.md (P2)
```markdown
# MoE相关问题

## 1. 专家路由问题
- Top-K选择错误
- 路由权重归一化
- 专家负载不均衡

## 2. 专家并行问题
- EP配置错误
- 专家权重sharding
- 通信问题

## 3. 共享专家问题
- shared_expert配置
- shared_expert_gate配置

## 4. 常见错误案例
- 案例1：num_experts配置错误
- 案例2：专家权重加载失败
- 案例3：DeepEP后端不兼容
```

#### 5. npu_specific_issues.md (P2)
```markdown
# NPU特定问题

## 1. 算子兼容性
- 不支持的算子列表
- 替代实现方案
- CPU fallback策略

## 2. 内存管理
- NPU内存模型
- 内存碎片问题
- 内存优化技巧

## 3. 性能调优
- Attention后端选择
- MoE实现选择
- 通信优化

## 4. 常见错误案例
- 案例1：Ascend后端初始化失败
- 案例2：算子不支持
- 案例3：内存不足
```

---

### Agent 3 知识库详细大纲

#### 1. basic_inference_test.md (P1)
```markdown
# 基础推理测试方法

## 1. 测试流程
- 模型加载测试
- 基础推理测试
- 输出验证

## 2. 测试命令
- SGLang启动命令
- 测试请求命令
- 常用参数

## 3. 测试用例
- 短文本推理
- 长文本推理
- 批量推理

## 4. 错误诊断
- 启动失败诊断
- 推理失败诊断
- 输出异常诊断
```

#### 2. correctness_validation.md (P2)
```markdown
# 正确性验证方法

## 1. 与HuggingFace对比
- 输出logits对比
- 采样输出对比
- 允许误差范围

## 2. 验证流程
- 加载HF模型
- 加载SGLang模型
- 相同输入对比输出

## 3. 常见问题
- 精度差异来源
- 数值稳定性问题
- 验证失败分析
```

#### 3. performance_benchmark.md (P2)
```markdown
# 性能基准测试

## 1. 性能指标
- TTFT (Time To First Token)
- 吞吐量 (tokens/s)
- 延迟 (latency)

## 2. 测试工具
- SGLang benchmark工具
- 自定义测试脚本

## 3. 测试配置
- Batch Size范围
- 序列长度范围
- 并行配置

## 4. 结果分析
- 性能瓶颈识别
- 优化建议
```

#### 4. npu_validation.md (P2)
```markdown
# NPU特定验证

## 1. NPU功能验证
- Ascend后端验证
- 算子正确性验证
- 内存使用验证

## 2. NPU性能验证
- 与GPU性能对比
- 内存效率对比
- 通信效率验证

## 3. NPU稳定性验证
- 长时间运行测试
- 大批量测试
- 边界条件测试
```

---

### Shared 知识库详细大纲

#### 1. sglang_basics.md (P0)
```markdown
# SGLang基础概念

## 1. SGLang架构概述
- 推理引擎架构
- 服务化架构
- 核心组件

## 2. 模型适配核心概念
- RadixAttention：SGLang的核心Attention实现
- 权重加载机制
- 模型注册机制

## 3. 关键导入
- from sglang.srt.layers.radix_attention import RadixAttention
- from sglang.srt.layers.rotary_embedding import get_rope
- from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
- from sglang.srt.layers.layernorm import RMSNorm
- from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead

## 4. 模型文件结构模板
- 必要的类定义
- 必要的方法
- EntryClass定义

## 5. 常用工具函数
- add_prefix：添加参数前缀
- make_layers：创建层列表
- get_layer_id：获取层ID
```

#### 2. npu_basics.md (P2)
```markdown
# NPU基础概念

## 1. NPU vs GPU
- 架构差异
- 算子支持差异
- 内存模型差异

## 2. Ascend NPU生态
- CANN软件栈
- PyTorch扩展
- SGLang集成

## 3. NPU特定配置
- 环境变量
- 后端选择
- 内存配置

## 4. 常见限制
- 不支持的操作
- 性能差异
- 兼容性问题
```

#### 3. code_templates_guide.md (P2)
```markdown
# 代码模板使用指南

## 1. 模板概述
- LLM模板
- MoE模板
- VLM模板

## 2. 模板变量
- 模型名称变量
- 配置字段变量
- 组件变量

## 3. 模板使用方法
- 复制模板
- 替换变量
- 添加自定义逻辑

## 4. 模板扩展
- 添加新组件
- 修改现有组件
- 自定义权重加载
```

---

## 七-B、知识库设计原则

| 原则 | 说明 |
|------|------|
| **职责分离** | 按Agent职责划分知识目录，避免交叉 |
| **按需加载** | 每个Agent只加载自己目录下的知识 |
| **最小化读取** | 知识文件粒度适中，避免读取不需要的内容 |
| **独立可读** | 每个知识文件可独立阅读，不依赖其他文件 |

### 按需加载规则

| 调用方 | 加载的知识库 | 说明 |
|--------|-------------|------|
| Agent 1 | `agent1_analyst/` + `shared/` | 只加载分析相关知识 |
| Agent 2 | `agent2_debug/` + `shared/` | 只加载Debug相关知识 |
| Agent 3 | `agent3_validator/` + `shared/` | 只加载验证相关知识 |
| 主Skill | `shared/` | 只加载共享知识 |

### 知识文件规范

```
每个知识文件应遵循：

1. 单一主题：一个文件只讲一个主题
2. 适中长度：建议200-500行，避免过长
3. 结构清晰：使用标题、表格、代码块
4. 可独立阅读：不依赖其他知识文件

示例结构：
# 主题名称

## 概述
简要说明本知识文件的内容

## 核心概念
关键概念解释

## 实践指南
具体操作步骤

## 常见问题
FAQ

## 参考
相关代码路径、文档链接
```

---

## 八、关键验证结果

### 8.1 Subagent工具能力

| 工具 | 主Skill | Subagent | 说明 |
|------|---------|----------|------|
| Skill | ✅ 可用 | ❌ 不可用 | Subagent无法调用其他Skill |
| AskUserQuestion | ✅ 可用 | ❌ 不可用 | Subagent无法与用户交互 |
| Task | ✅ 可用 | ❌ 不可用 | Subagent无法调用其他Subagent |
| Read/Write | ✅ 可用 | ✅ 可用 | 文件操作正常 |
| RunCommand | ✅ 可用 | ✅ 可用 | 命令执行正常 |

### 8.2 Subagent实例特性

| 特性 | 结果 |
|------|------|
| 每次Task调用都是全新实例 | ✅ 验证通过 |
| Subagent之间不共享内存 | ✅ 验证通过 |
| Subagent共享文件系统 | ✅ 验证通过 |
| Subagent无法暂停等待用户 | ✅ 验证通过 |

### 8.3 设计影响

基于以上验证结果，设计必须遵循：

1. **所有用户交互在主Skill完成**
2. **上下文通过文件系统传递**
3. **每个Agent独立完成，不依赖之前的调用历史**
4. **Agent 2按需调用，每次处理独立报错**
5. **避免Agent间的来回迭代**

---

## 九、错误处理机制

| 错误场景 | 处理方式 |
|----------|----------|
| Agent 1分析失败 | 报告错误，询问用户是否重试或手动提供信息 |
| 初始代码运行失败 | 调用Agent 2分析报错，应用修复 |
| Debug修复后仍失败 | 最多迭代10次，超过则询问用户 |
| Agent 3测试失败 | 询问用户：自动修复/手动修改/放弃 |
| 环境不满足要求 | 在开始前检查环境，不满足则提示用户 |
| 修复代码应用失败 | 记录失败原因，回滚并询问用户 |

### Debug迭代策略

```markdown
Debug迭代采用渐进式策略：

1. 第1-3次迭代：处理常见错误（配置格式、权重映射等）
2. 第4-6次迭代：处理中等复杂度错误（算子兼容、内存问题等）
3. 第7-10次迭代：处理复杂错误（架构不匹配、深层问题等）

每次迭代后：
- 更新 adapter_state.yaml 中的迭代计数和修复历史
- 记录已尝试的修复方案，避免重复
- 如果连续3次修复相同问题失败，提示用户考虑手动干预
```

---

## 十、与原设计的对比

| 方面 | 原设计 | 最终设计 |
|------|--------|----------|
| Agent数量 | 3个串行 | 3个（Agent 2按需调用） |
| Agent 2角色 | 兼容分析与适配编码 | Debug工程师 |
| Agent 2触发 | 必须调用 | 按需调用（仅出错时） |
| 初始代码生成 | Agent 2负责 | 主Skill负责 |
| Agent间迭代 | Agent2↔Agent3可能多次 | 无Agent间迭代 |
| 上下文传递 | 多次 | 最少（仅必要时） |
| 工作空间 | /tmp/... | 项目.trae/workspace/... |
