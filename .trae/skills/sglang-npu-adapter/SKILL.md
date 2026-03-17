---
name: sglang-npu-adapter
description: 适配新模型到SGLang框架以支持NPU设备。当用户需要在NPU(GPU)上运行SGLang尚未支持的模型时使用此技能。自动分析模型架构、生成适配代码、调试问题并验证正确性。
user-invocable: true
---

# SGLang NPU模型适配技能

帮助用户将新模型适配到SGLang框架，支持在NPU/GPU设备上运行。

## 硬性约束

- **永不升级transformers**
- **主实现根目录为当前项目仓库**
- 启动命令：`export PYTHONPATH=${PWD}/python:$PYTHONPATH`
- 默认API端口：`8000`
- 功能优先默认：验证ACLGraph/DeepEP/DP-Attention/MTP/多模态
- MoE参数（`--ep-size`等）仅适用于MoE模型
- **代码修改最小化，仅针对目标模型**
- **最终交付：单次签名提交**（`git commit -sm ...`）
- **最终文档使用中文**
- **Dummy-first加速，但真实权重验证强制**

---

## 工作流程

```
步骤1:收集上下文 → 步骤2:Agent1分析 → 步骤3:选择策略 → 步骤4:代码修改 
→ 步骤5:两阶段验证(失败→Agent2修复) → 步骤6:Agent3测试(失败→Agent2修复) 
→ 步骤7:生成产物 → 步骤8:交接
```

---

## Agent角色

| Agent | 角色 | 调用时机 | 输出文件 |
|-------|------|----------|----------|
| Agent 1 | 模型架构分析师 | 步骤2 | output_summary.json |
| Agent 2 | Debug工程师 | 步骤5/6失败时 | fix_instructions.json |
| Agent 3 | 测试验证工程师 | 步骤6 | test_result.json |

---

## 执行流程

### 步骤1: 收集上下文

使用AskUserQuestion收集：模型路径、目标设备（npu/gpu）、特殊需求

### 步骤2: 调用Agent 1

创建`input/input_params.json`，调用Task，解析`output/output_summary.json`

**关键输出：** architecture_name, architecture_type, recommended_tp, recommended_ep, npu_compatible, reference_model

### 步骤3: 选择适配策略

- **similarity="high"** → 直接复用参考模型
- **similarity="medium"** → 复用并添加条件分支
- **similarity="low"** → 新建模型文件

### 步骤4: 实现代码修改

**原则：**
1. 优先复用现有架构
2. 修改隔离：使用条件分支，避免影响其他模型
3. NPU兼容：优先torch native实现跑通功能
4. 最小化修改范围

### 步骤5: 两阶段验证

**启动命令构建：**
- 基础参数：`--model-path --port --tp --context-length --device --attention-backend`
- MoE额外：`--ep-size --moe-a2a-backend --deepep-mode`
- VLM额外：`--chat-template`

**服务就绪判断：** 参考`references/shared/service_utils.md`

**Stage A: Dummy验证** → 失败调用Agent 2
**Stage B: 真实权重验证** → 失败调用Agent 2（最多10次迭代）

### 步骤6: 调用Agent 3

创建`input/test_config.json`（基于Agent 1输出），调用Task，解析`output/test_result.json`

**失败处理：**
- `status=failed/error` → 调用Agent 2
- `status=config_issue` → 主Skill调整配置

### 步骤7: 生成产物并提交

1. 生成教程：`./<ModelName>.md`
2. 签名提交：`git commit -sm "feat: adapt <ModelName> for NPU support"`

### 步骤8: 准备交接产物

中文分析报告、运行手册、功能状态矩阵、修改文件列表、提交哈希

---

## 文件结构

```
{WORKSPACE_DIR}/
├── input/           # Agent输入文件
├── output/          # Agent输出文件
├── logs/            # 日志
└── adapter_state.json
```

---

## 错误处理

| 场景 | 处理 |
|------|------|
| Agent 1失败 | 询问用户重试或手动提供信息 |
| 验证失败 | 调用Agent 2，最多10次迭代 |
| Agent 3失败(代码) | 调用Agent 2 |
| Agent 3失败(配置) | 主Skill调整配置 |
| 超过迭代次数 | 询问用户处理方式 |

---

## 质量门禁

- [ ] 服务成功启动
- [ ] 推理请求成功（不仅是启动）
- [ ] 功能集已报告：ACLGraph/DeepEP/MTP/多模态
- [ ] 容量基线（128k+bs16）已报告
- [ ] Dummy+真实权重证据存在
- [ ] 教程文档存在
- [ ] 单次签名提交
- [ ] 最终响应包含提交哈希、文件路径、关键命令

---

## 知识库参考

- `references/shared/sglang_basics.md` - SGLang基础
- `references/shared/npu_basics.md` - NPU基础
- `references/shared/agent_call_templates.md` - Agent调用模板
- `references/shared/service_utils.md` - 服务工具函数
