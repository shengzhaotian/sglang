---
name: sglang-npu-adapter
description: Adapt new models to the SGLang framework to support NPU devices. Use this skill when users need to run models not yet supported by SGLang on NPU(GPU) devices. Automatically analyzes model architecture, generates adaptation code, debugs issues, and verifies correctness.
user-invocable: true
---

# SGLang NPU Model Adaptation Skill

Help users adapt new models to the SGLang framework to support running on NPU/GPU devices.

## Task Tracking and Memory Mechanism

To ensure the effective execution of complex tasks, this skill integrates the planning-with-files pattern, using three core memory files to track task progress and important information:

1. **task_plan.md** - Records task phases, progress, and key decisions (based on `templates/task_plan.md` template)
2. **findings.md** - Records research findings, model analysis results, and technical insights (based on `templates/findings.md` template)
3. **progress.md** - Records session logs, execution steps, and test results (based on `templates/progress.md` template)

### Template Management

Planning file templates are uniformly stored in the `templates/` directory for easy maintenance and updates. 

### Relationship Between Task Flow and Planning Files

- **Initialization Phase**: Create planning files based on templates
- **Before each step**: Read planning files to restore context
- **After each step**: Update planning files to record progress and findings
- **When task is completed**: Finally update planning files to summarize the entire task execution process

This mechanism ensures that information during task execution is not lost and supports task interruption and recovery.

## Hard Constraints

- For each task, create or read an independent folder in .trae/workspace/ to store input and output during the process
- **Never upgrade transformers**
- **Main implementation root directory is the current project repository**
- Startup command: `export PYTHONPATH=${PWD}/python:$PYTHONPATH`
- Default API port: `8000`
- Function priority default: Verify ACLGraph/DeepEP/DP-Attention/MTP/multimodal
- **Minimize code modifications, only for target models**
- **Final delivery: Single signed commit** (`git commit -sm ...`)
- **Final documentation in Chinese**
- **Dummy-first acceleration, but real weight verification is mandatory**
- **Task tracking: Must use planning-with-files mode**

---

## Architecture Design

### Agent Roles

| Agent | Role | Core Capabilities | Trigger Timing |
|-------|------|-------------------|----------------|
| Agent 1 | Architecture Analyst | Model identification, resource evaluation, configuration derivation, risk identification | Step 2 |
| Agent 2 | Debug Engineer | Error diagnosis, context correlation, fix generation | Automatically triggered when any step fails |
| Agent 3 | Verification Engineer | Test generation, execution verification, result analysis | Step 6 |

### Agent Calling Specifications

#### Call Type Constraints

**All Agent calls must use `general_purpose_task` type, forbidden to use other types (such as `search`).**
**For each Agent, the calling prompt must follow the template in references/shared/agent_call_templates.md**

-------

## Execution Flow

### Step 0: Initialize State Files and Planning Files

**【Mandatory】Execute before any operation:**

1. Create working directory: `mkdir -p {WORKSPACE_DIR}/{input,output,logs}`

2. **【Mandatory】Initialize planning files (planning-with-files mode):**
   - Create `task_plan.md`: Contains task phases, progress, and key decisions
   - Create `findings.md`: For recording research findings and technical insights
   - Create `progress.md`: For recording session logs and execution results

**Usage**: Directly read the corresponding template file and fill in variables (such as <ModelName>, <ModelPath>, {WORKSPACE_DIR})

### Step 1: Collect Context

**Responsibility: Collect information, do not make technical decisions**

1. Use AskUserQuestion to collect:
   - Model path
   - Target device (npu/gpu)
   - Special requirements

2. **【Mandatory】Device detection:**
```bash
# NPU device detection
npu-smi info 2>/dev/null | grep "Ascend" | wc -l
```

3. Write to `input/device_info.json`

4. **【Mandatory】Update planning files:**
   
### Step 2: Call Agent 1

1. **【Mandatory】Read planning files to confirm current step**
2. Create `input/input_params.json` (contains device_info)
3. Read `prompts/model_analyzer.md` and fill in `{{WORKSPACE_DIR}}`
4. **Call Task to start Agent 1**:
   - **subagent_type must be `general_purpose_task`**
   - query is the filled prompt content
   - description is "Model architecture analysis"
5. Parse `output/output_summary.json`
6. **【Mandatory】Update planning files:**
   - Update `task_plan.md`: Mark model architecture analysis completion status
   - Update `findings.md`: Record Agent 1's analysis results and parallel configuration derivation
   - Update `progress.md`: Record Agent 1 call process and output results

### Step 3: Select Adaptation Strategy

**Based on Agent 1's similarity output:**
- `high` → Directly reuse reference model
- `medium` → Reuse and add conditional branches
- `low` → Create new model file

**【Mandatory】Update planning files:**
   - Update `task_plan.md`: Mark adaptation strategy selection completion
   - Update `findings.md`: Record the selected adaptation strategy and reasons
   - Update `progress.md`: Record adaptation strategy selection process

### Step 4: Implement Code Modifications

**Principles:**
1. Prioritize reusing existing architecture
2. Modification isolation: Use conditional branches
3. NPU compatibility: Prioritize torch native implementation
4. Minimize modification scope

**【Mandatory】Update planning files:**
   - Update `task_plan.md`: Mark code modification completion status
   - Update `findings.md`: Record modified files, key code changes, and technical decisions
   - Update `progress.md`: Record code implementation process and results

### Step 5: Two-Phase Verification

**Responsibility: Execute verification, automatically trigger Agent 2 on failure**

**【Mandatory】Read planning files to confirm iteration count before execution**

**Stage A: Dummy verification**
- Success → Enter Stage B, **update planning files**
- Failure → **【Mandatory】Update planning files and call Agent 2**:
   - Update `task_plan.md`: Mark verification failure
   - Update `findings.md`: Record error information and possible causes
   - Update `progress.md`: Record verification process and failure results

**Stage B: Real weight verification**
- Success → Enter Step 6, **update planning files**
- Failure → **【Mandatory】Update planning files and call Agent 2**:
   - Update `task_plan.md`: Mark verification failure
   - Update `findings.md`: Record error information and possible causes
   - Update `progress.md`: Record verification process and failure results

**Update after verification success:**
   - Update `task_plan.md`: Mark verification phase completion status
   - Update `findings.md`: Record verification results and performance indicators
   - Update `progress.md`: Record verification process and success results

### Step 6: Call Agent 3

**Responsibility: Orchestrate calls, do not make test decisions**

1. **【Mandatory】Read planning files to confirm current step**
2. Create `input/test_config.json` (based on Agent 1 output)
3. Read `prompts/test_validator.md` and fill in `{{WORKSPACE_DIR}}`
4. **Call Task to start Agent 3**:
   - **subagent_type must be `general_purpose_task`**
   - query is the filled prompt content
   - description is "Test verification"
5. Parse `output/test_result.json`
6. **【Mandatory】Update planning files:**
   - Update `task_plan.md`: Mark test verification completion status
   - Update `findings.md`: Record Agent 3's test results and issues
   - Update `progress.md`: Record Agent 3 call process and output results

### Step 7: Generate Artifacts and Submit

1. Generate tutorial: `./<ModelName>.md`
2. Signed commit: `git commit -sm "feat: adapt <ModelName> for NPU support"`

**【Mandatory】Update planning files:**
   - Update `task_plan.md`: Mark task completion, update all phase statuses
   - Update `findings.md`: Record final artifact information and commit hash
   - Update `progress.md`: Record completion process and final results

### Step 8: Prepare Handover Artifacts

Chinese analysis report, operation manual, feature status matrix, modified file list, commit hash

---

## File Structure

```
{WORKSPACE_DIR}/
├── adapter_state.json        # ⚠️ Core status file (must be read/written before/after each operation)
├── task_plan.md              # ⚠️ Task plan file (planning-with-files mode)
├── findings.md               # ⚠️ Findings record file (planning-with-files mode)
├── progress.md               # ⚠️ Progress record file (planning-with-files mode)
├── input/
│   ├── input_params.json      # Agent 1 input
│   ├── device_info.json       # Device information
│   ├── error_context.json     # Agent 2 input
│   └── test_config.json       # Agent 3 input
├── output/
│   ├── output_summary.json    # Agent 1 output
│   ├── fix_instructions.json  # Agent 2 output
│   └── test_result.json       # Agent 3 output
├── logs/
│   ├── dummy_run.log          # Dummy verification log
│   └── real_run.log           # Real verification log
```

---

## Anti-forgetting Checklist

### Before Each Operation

- [ ] Read `adapter_state.json` (machine-readable status)
- [ ] Read `task_plan.md` (task phases and progress)
- [ ] Read `findings.md` (research findings and technical insights)
- [ ] Read `progress.md` (session logs and execution results)
- [ ] Confirm current step
- [ ] Check if `next_action` is `call_agent2`
- [ ] Check if iteration count exceeds limit

### After Each Operation

- [ ] Update `adapter_state.json` (machine-readable status)
- [ ] Update `task_plan.md` (task phases and progress)
- [ ] Update `findings.md` (research findings and technical insights)
- [ ] Update `progress.md` (session logs and execution results)
- [ ] Record error information (if any)
- [ ] Set correct `next_action`
- [ ] Update timestamp

### When Verification Fails

- [ ] Immediately update all planning files
- [ ] Record complete error log to adapter_state.json
- [ ] Analyze error causes in findings.md
- [ ] Record failure process in progress.md
- [ ] Increase iteration count
- [ ] Call Agent 2

---

## Reference Documents Structure

```
references/
├── agent1_analyst/               # Model Architecture Analyst references
│   ├── llm_architecture.md       # LLM architecture identification
│   ├── moe_architecture.md       # MoE architecture details
│   ├── mla_architecture.md       # MLA architecture details
│   ├── vlm_architecture.md       # VLM architecture details
│   ├── memory_calculation.md     # Memory calculation models
│   ├── npu_specifications.md     # NPU specifications
│   └── sglang_model_registry.md  # SGLang model registry
├── agent2_debug/                 # Debug Engineer references
│   ├── common_errors.md          # Common error patterns
│   ├── npu_specific_issues.md    # NPU specific issues
│   ├── attention_debug.md        # Attention mechanism debugging
│   ├── moe_debug.md              # MoE debugging
│   └── rope_debug.md             # RoPE position encoding debugging
├── agent3_validator/             # Test Validation Engineer references
│   ├── basic_inference_test.md   # Basic inference tests
│   ├── correctness_validation.md # Correctness validation methods
│   ├── npu_validation.md         # NPU-specific validation
│   └── performance_benchmark.md  # Performance benchmarking
└── shared/
    ├── sglang_basics.md          # SGLang fundamentals
    ├── npu_basics.md             # NPU basics
    ├── service_utils.md          # Service utilities
    └── agent_call_templates.md   # Agent calling templates
```

---

## Quality Gate

- [ ] Service started successfully
- [ ] Inference request succeeded (not just startup)
- [ ] Feature set reported: ACLGraph/DeepEP/MTP/multimodal
- [ ] Capacity baseline (128k+bs16) reported
- [ ] Dummy+real weight evidence exists
- [ ] Tutorial documentation exists
- [ ] Single signed commit
- [ ] Final response contains commit hash, file paths, key commands