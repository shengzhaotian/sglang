# Agent调用模板

本文档提供三个Agent的调用模板，供主Skill参考。

## Agent 1调用模板

```python
# 创建输入文件
input_params = {
    "model_path": model_path,
    "target_device": target_device,
    "special_requirements": special_requirements,
    "task_id": task_id
}
Write(f"{workspace_dir}/input/input_params.json", json.dumps(input_params, indent=2))

# 读取提示模板并填充
prompt_template = Read("prompts/model_analyzer.md")
prompt = prompt_template.replace("{{WORKSPACE_DIR}}", workspace_dir)

# 调用Task
result = Task(
    subagent_type="general_purpose_task",
    query=prompt,
    description="模型架构分析"
)

# 解析输出摘要
output_summary = json.loads(Read(f"{workspace_dir}/output/output_summary.json"))
```

## Agent 2调用模板

```python
# 创建输入文件
error_context = {
    "error_type": error_type,
    "error_message": error_message,
    "error_stacktrace": error_stacktrace,
    "error_location": error_location,
    "run_command": run_command,
    "timestamp": datetime.now().isoformat()
}
Write(f"{workspace_dir}/input/error_context.json", json.dumps(error_context, indent=2))
Write(f"{workspace_dir}/input/analysis_summary.json", json.dumps(analysis_summary, indent=2))
Write(f"{workspace_dir}/input/current_adapter_code.py", current_code)
Write(f"{workspace_dir}/input/previous_fixes.json", json.dumps(previous_fixes, indent=2))

# 读取提示模板并填充
prompt_template = Read("prompts/debug_engineer.md")
prompt = prompt_template.replace("{{WORKSPACE_DIR}}", workspace_dir)
prompt = prompt.replace("{{ITERATION_COUNT}}", str(iteration_count))
prompt = prompt.replace("{{MAX_ITERATIONS}}", "10")

# 调用Task
result = Task(
    subagent_type="general_purpose_task",
    query=prompt,
    description="Debug分析"
)

# 解析修复指令
fix_instructions = json.loads(Read(f"{workspace_dir}/output/fix_instructions.json"))
```

## Agent 3调用模板

```python
# 创建测试配置
test_config = {
    "model_path": model_path,
    "target_device": target_device,
    "tp_size": output_summary["recommended_tp"],
    "test_mode": test_mode,
    "compare_with_hf": compare_with_hf,
    "server_port": 8000,
    "timeout_seconds": 300,
    "attention_backend": "ascend" if target_device == "npu" else "flashinfer",
    "context_length": output_summary["recommended_context_length"],
    "max_running_requests": 16,
    "launch_command": launch_command,
    "config_verified": True
}
Write(f"{workspace_dir}/input/test_config.json", json.dumps(test_config, indent=2))
Write(f"{workspace_dir}/input/analysis_summary.json", json.dumps(output_summary, indent=2))

adapter_info = {
    "adapter_file": adapter_file_path,
    "adapter_class": adapter_class,
    "created_time": created_time
}
Write(f"{workspace_dir}/input/adapter_info.json", json.dumps(adapter_info, indent=2))

# 读取提示模板并填充
prompt_template = Read("prompts/test_validator.md")
prompt = prompt_template.replace("{{WORKSPACE_DIR}}", workspace_dir)

# 调用Task
result = Task(
    subagent_type="general_purpose_task",
    query=prompt,
    description="测试验证"
)

# 解析测试结果
test_result = json.loads(Read(f"{workspace_dir}/output/test_result.json"))
```

## 修复指令执行

```python
def apply_fix(fix_instructions):
    for fix in fix_instructions["fixes"]:
        fix_type = fix["fix_type"]
        target_file = fix["target_file"]
        
        if fix_type == "REPLACE_BLOCK":
            SearchReplace(target_file, fix["old_code"], fix["new_code"])
        elif fix_type == "INSERT_BEFORE":
            content = Read(target_file)
            new_content = content.replace(fix["anchor_code"], fix["new_code"] + fix["anchor_code"])
            Write(target_file, new_content)
        elif fix_type == "INSERT_AFTER":
            content = Read(target_file)
            new_content = content.replace(fix["anchor_code"], fix["anchor_code"] + fix["new_code"])
            Write(target_file, new_content)
        elif fix_type == "DELETE_BLOCK":
            SearchReplace(target_file, fix["old_code"], "")
        elif fix_type == "ADD_FILE":
            Write(target_file, fix["new_code"])
```

## fix_type说明

| 类型 | 说明 | 必填字段 |
|------|------|----------|
| REPLACE_BLOCK | 替换代码块 | old_code, new_code |
| INSERT_BEFORE | 在锚点前插入 | anchor_code, new_code |
| INSERT_AFTER | 在锚点后插入 | anchor_code, new_code |
| DELETE_BLOCK | 删除代码块 | old_code |
| ADD_FILE | 添加新文件 | new_code, target_file |
