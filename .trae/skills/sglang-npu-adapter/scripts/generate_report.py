#!/usr/bin/env python3
"""
报告生成脚本
汇总分析报告、测试结果、提交信息，生成最终报告
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def read_json_file(filepath: str) -> Optional[Dict]:
    """读取JSON文件"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def read_markdown_file(filepath: str) -> str:
    """读取Markdown文件"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def get_git_info() -> Dict[str, str]:
    """获取Git信息"""
    info = {
        "branch": "",
        "commit_hash": "",
        "commit_message": "",
        "commit_author": "",
        "commit_date": "",
        "status": ""
    }
    
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        
        info["commit_message"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        info["commit_author"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%an"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        info["commit_date"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%ci"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        status_output = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info["status"] = "clean" if not status_output else "dirty"
        
    except:
        pass
    
    return info


def get_changed_files() -> List[str]:
    """获取修改的文件列表"""
    files = []
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if output:
            files = output.split('\n')
    except:
        pass
    return files


def generate_final_report(
    workspace_dir: str,
    model_name: str,
    output_file: str
) -> str:
    """生成最终报告"""
    
    analysis_summary = read_json_file(f"{workspace_dir}/output/output_summary.json")
    analysis_report = read_markdown_file(f"{workspace_dir}/output/analysis_report.md")
    test_result = read_json_file(f"{workspace_dir}/output/test_result.json")
    test_report = read_markdown_file(f"{workspace_dir}/output/test_report.md")
    debug_report = read_markdown_file(f"{workspace_dir}/output/debug_report.md")
    adapter_state = read_json_file(f"{workspace_dir}/adapter_state.json")
    
    git_info = get_git_info()
    changed_files = get_changed_files()
    
    report_lines = []
    report_lines.append(f"# {model_name} 模型适配报告")
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_lines.append("\n---\n")
    report_lines.append("## 1. 基本信息")
    
    if analysis_summary:
        report_lines.append(f"\n- **模型架构**: {analysis_summary.get('architecture_name', 'N/A')}")
        report_lines.append(f"- **架构类型**: {analysis_summary.get('architecture_type', 'N/A')}")
        report_lines.append(f"- **参考模型**: {analysis_summary.get('reference_model', 'N/A')}")
        report_lines.append(f"- **相似度**: {analysis_summary.get('similarity', 'N/A')}")
        report_lines.append(f"- **NPU兼容**: {'是' if analysis_summary.get('npu_compatible') else '否'}")
    
    report_lines.append("\n---\n")
    report_lines.append("## 2. 配置建议")
    
    if analysis_summary:
        report_lines.append(f"\n- **推荐TP**: {analysis_summary.get('recommended_tp', 'N/A')}")
        report_lines.append(f"- **推荐EP**: {analysis_summary.get('recommended_ep', 'N/A')}")
        report_lines.append(f"- **推荐上下文长度**: {analysis_summary.get('recommended_context_length', 'N/A')}")
        report_lines.append(f"- **权重大小**: {analysis_summary.get('weight_size_gb', 'N/A')} GB")
    
    report_lines.append("\n---\n")
    report_lines.append("## 3. 测试结果")
    
    if test_result:
        status = test_result.get('status', 'N/A')
        status_emoji = "✓" if status == "passed" else "✗"
        report_lines.append(f"\n**总体状态**: {status_emoji} {status}")
        report_lines.append(f"\n- **通过用例**: {test_result.get('passed_count', 0)}/{test_result.get('total_count', 0)}")
        
        if test_result.get('issues'):
            report_lines.append("\n**问题列表**:")
            for issue in test_result['issues']:
                report_lines.append(f"- [{issue.get('severity', 'N/A')}] {issue.get('description', 'N/A')}")
    
    report_lines.append("\n---\n")
    report_lines.append("## 4. 功能状态矩阵")
    
    report_lines.append("\n| 功能 | 状态 | 说明 |")
    report_lines.append("|------|------|------|")
    
    if analysis_summary:
        arch_type = analysis_summary.get('architecture_type', '')
        
        aclgraph_status = "待验证"
        report_lines.append(f"| ACLGraph | {aclgraph_status} | - |")
        
        if 'MoE' in arch_type:
            deepep_status = "待验证"
            report_lines.append(f"| DeepEP | {deepep_status} | MoE模型 |")
        else:
            report_lines.append("| DeepEP | 不适用 | 非MoE模型 |")
        
        if 'VLM' in arch_type:
            multimodal_status = "待验证"
            report_lines.append(f"| 多模态 | {multimodal_status} | VLM模型 |")
        else:
            report_lines.append("| 多模态 | 不适用 | 非VLM模型 |")
    
    report_lines.append("\n---\n")
    report_lines.append("## 5. 验证矩阵")
    
    report_lines.append("\n| 阶段 | 状态 | 说明 |")
    report_lines.append("|------|------|------|")
    
    if adapter_state:
        validation = adapter_state.get('validation', {})
        dummy_passed = validation.get('dummy_passed', False)
        real_passed = validation.get('real_weight_passed', False)
        
        dummy_status = "✓ 通过" if dummy_passed else "✗ 未通过"
        real_status = "✓ 通过" if real_passed else "✗ 未通过"
        
        report_lines.append(f"| Dummy验证 | {dummy_status} | 架构/算子验证 |")
        report_lines.append(f"| 真实权重验证 | {real_status} | 功能/精度验证 |")
        
        if dummy_passed and not real_passed:
            report_lines.append("\n**注意**: Dummy验证通过但真实权重验证未通过，请检查权重映射。")
    
    report_lines.append("\n---\n")
    report_lines.append("## 6. 修改文件")
    
    if changed_files:
        report_lines.append("\n```\n" + "\n".join(changed_files) + "\n```")
    else:
        report_lines.append("\n无修改文件记录")
    
    report_lines.append("\n---\n")
    report_lines.append("## 7. 提交信息")
    
    if git_info:
        report_lines.append(f"\n- **分支**: {git_info.get('branch', 'N/A')}")
        report_lines.append(f"- **提交哈希**: {git_info.get('commit_hash', 'N/A')}")
        report_lines.append(f"- **提交信息**: {git_info.get('commit_message', 'N/A')}")
        report_lines.append(f"- **提交者**: {git_info.get('commit_author', 'N/A')}")
        report_lines.append(f"- **提交时间**: {git_info.get('commit_date', 'N/A')}")
        report_lines.append(f"- **工作区状态**: {git_info.get('status', 'N/A')}")
    
    report_lines.append("\n---\n")
    report_lines.append("## 8. 关键发现")
    
    if analysis_summary and analysis_summary.get('key_findings'):
        report_lines.append("")
        for i, finding in enumerate(analysis_summary['key_findings'], 1):
            report_lines.append(f"{i}. {finding}")
    
    if analysis_summary and analysis_summary.get('warnings'):
        report_lines.append("\n**警告**:")
        for warning in analysis_summary['warnings']:
            report_lines.append(f"- {warning}")
    
    report_lines.append("\n---\n")
    report_lines.append("## 9. 运行命令")
    
    if analysis_summary:
        tp = analysis_summary.get('recommended_tp', 1)
        ctx = analysis_summary.get('recommended_context_length', 4096)
        
        report_lines.append("\n```bash")
        report_lines.append("export PYTHONPATH=${PWD}/python:$PYTHONPATH")
        report_lines.append(f"python -m sglang.launch_server \\")
        report_lines.append(f"    --model-path /path/to/{model_name} \\")
        report_lines.append(f"    --port 8000 \\")
        report_lines.append(f"    --tp {tp} \\")
        report_lines.append(f"    --context-length {ctx} \\")
        report_lines.append(f"    --device npu \\")
        report_lines.append(f"    --attention-backend ascend")
        report_lines.append("```")
    
    report_lines.append("\n---")
    report_lines.append("\n*报告由SGLang NPU适配技能套件自动生成*")
    
    report_content = "\n".join(report_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="报告生成脚本")
    parser.add_argument("--workspace", "-w", required=True, help="工作目录路径")
    parser.add_argument("--model", "-m", required=True, help="模型名称")
    parser.add_argument("--output", "-o", required=True, help="输出报告文件路径")
    args = parser.parse_args()
    
    report = generate_final_report(args.workspace, args.model, args.output)
    print(f"报告已生成: {args.output}")
    print("\n" + "=" * 60)
    print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()
