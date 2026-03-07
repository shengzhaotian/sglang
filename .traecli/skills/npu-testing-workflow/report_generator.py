#!/usr/bin/env python3
"""
NPU Test Report Generator
This module provides functionality to generate comprehensive test reports
from benchmark results, including visualization and comparison features.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TestEnvironment:
    hardware: str = ""
    cann_version: str = ""
    sglang_version: str = ""
    model_name: str = ""
    model_path: str = ""
    deployment_mode: str = ""
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    quantization: str = ""
    dtype: str = ""
    mem_fraction: float = 0.0
    max_running_requests: int = 0
    cuda_graph_bs: str = ""


@dataclass
class PerformanceResult:
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_p99: float = 0.0
    tpot_mean: float = 0.0
    tpot_median: float = 0.0
    tpot_p99: float = 0.0
    throughput: float = 0.0
    itl_mean: float = 0.0
    itl_median: float = 0.0
    itl_p99: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0


@dataclass
class AccuracyResult:
    benchmark_name: str = ""
    score: float = 0.0
    total_questions: int = 0
    correct_answers: int = 0
    baseline_score: Optional[float] = None


@dataclass
class FunctionalTestResult:
    name: str = ""
    expected: str = ""
    actual: str = ""
    passed: bool = False
    error: Optional[str] = None


@dataclass
class ComparisonData:
    metric_name: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    difference_percent: float = 0.0
    status: str = ""


class NPUTestReportGenerator:
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now()
        
        self.environment: Optional[TestEnvironment] = None
        self.performance: Optional[PerformanceResult] = None
        self.accuracy: Optional[AccuracyResult] = None
        self.functional_tests: List[FunctionalTestResult] = []
        self.issues: List[str] = []
        self.suggestions: List[str] = []
        
        self.baselines: Dict[str, float] = {}
        self.comparisons: List[ComparisonData] = []
    
    def set_environment(self, **kwargs):
        self.environment = TestEnvironment(**kwargs)
    
    def set_performance(self, **kwargs):
        self.performance = PerformanceResult(**kwargs)
    
    def set_accuracy(self, **kwargs):
        self.accuracy = AccuracyResult(**kwargs)
    
    def add_functional_test(self, **kwargs):
        self.functional_tests.append(FunctionalTestResult(**kwargs))
    
    def add_issue(self, issue: str):
        self.issues.append(issue)
    
    def add_suggestion(self, suggestion: str):
        self.suggestions.append(suggestion)
    
    def load_baselines(self, baseline_file: str):
        with open(baseline_file, "r") as f:
            self.baselines = json.load(f)
    
    def set_baselines(self, baselines: Dict[str, float]):
        self.baselines = baselines
    
    def _compute_comparisons(self):
        self.comparisons = []
        
        if not self.performance or not self.baselines:
            return
        
        metrics_mapping = {
            "ttft_mean": self.performance.ttft_mean,
            "ttft_median": self.performance.ttft_median,
            "ttft_p99": self.performance.ttft_p99,
            "tpot_mean": self.performance.tpot_mean,
            "tpot_median": self.performance.tpot_median,
            "tpot_p99": self.performance.tpot_p99,
            "throughput": self.performance.throughput,
            "itl_mean": self.performance.itl_mean,
            "itl_median": self.performance.itl_median,
            "itl_p99": self.performance.itl_p99,
        }
        
        for metric_name, current_value in metrics_mapping.items():
            if metric_name in self.baselines:
                baseline_value = self.baselines[metric_name]
                if baseline_value != 0:
                    diff_percent = ((current_value - baseline_value) / baseline_value) * 100
                    
                    if metric_name == "throughput":
                        status = "BETTER" if diff_percent > 0 else "WORSE" if diff_percent < 0 else "SAME"
                    else:
                        status = "BETTER" if diff_percent < 0 else "WORSE" if diff_percent > 0 else "SAME"
                else:
                    diff_percent = 0
                    status = "N/A"
                
                self.comparisons.append(ComparisonData(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    difference_percent=diff_percent,
                    status=status
                ))
    
    def _analyze_issues(self):
        if not self.performance:
            return
        
        if self.performance.tpot_mean > 100:
            self.add_issue(f"High TPOT detected: {self.performance.tpot_mean:.2f}ms")
            self.add_suggestion("Consider increasing DP size or optimizing CUDA graph batch sizes")
        
        if self.performance.ttft_mean > 500:
            self.add_issue(f"High TTFT detected: {self.performance.ttft_mean:.2f}ms")
            self.add_suggestion("Consider increasing max-prefill-tokens or using PD separation mode")
        
        if self.performance.failed_requests > 0:
            fail_rate = self.performance.failed_requests / self.performance.total_requests * 100
            self.add_issue(f"Request failure rate: {fail_rate:.2f}%")
            self.add_suggestion("Check server logs for errors, consider reducing max-running-requests")
        
        if self.performance.itl_p99 > self.performance.itl_mean * 3:
            self.add_issue(f"High ITL variance detected (P99: {self.performance.itl_p99:.2f}ms vs Mean: {self.performance.itl_mean:.2f}ms)")
            self.add_suggestion("Consider optimizing memory allocation or checking for network issues")
        
        if self.environment and self.environment.mem_fraction > 0.85:
            self.add_issue(f"High memory fraction: {self.environment.mem_fraction}")
            self.add_suggestion("Consider reducing mem-fraction-static to avoid OOM errors")
    
    def generate_markdown_report(self) -> str:
        self._compute_comparisons()
        self._analyze_issues()
        
        lines = []
        
        lines.append("# NPU Model Testing Report")
        lines.append(f"\n**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("## 1. Test Environment Configuration")
        lines.append("")
        if self.environment:
            lines.append("| Configuration | Value |")
            lines.append("|--------------|-------|")
            env_dict = asdict(self.environment)
            for key, value in env_dict.items():
                if value:
                    key_display = key.replace("_", " ").title()
                    lines.append(f"| {key_display} | {value} |")
        else:
            lines.append("*Environment information not provided*")
        
        lines.append("")
        lines.append("## 2. Functional Verification Results")
        lines.append("")
        if self.functional_tests:
            lines.append("| Test Case | Expected | Actual | Status |")
            lines.append("|-----------|----------|--------|--------|")
            for test in self.functional_tests:
                status = "PASS" if test.passed else "FAIL"
                actual_display = test.actual[:50] + "..." if len(test.actual) > 50 else test.actual
                lines.append(f"| {test.name} | {test.expected[:30]}... | {actual_display} | {status} |")
            
            passed = sum(1 for t in self.functional_tests if t.passed)
            total = len(self.functional_tests)
            lines.append("")
            lines.append(f"**Pass Rate:** {passed}/{total} ({passed/total*100:.1f}%)")
        else:
            lines.append("*No functional tests performed*")
        
        lines.append("")
        lines.append("## 3. Performance Metrics")
        lines.append("")
        if self.performance:
            lines.append("### 3.1 Latency Metrics")
            lines.append("")
            lines.append("| Metric | Mean (ms) | Median (ms) | P99 (ms) |")
            lines.append("|--------|-----------|-------------|----------|")
            lines.append(f"| TTFT | {self.performance.ttft_mean:.2f} | {self.performance.ttft_median:.2f} | {self.performance.ttft_p99:.2f} |")
            lines.append(f"| TPOT | {self.performance.tpot_mean:.2f} | {self.performance.tpot_median:.2f} | {self.performance.tpot_p99:.2f} |")
            lines.append(f"| ITL | {self.performance.itl_mean:.2f} | {self.performance.itl_median:.2f} | {self.performance.itl_p99:.2f} |")
            
            lines.append("")
            lines.append("### 3.2 Throughput Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Output Throughput | {self.performance.throughput:.2f} tok/s |")
            lines.append(f"| Total Requests | {self.performance.total_requests} |")
            lines.append(f"| Successful Requests | {self.performance.successful_requests} |")
            lines.append(f"| Failed Requests | {self.performance.failed_requests} |")
            lines.append(f"| Total Time | {self.performance.total_time:.2f} s |")
            
            if self.comparisons:
                lines.append("")
                lines.append("### 3.3 Performance Comparison with Baseline")
                lines.append("")
                lines.append("| Metric | Current | Baseline | Diff (%) | Status |")
                lines.append("|--------|---------|----------|----------|--------|")
                for comp in self.comparisons:
                    lines.append(f"| {comp.metric_name} | {comp.current_value:.2f} | {comp.baseline_value:.2f} | {comp.difference_percent:+.2f}% | {comp.status} |")
        else:
            lines.append("*No performance tests performed*")
        
        lines.append("")
        lines.append("## 4. Accuracy Evaluation Results")
        lines.append("")
        if self.accuracy:
            lines.append("| Benchmark | Score | Baseline | Questions | Status |")
            lines.append("|-----------|-------|----------|-----------|--------|")
            baseline_str = f"{self.accuracy.baseline_score:.4f}" if self.accuracy.baseline_score else "N/A"
            if self.accuracy.baseline_score:
                diff = self.accuracy.score - self.accuracy.baseline_score
                status = "PASS" if diff >= 0 else "FAIL"
            else:
                status = "N/A"
            lines.append(f"| {self.accuracy.benchmark_name} | {self.accuracy.score:.4f} | {baseline_str} | {self.accuracy.total_questions} | {status} |")
        else:
            lines.append("*No accuracy tests performed*")
        
        lines.append("")
        lines.append("## 5. Issues Found and Optimization Suggestions")
        lines.append("")
        if self.issues:
            lines.append("### 5.1 Issues Found")
            lines.append("")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"{i}. {issue}")
        else:
            lines.append("*No issues found*")
        
        lines.append("")
        if self.suggestions:
            lines.append("### 5.2 Optimization Suggestions")
            lines.append("")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"{i}. {suggestion}")
        
        lines.append("")
        lines.append("## 6. Test Conclusion")
        lines.append("")
        
        func_pass_rate = 0
        if self.functional_tests:
            func_pass_rate = sum(1 for t in self.functional_tests if t.passed) / len(self.functional_tests)
        
        req_success_rate = 0
        if self.performance and self.performance.total_requests > 0:
            req_success_rate = self.performance.successful_requests / self.performance.total_requests
        
        overall_status = "PASS"
        if func_pass_rate < 1.0 or req_success_rate < 0.99:
            overall_status = "PARTIAL PASS"
        if func_pass_rate < 0.5 or req_success_rate < 0.9:
            overall_status = "FAIL"
        
        lines.append(f"**Overall Status:** {overall_status}")
        lines.append("")
        lines.append("### Summary")
        lines.append(f"- Functional Test Pass Rate: {func_pass_rate*100:.1f}%")
        lines.append(f"- Request Success Rate: {req_success_rate*100:.1f}%")
        if self.performance:
            lines.append(f"- Average TTFT: {self.performance.ttft_mean:.2f} ms")
            lines.append(f"- Average TPOT: {self.performance.tpot_mean:.2f} ms")
            lines.append(f"- Throughput: {self.performance.throughput:.2f} tok/s")
        if self.accuracy:
            lines.append(f"- Accuracy ({self.accuracy.benchmark_name}): {self.accuracy.score:.4f}")
        
        lines.append("")
        lines.append("### Next Steps")
        if overall_status == "PASS":
            lines.append("- Model is ready for production deployment")
            lines.append("- Consider running extended stability tests")
            lines.append("- Monitor performance metrics in production")
        elif overall_status == "PARTIAL PASS":
            lines.append("- Address identified issues before production")
            lines.append("- Re-run tests after applying optimizations")
            lines.append("- Consider performance tuning for better results")
        else:
            lines.append("- Critical issues need to be resolved")
            lines.append("- Review configuration and deployment settings")
            lines.append("- Contact support if issues persist")
        
        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated by NPU Test Report Generator on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "environment": asdict(self.environment) if self.environment else None,
            "performance": asdict(self.performance) if self.performance else None,
            "accuracy": asdict(self.accuracy) if self.accuracy else None,
            "functional_tests": [asdict(t) for t in self.functional_tests],
            "comparisons": [asdict(c) for c in self.comparisons],
            "issues": self.issues,
            "suggestions": self.suggestions,
        }
    
    def save_reports(self, prefix: str = "npu_test"):
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        md_path = self.output_dir / f"{prefix}_{timestamp_str}.md"
        md_content = self.generate_markdown_report()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Markdown report saved to: {md_path}")
        
        json_path = self.output_dir / f"{prefix}_{timestamp_str}.json"
        json_content = self.generate_json_report()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)
        print(f"JSON report saved to: {json_path}")
        
        return md_path, json_path
    
    def load_from_json(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if data.get("environment"):
            self.environment = TestEnvironment(**data["environment"])
        if data.get("performance"):
            self.performance = PerformanceResult(**data["performance"])
        if data.get("accuracy"):
            self.accuracy = AccuracyResult(**data["accuracy"])
        if data.get("functional_tests"):
            self.functional_tests = [FunctionalTestResult(**t) for t in data["functional_tests"]]
        if data.get("issues"):
            self.issues = data["issues"]
        if data.get("suggestions"):
            self.suggestions = data["suggestions"]


def create_sample_report():
    generator = NPUTestReportGenerator()
    
    generator.set_environment(
        hardware="Atlas 800I A3",
        cann_version="8.5.0",
        sglang_version="0.4.0",
        model_name="DeepSeek-R1",
        model_path="/path/to/deepseek-r1",
        deployment_mode="PD Separation",
        tp_size=32,
        dp_size=32,
        ep_size=1,
        quantization="modelslim W8A8",
        dtype="bfloat16",
        mem_fraction=0.8,
        max_running_requests=832,
        cuda_graph_bs="12 14 16 18 20 22 24 26"
    )
    
    generator.set_performance(
        ttft_mean=156.32,
        ttft_median=142.18,
        ttft_p99=312.45,
        tpot_mean=48.72,
        tpot_median=45.33,
        tpot_p99=89.21,
        throughput=1256.78,
        itl_mean=48.12,
        itl_median=45.01,
        itl_p99=92.34,
        total_requests=3072,
        successful_requests=3068,
        failed_requests=4,
        total_time=3672.45
    )
    
    generator.set_accuracy(
        benchmark_name="GSM8K",
        score=0.8756,
        total_questions=200,
        correct_answers=175,
        baseline_score=0.8700
    )
    
    generator.add_functional_test(
        name="Basic Inference",
        expected="Contains '4'",
        actual="The answer is 4.",
        passed=True
    )
    generator.add_functional_test(
        name="Streaming Output",
        expected="Contains numbers 1-5",
        actual="1, 2, 3, 4, 5",
        passed=True
    )
    generator.add_functional_test(
        name="Long Context",
        expected="Contains '1000'",
        actual="There were approximately 1000 words.",
        passed=True
    )
    
    generator.set_baselines({
        "ttft_mean": 150.0,
        "tpot_mean": 50.0,
        "throughput": 1200.0,
        "itl_mean": 50.0
    })
    
    generator.save_reports()
    
    print("\n" + generator.generate_markdown_report())


if __name__ == "__main__":
    create_sample_report()
