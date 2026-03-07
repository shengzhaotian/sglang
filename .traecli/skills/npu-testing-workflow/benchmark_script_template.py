#!/usr/bin/env python3
"""
NPU Client Benchmarking Script Template
This script provides comprehensive benchmarking capabilities for NPU model testing,
including performance testing, functional verification, and accuracy evaluation.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    host: str = "127.0.0.1"
    port: int = 6688
    model: str = "default"
    max_concurrency: int = 256
    num_prompts: int = 1024
    random_input_len: int = 3500
    random_output_len: int = 1500
    random_range_ratio: float = 1.0
    request_rate: float = 0.0
    timeout: int = 3600
    output_dir: str = "./benchmark_results"
    dataset_name: str = "random"
    dataset_path: Optional[str] = None
    accuracy_test: bool = False
    num_questions: int = 200
    max_new_tokens: int = 512
    num_shots: int = 5


@dataclass
class PerformanceMetrics:
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_p99: float = 0.0
    tpot_mean: float = 0.0
    tpot_median: float = 0.0
    tpot_p99: float = 0.0
    throughput: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time: float = 0.0
    itl_mean: float = 0.0
    itl_median: float = 0.0
    itl_p99: float = 0.0


@dataclass
class AccuracyMetrics:
    benchmark_name: str = ""
    accuracy: float = 0.0
    total_questions: int = 0
    correct_answers: int = 0
    invalid_answers: int = 0


@dataclass
class TestResult:
    config: BenchmarkConfig
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    accuracy: Optional[AccuracyMetrics] = None
    functional_tests: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NPUBenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}"
        self.results_dir = Path(config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def wait_for_server(self, timeout: int = 120) -> bool:
        print(f"Waiting for server at {self.base_url}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Server ready after {time.time() - start_time:.1f}s")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        print(f"Server not ready after {timeout}s")
        return False
    
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        request_id: int
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True
        }
        
        result = {
            "request_id": request_id,
            "success": False,
            "ttft": 0.0,
            "itl": [],
            "latency": 0.0,
            "output_tokens": 0,
            "generated_text": "",
            "error": None
        }
        
        start_time = time.perf_counter()
        first_token_time = None
        last_token_time = start_time
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    result["error"] = f"HTTP {response.status}"
                    return result
                
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            token_time = time.perf_counter()
                            
                            if first_token_time is None:
                                first_token_time = token_time
                                result["ttft"] = (token_time - start_time) * 1000
                            else:
                                result["itl"].append((token_time - last_token_time) * 1000)
                            
                            last_token_time = token_time
                            result["output_tokens"] += 1
                            
                            if "text" in data["choices"][0]:
                                result["generated_text"] += data["choices"][0]["text"]
                    except json.JSONDecodeError:
                        continue
                
                result["success"] = True
                result["latency"] = (time.perf_counter() - start_time) * 1000
                
        except asyncio.TimeoutError:
            result["error"] = "Request timeout"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def run_performance_benchmark(self) -> PerformanceMetrics:
        print(f"\n{'='*60}")
        print("Running Performance Benchmark")
        print(f"{'='*60}")
        print(f"Concurrency: {self.config.max_concurrency}")
        print(f"Num prompts: {self.config.num_prompts}")
        print(f"Input length: {self.config.random_input_len}")
        print(f"Output length: {self.config.random_output_len}")
        
        prompts = self._generate_random_prompts()
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrency * 2)
        
        all_results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            semaphore = asyncio.Semaphore(self.config.max_concurrency)
            
            async def bounded_request(prompt, max_tokens, request_id):
                async with semaphore:
                    return await self.send_request(session, prompt, max_tokens, request_id)
            
            tasks = [
                bounded_request(prompt, self.config.random_output_len, i)
                for i, prompt in enumerate(prompts)
            ]
            
            with tqdm(total=len(tasks), desc="Processing requests") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    all_results.append(result)
                    pbar.update(1)
        
        total_time = time.time() - start_time
        return self._compute_metrics(all_results, total_time)
    
    def _generate_random_prompts(self) -> List[str]:
        np.random.seed(42)
        prompts = []
        for _ in range(self.config.num_prompts):
            input_len = self.config.random_input_len
            if self.config.random_range_ratio < 1.0:
                input_len = int(input_len * (1 + np.random.uniform(
                    -self.config.random_range_ratio, 
                    self.config.random_range_ratio
                )))
            prompt = " ".join(["test"] * input_len)
            prompts.append(prompt)
        return prompts
    
    def _compute_metrics(
        self, 
        results: List[Dict[str, Any]], 
        total_time: float
    ) -> PerformanceMetrics:
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        ttfts = [r["ttft"] for r in successful if r["ttft"] > 0]
        tpots = []
        for r in successful:
            if r["output_tokens"] > 1 and r["latency"] > 0:
                tpot = (r["latency"] - r["ttft"]) / (r["output_tokens"] - 1)
                tpots.append(tpot)
        
        all_itls = []
        for r in successful:
            all_itls.extend(r["itl"])
        
        total_output_tokens = sum(r["output_tokens"] for r in successful)
        total_input_tokens = self.config.random_input_len * len(successful)
        
        metrics = PerformanceMetrics(
            ttft_mean=np.mean(ttfts) if ttfts else 0,
            ttft_median=np.median(ttfts) if ttfts else 0,
            ttft_p99=np.percentile(ttfts, 99) if ttfts else 0,
            tpot_mean=np.mean(tpots) if tpots else 0,
            tpot_median=np.median(tpots) if tpots else 0,
            tpot_p99=np.percentile(tpots, 99) if tpots else 0,
            throughput=total_output_tokens / total_time if total_time > 0 else 0,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_time=total_time,
            itl_mean=np.mean(all_itls) if all_itls else 0,
            itl_median=np.median(all_itls) if all_itls else 0,
            itl_p99=np.percentile(all_itls, 99) if all_itls else 0,
        )
        
        return metrics
    
    def run_functional_tests(self) -> List[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print("Running Functional Tests")
        print(f"{'='*60}")
        
        test_cases = [
            {
                "name": "Basic Inference",
                "prompt": "What is 2 + 2?",
                "expected_pattern": r"4",
                "max_tokens": 50
            },
            {
                "name": "Streaming Output",
                "prompt": "Count from 1 to 5.",
                "expected_pattern": r"(1|one).*(2|two).*(3|three).*(4|four).*(5|five)",
                "max_tokens": 100
            },
            {
                "name": "Long Context",
                "prompt": " ".join(["word"] * 1000) + "\nHow many words were in the previous text?",
                "expected_pattern": r"1000|thousand",
                "max_tokens": 50
            },
            {
                "name": "JSON Generation",
                "prompt": 'Generate a JSON object with fields "name" and "age".',
                "expected_pattern": r'\{.*"name".*:.*"age".*\}',
                "max_tokens": 100
            }
        ]
        
        results = []
        for test in test_cases:
            print(f"  Testing: {test['name']}...")
            result = {
                "name": test["name"],
                "expected": test["expected_pattern"],
                "actual": "",
                "passed": False,
                "error": None
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.config.model,
                        "prompt": test["prompt"],
                        "max_tokens": test["max_tokens"],
                        "temperature": 0.0
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        result["actual"] = data["choices"][0].get("text", "")
                        if re.search(test["expected_pattern"], result["actual"], re.IGNORECASE):
                            result["passed"] = True
                else:
                    result["error"] = f"HTTP {response.status_code}"
            except Exception as e:
                result["error"] = str(e)
            
            status = "PASS" if result["passed"] else "FAIL"
            print(f"    {status}: {test['name']}")
            results.append(result)
        
        return results
    
    def run_accuracy_test(self) -> Optional[AccuracyMetrics]:
        if not self.config.accuracy_test:
            return None
        
        print(f"\n{'='*60}")
        print("Running Accuracy Test (GSM8K)")
        print(f"{'='*60}")
        
        try:
            from sglang.test.few_shot_gsm8k import run_eval
            from types import SimpleNamespace
            
            args = SimpleNamespace(
                num_shots=self.config.num_shots,
                data_path=self.config.dataset_path,
                num_questions=self.config.num_questions,
                max_new_tokens=self.config.max_new_tokens,
                parallel=min(32, self.config.max_concurrency),
                host=f"http://{self.config.host}",
                port=self.config.port,
            )
            
            metrics = run_eval(args)
            
            return AccuracyMetrics(
                benchmark_name="GSM8K",
                accuracy=metrics.get("accuracy", 0.0),
                total_questions=self.config.num_questions,
                correct_answers=int(metrics.get("accuracy", 0) * self.config.num_questions),
                invalid_answers=metrics.get("invalid", 0)
            )
        except ImportError:
            print("Warning: Could not import GSM8K test module")
            return None
        except Exception as e:
            print(f"Error running accuracy test: {e}")
            return None
    
    def generate_report(self, result: TestResult) -> str:
        report = []
        report.append("# NPU Model Testing Report")
        report.append(f"\nGenerated: {result.timestamp}")
        
        report.append("\n## 1. Test Environment Configuration")
        report.append(f"- Host: {result.config.host}:{result.config.port}")
        report.append(f"- Model: {result.config.model}")
        report.append(f"- Max Concurrency: {result.config.max_concurrency}")
        report.append(f"- Input Length: {result.config.random_input_len}")
        report.append(f"- Output Length: {result.config.random_output_len}")
        
        report.append("\n## 2. Functional Verification Results")
        report.append("| Test Case | Expected | Status |")
        report.append("|-----------|----------|--------|")
        for test in result.functional_tests:
            status = "PASS" if test["passed"] else "FAIL"
            report.append(f"| {test['name']} | Pattern match | {status} |")
        
        report.append("\n## 3. Performance Metrics")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| TTFT Mean (ms) | {result.performance.ttft_mean:.2f} |")
        report.append(f"| TTFT Median (ms) | {result.performance.ttft_median:.2f} |")
        report.append(f"| TTFT P99 (ms) | {result.performance.ttft_p99:.2f} |")
        report.append(f"| TPOT Mean (ms) | {result.performance.tpot_mean:.2f} |")
        report.append(f"| TPOT Median (ms) | {result.performance.tpot_median:.2f} |")
        report.append(f"| TPOT P99 (ms) | {result.performance.tpot_p99:.2f} |")
        report.append(f"| Throughput (tok/s) | {result.performance.throughput:.2f} |")
        report.append(f"| ITL Mean (ms) | {result.performance.itl_mean:.2f} |")
        report.append(f"| ITL Median (ms) | {result.performance.itl_median:.2f} |")
        report.append(f"| ITL P99 (ms) | {result.performance.itl_p99:.2f} |")
        report.append(f"| Total Requests | {result.performance.total_requests} |")
        report.append(f"| Successful Requests | {result.performance.successful_requests} |")
        report.append(f"| Failed Requests | {result.performance.failed_requests} |")
        report.append(f"| Total Time (s) | {result.performance.total_time:.2f} |")
        
        if result.accuracy:
            report.append("\n## 4. Accuracy Evaluation Results")
            report.append("| Benchmark | Score | Questions |")
            report.append("|-----------|-------|-----------|")
            report.append(f"| {result.accuracy.benchmark_name} | {result.accuracy.accuracy:.4f} | {result.accuracy.total_questions} |")
        
        if result.errors:
            report.append("\n## 5. Errors Encountered")
            for error in result.errors:
                report.append(f"- {error}")
        
        report.append("\n## 6. Test Conclusion")
        pass_rate = sum(1 for t in result.functional_tests if t["passed"]) / len(result.functional_tests) if result.functional_tests else 0
        overall_status = "PASS" if pass_rate == 1.0 and result.performance.failed_requests == 0 else "PARTIAL PASS" if pass_rate >= 0.5 else "FAIL"
        report.append(f"- Overall Status: {overall_status}")
        report.append(f"- Functional Test Pass Rate: {pass_rate*100:.1f}%")
        report.append(f"- Request Success Rate: {result.performance.successful_requests/result.performance.total_requests*100:.1f}%" if result.performance.total_requests > 0 else "- Request Success Rate: N/A")
        
        return "\n".join(report)
    
    def save_results(self, result: TestResult):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = self.results_dir / f"benchmark_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "config": asdict(result.config),
                "performance": asdict(result.performance),
                "accuracy": asdict(result.accuracy) if result.accuracy else None,
                "functional_tests": result.functional_tests,
                "errors": result.errors,
                "timestamp": result.timestamp
            }, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        report_path = self.results_dir / f"report_{timestamp}.md"
        report = self.generate_report(result)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    def run_all(self) -> TestResult:
        if not self.wait_for_server():
            raise RuntimeError("Server not ready")
        
        result = TestResult(config=self.config)
        
        result.functional_tests = self.run_functional_tests()
        
        result.performance = asyncio.run(self.run_performance_benchmark())
        
        result.accuracy = self.run_accuracy_test()
        
        result.errors = [
            test["error"] for test in result.functional_tests 
            if test.get("error")
        ]
        
        self.save_results(result)
        
        print("\n" + self.generate_report(result))
        
        return result


def main():
    parser = argparse.ArgumentParser(description="NPU Model Benchmarking Script")
    
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=6688, help="Server port")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--max-concurrency", type=int, default=256, help="Max concurrent requests")
    parser.add_argument("--num-prompts", type=int, default=1024, help="Number of test prompts")
    parser.add_argument("--random-input-len", type=int, default=3500, help="Random input length")
    parser.add_argument("--random-output-len", type=int, default=1500, help="Random output length")
    parser.add_argument("--random-range-ratio", type=float, default=1.0, help="Random range ratio")
    parser.add_argument("--request-rate", type=float, default=0.0, help="Request rate (0 = unlimited)")
    parser.add_argument("--timeout", type=int, default=3600, help="Request timeout in seconds")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--dataset-name", default="random", help="Dataset name")
    parser.add_argument("--dataset-path", help="Path to custom dataset")
    parser.add_argument("--accuracy-test", action="store_true", help="Run accuracy test")
    parser.add_argument("--num-questions", type=int, default=200, help="Number of questions for accuracy test")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for accuracy test")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of shots for few-shot evaluation")
    parser.add_argument("--skip-functional", action="store_true", help="Skip functional tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        max_concurrency=args.max_concurrency,
        num_prompts=args.num_prompts,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        request_rate=args.request_rate,
        timeout=args.timeout,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        accuracy_test=args.accuracy_test,
        num_questions=args.num_questions,
        max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots,
    )
    
    runner = NPUBenchmarkRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
