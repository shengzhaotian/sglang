#!/usr/bin/env python3
"""
NPU Model Test Script
Tests NPU with TP, DP-Attention, EP parallelism and ACLGraph features.

Usage:
    python test_npu_model.py [--model-path MODEL_PATH] [--tp-size TP] [--dp-size DP] [--ep-size EP]
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import requests
from urllib.parse import urlparse

# Set PYTHONPATH for current repo
os.environ["PYTHONPATH"] = "/home/trae/testCode/sglang/python:" + os.environ.get("PYTHONPATH", "")

DEFAULT_MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
DEFAULT_URL = "http://127.0.0.1:30000"


def launch_server(model_path, tp_size, dp_size, ep_size, enable_dp_attention, enable_graph, port=30000):
    """Launch SGLang server with NPU configuration."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--mem-fraction-static", "0.8",
    ]

    # Add TP size
    if tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])

    # Add DP-Attention
    if enable_dp_attention and dp_size > 1:
        cmd.extend([
            "--enable-dp-attention",
            "--dp-size", str(dp_size),
        ])

    # Add EP size
    if ep_size > 1:
        cmd.extend(["--ep-size", str(ep_size)])

    # Enable/disable graph (ACLGraph on NPU)
    if not enable_graph:
        cmd.append("--disable-cuda-graph")
    else:
        cmd.extend(["--cuda-graph-max-bs", "32"])

    print(f"Launching server with command: {' '.join(cmd)}")
    print(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=os.environ,
    )

    return process


def wait_for_server(url, timeout=300):
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server is ready after {time.time() - start_time:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def test_inference(url, prompt="Hello, how are you?"):
    """Test basic inference."""
    print(f"Testing inference with prompt: {prompt}")
    response = requests.post(
        f"{url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 32,
            },
        },
        timeout=60,
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result.get('text', '')[:100]}...")
        return True
    else:
        print(f"Request failed with status: {response.status_code}")
        return False


def test_gsm8k(url, num_questions=10):
    """Test GSM8K accuracy with a small subset."""
    from types import SimpleNamespace

    try:
        from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
    except ImportError:
        print("Could not import GSM8K eval, skipping...")
        return None

    parsed_url = urlparse(url)
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=num_questions,
        max_new_tokens=256,
        parallel=8,
        host=f"http://{parsed_url.hostname}",
        port=int(parsed_url.port),
    )

    try:
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"GSM8K accuracy: {metrics['accuracy']:.2%}")
        return metrics
    except Exception as e:
        print(f"GSM8K test failed: {e}")
        return None


def run_test(config):
    """Run a single test configuration."""
    print("\n" + "=" * 60)
    print(f"Testing: TP={config['tp_size']}, DP={config['dp_size']}, EP={config['ep_size']}, "
          f"DP-Attention={config['enable_dp_attention']}, Graph={config['enable_graph']}")
    print("=" * 60)

    process = None
    try:
        process = launch_server(
            model_path=config["model_path"],
            tp_size=config["tp_size"],
            dp_size=config["dp_size"],
            ep_size=config["ep_size"],
            enable_dp_attention=config["enable_dp_attention"],
            enable_graph=config["enable_graph"],
        )

        # Wait for server to start
        if not wait_for_server(DEFAULT_URL, timeout=300):
            print("Server failed to start!")
            return False

        # Test basic inference
        if not test_inference(DEFAULT_URL):
            print("Basic inference test failed!")
            return False

        print("Test passed!")
        return True

    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False
    finally:
        if process:
            print("Stopping server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description="NPU Model Test")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                        help="Model path")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor parallelism size")
    parser.add_argument("--dp-size", type=int, default=2,
                        help="Data parallelism size for DP-Attention")
    parser.add_argument("--ep-size", type=int, default=1,
                        help="Expert parallelism size")
    parser.add_argument("--enable-dp-attention", action="store_true",
                        help="Enable DP-Attention")
    parser.add_argument("--disable-graph", action="store_true",
                        help="Disable ACLGraph/NPU Graph")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with minimal questions")

    args = parser.parse_args()

    config = {
        "model_path": args.model_path,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "ep_size": args.ep_size,
        "enable_dp_attention": args.enable_dp_attention,
        "enable_graph": not args.disable_graph,
    }

    success = run_test(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
